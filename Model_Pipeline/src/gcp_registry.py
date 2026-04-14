"""
EcoPulse GCP Artifact Registry Integration
===========================================
Bidirectional bridge between local model training (MLflow) and GCP Artifact Registry.

Models are uploaded as generic artifacts to the configured Artifact Registry
repository, with a rich metadata sidecar (MLflow run_id, metrics, git commit, etc.)
stored alongside each model file.  The ``production`` tag mechanism in Artifact
Registry is used to track the currently-deployed version per model name.

Architecture
------------
- **Upload / Download**: ``gcloud artifacts generic upload/download`` CLI commands
  are the primary transport layer — they handle auth, chunking, and retry internally.
- **Listing / Tagging**: the ``google-cloud-artifact-registry`` Python client
  (v1 API) is used for metadata operations (list versions, get/set tags).
- **Auth**: service account key file (GOOGLE_APPLICATION_CREDENTIALS) in local/dev;
  Workload Identity / ADC in CI/CD.  Falls back automatically.
- **Resilience**: transient GCP errors are retried with exponential backoff.
  All push operations fail *gracefully* — training is never aborted by a GCP error.

Environment variables
---------------------
    GCP_PROJECT_ID              GCP project (default: ecopulse-mlops-pratham)
    GCP_REGISTRY_LOCATION       Region           (default: us-central1)
    GCP_REGISTRY_REPO           Repository name  (default: ecopulse-models-generic)
    GOOGLE_APPLICATION_CREDENTIALS
                                Path to service-account JSON key.
                                Falls back to ~/ecopulse-sa-key.json, then ADC.

Usage example
-------------
::

    from gcp_registry import push_after_mlflow_log, get_production_model, make_version_string
    import mlflow, joblib

    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(model, "xgboost_1h")
        joblib.dump(model, "models/xgboost_1h.joblib")

        push_after_mlflow_log(
            model_path="models/xgboost_1h.joblib",
            model_name="xgboost_1h",
            version=make_version_string("xgboost", 1),
            mlflow_run_id=run.info.run_id,
            horizon=1,
            model_type="xgboost",
            metrics={"mae": 25.1, "rmse": 33.4, "r2": 0.90},
            performance_tier="excellent",
            auto_promote=True,
        )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ecopulse-model.gcp_registry")

# ============================================================
# CONFIGURATION  (env-var overridable)
# ============================================================
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID",       "ecopulse-mlops-pratham")
GCP_REGISTRY_LOCATION = os.getenv("GCP_REGISTRY_LOCATION", "us-central1")
GCP_REGISTRY_REPO    = os.getenv("GCP_REGISTRY_REPO",    "ecopulse-models-generic")

# Default SA key location — CI/CD uses ADC, local dev uses this file
_DEFAULT_SA_KEY = os.path.expanduser("~/ecopulse-sa-key.json")

# Mutable tag that always points to the current production version
_PRODUCTION_TAG = "production"

# Retry parameters for transient GCP errors
_MAX_RETRIES  = 3
_RETRY_DELAY  = 2.0  # seconds (multiplied by attempt number)


# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

def _sa_key_path() -> str:
    """Return the resolved service-account key path from env or default."""
    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS", _DEFAULT_SA_KEY)


def _get_credentials():
    """
    Build GCP credentials for the Python SDK.

    Resolution order:
    1. Service-account key file (GOOGLE_APPLICATION_CREDENTIALS env var,
       or ~/ecopulse-sa-key.json).
    2. Application Default Credentials (``gcloud auth application-default login``
       or Workload Identity in CI/CD).

    Returns:
        A ``google.oauth2.credentials.Credentials`` object.

    Raises:
        ImportError:  If ``google-auth`` is not installed.
        RuntimeError: If neither key file nor ADC are available.
    """
    try:
        from google.oauth2 import service_account as _sa
        from google.auth import default as _adc
    except ImportError as exc:
        raise ImportError(
            "google-auth is required for GCP registry integration. "
            "Install with:  pip install google-auth google-auth-httplib2"
        ) from exc

    key_path = _sa_key_path()
    if os.path.isfile(key_path):
        logger.debug("GCP auth: using service-account key at %s", key_path)
        return _sa.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    # Fallback: Application Default Credentials (CI/CD Workload Identity, etc.)
    try:
        creds, _ = _adc(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        logger.debug("GCP auth: using Application Default Credentials")
        return creds
    except Exception as exc:
        raise RuntimeError(
            f"GCP authentication failed.  Neither a service-account key at "
            f"'{key_path}' nor ADC are available.\n"
            "• Local:  set GOOGLE_APPLICATION_CREDENTIALS to your key file path\n"
            "• CI/CD:  configure Workload Identity or set GOOGLE_APPLICATION_CREDENTIALS"
        ) from exc


def _get_ar_client():
    """
    Return an authenticated Artifact Registry v1 client.

    Raises:
        ImportError: If ``google-cloud-artifact-registry`` is not installed.
    """
    try:
        from google.cloud import artifactregistry_v1
    except ImportError as exc:
        raise ImportError(
            "google-cloud-artifact-registry is required for listing/tagging. "
            "Install with:  pip install google-cloud-artifact-registry"
        ) from exc

    creds = _get_credentials()
    return artifactregistry_v1.ArtifactRegistryClient(credentials=creds)


# ============================================================
# RETRY DECORATOR
# ============================================================

def _with_retry(max_retries: int = _MAX_RETRIES, delay: float = _RETRY_DELAY):
    """
    Decorator that retries a function on transient GCP / network errors.

    Transient errors (HTTP 429, 503, connection timeouts) are retried up to
    ``max_retries`` times with linear backoff (``delay * attempt`` seconds).
    Non-transient errors are re-raised immediately without retry.

    Args:
        max_retries: Maximum number of retry attempts after the first failure.
        delay:       Base delay in seconds between attempts.
    """
    _TRANSIENT_SIGNALS = ("503", "429", "timeout", "connection", "temporary",
                          "unavailable", "reset by peer")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 2):  # +2: first try + retries
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    err = str(exc).lower()
                    is_transient = any(s in err for s in _TRANSIENT_SIGNALS)
                    if is_transient and attempt <= max_retries:
                        last_exc = exc
                        wait = delay * attempt
                        logger.warning(
                            "%s attempt %d/%d failed (%s: %s). Retrying in %.1fs…",
                            func.__name__, attempt, max_retries + 1,
                            type(exc).__name__, exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        raise
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ============================================================
# RESOURCE PATH HELPERS
# ============================================================

def _package_name(model_name: str) -> str:
    """
    Convert a model name to an Artifact Registry package name.

    Artifact Registry package names must be lowercase and may contain only
    letters, digits, and hyphens.

    Examples::
        "xgboost_1h"       → "ecopulse-xgboost-1h"
        "xgboost_tuned_6h" → "ecopulse-xgboost-tuned-6h"
    """
    return f"ecopulse-{model_name.replace('_', '-').lower()}"


def _ar_parent() -> str:
    """Return the fully-qualified Artifact Registry repository resource path."""
    return (
        f"projects/{GCP_PROJECT_ID}"
        f"/locations/{GCP_REGISTRY_LOCATION}"
        f"/repositories/{GCP_REGISTRY_REPO}"
    )


def _ar_package_path(model_name: str) -> str:
    return f"{_ar_parent()}/packages/{_package_name(model_name)}"


def _ar_version_path(model_name: str, version: str) -> str:
    return f"{_ar_package_path(model_name)}/versions/{version}"


# ============================================================
# VALIDATION & FILE UTILITIES
# ============================================================

def _validate_model_path(model_path: str) -> Path:
    """
    Validate that a model file exists and is a regular file.

    Returns:
        ``pathlib.Path`` object for the model file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the path points to a directory.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Ensure the model has been trained and saved (via save_model() "
            "or joblib.dump()) before pushing to GCP Artifact Registry."
        )
    if not p.is_file():
        raise ValueError(
            f"model_path must be a regular file, not a directory: {model_path}"
        )
    return p


def _compute_md5(file_path: str) -> str:
    """Compute the MD5 hex-digest of a file for upload integrity verification."""
    h = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# LOW-LEVEL gcloud TRANSPORT
# ============================================================

def _run_gcloud(
    args: List[str],
    timeout: int = 300,
    operation: str = "gcloud",
) -> subprocess.CompletedProcess:
    """
    Execute a ``gcloud`` command, injecting auth credentials from the env.

    Args:
        args:      Argument list starting *after* ``gcloud`` (e.g. ``["artifacts", ...]``).
        timeout:   Subprocess timeout in seconds.
        operation: Human-readable label for error messages.

    Returns:
        Completed subprocess result.

    Raises:
        RuntimeError: If the command exits non-zero or gcloud is not found.
    """
    cmd = ["gcloud"] + args
    env = os.environ.copy()

    key = _sa_key_path()
    if os.path.isfile(key):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = key

    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gcloud CLI not found.  Install the Google Cloud SDK:\n"
            "  https://cloud.google.com/sdk/docs/install"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{operation} timed out after {timeout}s. "
            "The model file may be very large — increase _RETRY_DELAY or check connectivity."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"{operation} failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


# ============================================================
# PUSH MODEL TO REGISTRY
# ============================================================

@_with_retry()
def push_model_to_registry(
    model_path: str,
    model_name: str,
    version: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Upload a trained model file to GCP Artifact Registry as a generic artifact.

    The model file and a companion ``*_metadata.json`` sidecar are uploaded
    together under the same package + version.  The sidecar captures full
    provenance (MLflow run_id, metrics, git commit, training timestamp, etc.)
    so the artifact is self-describing without requiring MLflow access.

    Args:
        model_path:  Absolute or relative path to the model file
                     (.joblib, .pkl, etc.).
        model_name:  Logical model identifier used as the Artifact Registry
                     *package* name (e.g. ``"xgboost_1h"``, ``"lightgbm_24h"``).
        version:     Unique version string for this upload.
                     Use :func:`make_version_string` for a standardised format.
        metadata:    Provenance dict stored in the sidecar JSON.  Recommended keys:

                     - ``mlflow_run_id``      (str)   — MLflow run identifier
                     - ``training_timestamp`` (str)   — ISO-8601 UTC timestamp
                     - ``horizon``            (int)   — forecast horizon in hours
                     - ``model_type``         (str)   — ``"xgboost"`` / ``"lightgbm"``
                     - ``metrics``            (dict)  — ``{"mae": …, "rmse": …, "r2": …}``
                     - ``git_commit``         (str)   — short SHA
                     - ``performance_tier``   (str)   — ``"excellent"`` / ``"good"`` / …

    Returns:
        Dict containing::

            {
                "model_name":       str,   # e.g. "xgboost_1h"
                "version":          str,   # version string uploaded
                "package":          str,   # Artifact Registry package name
                "package_path":     str,   # full resource path
                "upload_timestamp": str,   # ISO-8601 UTC
                "file_size_mb":     float,
                "md5_checksum":     str,
            }

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError:      If authentication fails or the upload fails after
                           all retries.
        ImportError:       If required GCP packages are not installed.

    Example::

        result = push_model_to_registry(
            model_path="models/xgboost_1h.joblib",
            model_name="xgboost_1h",
            version=make_version_string("xgboost", 1),
            metadata={
                "mlflow_run_id":   "abc123ef",
                "horizon":         1,
                "model_type":      "xgboost",
                "metrics":         {"mae": 25.1, "rmse": 33.4, "r2": 0.90},
                "git_commit":      "a1b2c3d",
                "performance_tier":"excellent",
            },
        )
        print(result["upload_timestamp"])
    """
    model_file   = _validate_model_path(model_path)
    package      = _package_name(model_name)
    file_size_mb = model_file.stat().st_size / (1024 ** 2)
    md5          = _compute_md5(str(model_file))
    upload_ts    = datetime.utcnow().isoformat() + "Z"

    # Enrich metadata with upload-time provenance
    enriched_meta: Dict[str, Any] = {
        **metadata,
        "model_name":       model_name,
        "version":          version,
        "package":          package,
        "file_name":        model_file.name,
        "file_size_mb":     round(file_size_mb, 3),
        "md5_checksum":     md5,
        "upload_timestamp": upload_ts,
        "gcp_project":      GCP_PROJECT_ID,
        "gcp_location":     GCP_REGISTRY_LOCATION,
        "gcp_repository":   GCP_REGISTRY_REPO,
    }

    logger.info(
        "Pushing %s v%s to Artifact Registry (%s / %s / %s) — %.1f MB…",
        model_name, version,
        GCP_PROJECT_ID, GCP_REGISTRY_LOCATION, GCP_REGISTRY_REPO,
        file_size_mb,
    )

    # -- Upload model file ---------------------------------------------------
    _run_gcloud(
        [
            "artifacts", "generic", "upload",
            f"--project={GCP_PROJECT_ID}",
            f"--repository={GCP_REGISTRY_REPO}",
            f"--location={GCP_REGISTRY_LOCATION}",
            f"--package={package}",
            f"--version={version}",
            f"--source={model_file}",
        ],
        timeout=300,
        operation=f"upload model '{model_name}' v{version}",
    )

    # -- Upload metadata sidecar ---------------------------------------------
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{model_name}_metadata_",
        delete=False,
    ) as tmp:
        json.dump(enriched_meta, tmp, indent=2, default=str)
        meta_tmp_path = tmp.name

    try:
        _run_gcloud(
            [
                "artifacts", "generic", "upload",
                f"--project={GCP_PROJECT_ID}",
                f"--repository={GCP_REGISTRY_REPO}",
                f"--location={GCP_REGISTRY_LOCATION}",
                f"--package={package}",
                f"--version={version}",
                f"--source={meta_tmp_path}",
                f"--name={model_file.stem}_metadata.json",
            ],
            timeout=60,
            operation=f"upload metadata for '{model_name}' v{version}",
        )
    finally:
        os.unlink(meta_tmp_path)

    result: Dict[str, Any] = {
        "model_name":       model_name,
        "version":          version,
        "package":          package,
        "package_path":     _ar_package_path(model_name),
        "upload_timestamp": upload_ts,
        "file_size_mb":     round(file_size_mb, 3),
        "md5_checksum":     md5,
    }
    logger.info(
        "Successfully pushed %s v%s to GCP Artifact Registry (package: %s).",
        model_name, version, package,
    )
    return result


# ============================================================
# PULL MODEL FROM REGISTRY
# ============================================================

@_with_retry()
def pull_model_from_registry(
    model_name: str,
    version: str,
    destination_dir: Optional[str] = None,
) -> str:
    """
    Download a specific model version from GCP Artifact Registry.

    Both the model file and the metadata sidecar are downloaded.  The path
    to the primary model file (the non-JSON artifact) is returned.

    Args:
        model_name:      Logical model name (e.g. ``"xgboost_1h"``).
        version:         Exact version string to download.
        destination_dir: Local directory to save the downloaded artifacts.
                         Defaults to ``Model_Pipeline/models/``.

    Returns:
        Absolute path (str) to the downloaded model file.

    Raises:
        FileNotFoundError: If the version does not exist in the registry,
                           or if no model file is found after download.
        RuntimeError:      If the gcloud download command fails.

    Example::

        path = pull_model_from_registry("xgboost_1h", "20240315_1423_xgboost_1h")
        model = joblib.load(path)
        predictions = model.predict(X_test)
    """
    if destination_dir is None:
        _src = os.path.dirname(os.path.abspath(__file__))
        destination_dir = os.path.join(os.path.dirname(_src), "models")

    os.makedirs(destination_dir, exist_ok=True)
    package = _package_name(model_name)

    logger.info(
        "Pulling %s v%s from Artifact Registry → %s…",
        model_name, version, destination_dir,
    )

    _run_gcloud(
        [
            "artifacts", "generic", "download",
            f"--project={GCP_PROJECT_ID}",
            f"--repository={GCP_REGISTRY_REPO}",
            f"--location={GCP_REGISTRY_LOCATION}",
            f"--package={package}",
            f"--version={version}",
            f"--destination={destination_dir}",
        ],
        timeout=300,
        operation=f"download '{model_name}' v{version}",
    )

    # Return the newest non-JSON file in the destination directory
    candidates = [
        f for f in Path(destination_dir).iterdir()
        if f.is_file() and f.suffix != ".json"
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in '{destination_dir}' after downloading "
            f"{model_name} v{version}.  Check the repository for the expected artifact."
        )

    model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
    logger.info("Downloaded %s v%s → %s", model_name, version, model_path)
    return model_path


# ============================================================
# LIST MODEL VERSIONS
# ============================================================

def list_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """
    List all available versions for a model in Artifact Registry.

    Each entry includes the version string, creation/update timestamps, and
    any Artifact Registry tags pointing to that version (e.g. ``"production"``).

    Args:
        model_name: Logical model name (e.g. ``"xgboost_1h"``).

    Returns:
        List of dicts sorted by creation time (newest first).  Each dict::

            {
                "version":     str,            # e.g. "20240315_1423_xgboost_1h"
                "create_time": str | None,     # ISO-8601 UTC
                "update_time": str | None,     # ISO-8601 UTC
                "tags":        List[str],      # e.g. ["production"]
                "resource":    str,            # full AR resource path
            }

        Returns ``[]`` if the model has not been pushed yet.

    Raises:
        ImportError: If ``google-cloud-artifact-registry`` is not installed.
        RuntimeError: On non-recoverable GCP errors.

    Example::

        for v in list_model_versions("xgboost_1h"):
            print(v["version"], v["tags"])
    """
    from google.cloud import artifactregistry_v1  # imported here to defer ImportError

    client       = _get_ar_client()
    package_path = _ar_package_path(model_name)

    try:
        raw_versions = list(
            client.list_versions(
                request=artifactregistry_v1.ListVersionsRequest(parent=package_path)
            )
        )
    except Exception as exc:
        err = str(exc).lower()
        if "not found" in err or "404" in err:
            logger.info(
                "No versions found for '%s' — package may not exist yet.", model_name
            )
            return []
        logger.error("Failed to list versions for '%s': %s", model_name, exc)
        raise

    # Fetch all tags once (more efficient than one request per version)
    try:
        all_tags = list(
            client.list_tags(
                request=artifactregistry_v1.ListTagsRequest(parent=package_path)
            )
        )
    except Exception:
        all_tags = []

    # Build a mapping: version_resource_name → [tag_names]
    tag_map: Dict[str, List[str]] = {}
    for tag in all_tags:
        ver_key = tag.version  # full resource path
        tag_map.setdefault(ver_key, []).append(tag.name.split("/")[-1])

    results = []
    for ver in raw_versions:
        create_t = ver.create_time.isoformat() if ver.create_time else None
        update_t = ver.update_time.isoformat() if ver.update_time else None
        results.append({
            "version":     ver.name.split("/")[-1],
            "create_time": create_t,
            "update_time": update_t,
            "tags":        tag_map.get(ver.name, []),
            "resource":    ver.name,
        })

    results.sort(key=lambda v: v["create_time"] or "", reverse=True)
    return results


# ============================================================
# GET LATEST VERSION
# ============================================================

def get_latest_version(model_name: str) -> Optional[str]:
    """
    Return the most recently created version string for a model.

    Convenience wrapper around :func:`list_model_versions` that extracts the
    version string of the newest entry.

    Args:
        model_name: Logical model name (e.g. ``"xgboost_1h"``).

    Returns:
        Version string (str) of the latest upload, or ``None`` if no versions
        exist for this model.

    Example::

        ver = get_latest_version("lightgbm_24h")
        if ver:
            path = pull_model_from_registry("lightgbm_24h", ver)
    """
    versions = list_model_versions(model_name)
    if not versions:
        logger.info("No versions found for '%s'.", model_name)
        return None
    latest = versions[0]["version"]
    logger.info("Latest version of '%s': %s", model_name, latest)
    return latest


# ============================================================
# PROMOTE MODEL TO PRODUCTION
# ============================================================

def promote_model_to_production(
    model_name: str,
    version: str,
) -> Dict[str, Any]:
    """
    Tag a specific model version as production-ready in Artifact Registry.

    Artifact Registry *tags* are mutable pointers to versions.  This function
    atomically moves the ``production`` tag to the requested version, archiving
    the previous production version in the process (the old version is not
    deleted — only its ``production`` tag is reassigned).

    Args:
        model_name: Logical model name (e.g. ``"xgboost_1h"``).
        version:    Version string to promote (must already exist in the registry).

    Returns:
        Dict containing::

            {
                "model_name":                  str,
                "version":                     str,   # newly promoted
                "tag":                         str,   # "production"
                "promoted_at":                 str,   # ISO-8601 UTC
                "previous_production_version": str | None,
            }

    Raises:
        ValueError:  If ``version`` does not exist in the registry.
        RuntimeError: If the Artifact Registry tag operation fails.
        ImportError:  If ``google-cloud-artifact-registry`` is not installed.

    Example::

        result = promote_model_to_production("xgboost_1h", "20240315_1423_xgboost_1h")
        print(f"Promoted {result['version']} (was {result['previous_production_version']})")
    """
    from google.cloud import artifactregistry_v1

    # Validate that the requested version exists
    existing = list_model_versions(model_name)
    existing_strs = [v["version"] for v in existing]
    if version not in existing_strs:
        raise ValueError(
            f"Version '{version}' not found for model '{model_name}'.\n"
            f"Available versions: {existing_strs}"
        )

    client            = _get_ar_client()
    package_path      = _ar_package_path(model_name)
    tag_resource_name = f"{package_path}/tags/{_PRODUCTION_TAG}"
    version_resource  = f"{package_path}/versions/{version}"
    promoted_at       = datetime.utcnow().isoformat() + "Z"
    previous_version: Optional[str] = None

    # Check whether the production tag already exists
    existing_tag = None
    try:
        existing_tag   = client.get_tag(name=tag_resource_name)
        previous_version = existing_tag.version.split("/")[-1] if existing_tag.version else None
    except Exception as exc:
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise  # Unexpected error — propagate

    if existing_tag is not None:
        # Move the tag to the new version
        existing_tag.version = version_resource
        client.update_tag(
            request=artifactregistry_v1.UpdateTagRequest(
                tag=existing_tag,
                update_mask={"paths": ["version"]},
            )
        )
        logger.info(
            "Moved 'production' tag from v%s → v%s for '%s'.",
            previous_version, version, model_name,
        )
    else:
        # First-ever promotion — create the tag
        client.create_tag(
            request=artifactregistry_v1.CreateTagRequest(
                parent=package_path,
                tag_id=_PRODUCTION_TAG,
                tag=artifactregistry_v1.Tag(
                    name=tag_resource_name,
                    version=version_resource,
                ),
            )
        )
        logger.info("Created 'production' tag on v%s for '%s'.", version, model_name)

    result = {
        "model_name":                  model_name,
        "version":                     version,
        "tag":                         _PRODUCTION_TAG,
        "promoted_at":                 promoted_at,
        "previous_production_version": previous_version,
    }
    logger.info("Model '%s' v%s is now production.", model_name, version)
    return result


# ============================================================
# GET PRODUCTION MODEL
# ============================================================

def get_production_model(
    model_name: str,
    destination_dir: Optional[str] = None,
) -> str:
    """
    Download the current production-tagged model version for a given model name.

    Resolves the ``production`` Artifact Registry tag to its target version,
    then delegates to :func:`pull_model_from_registry`.

    Args:
        model_name:      Logical model name (e.g. ``"xgboost_1h"``).
        destination_dir: Local directory to save the model.
                         Defaults to ``Model_Pipeline/models/``.

    Returns:
        Absolute path (str) to the downloaded model file.

    Raises:
        ValueError:        If no ``production`` tag exists for this model.
        FileNotFoundError: If the tagged version cannot be downloaded.
        ImportError:       If ``google-cloud-artifact-registry`` is not installed.

    Example::

        import joblib
        path = get_production_model("xgboost_1h")
        model = joblib.load(path)
        predictions = model.predict(X_test)
    """
    client            = _get_ar_client()
    package_path      = _ar_package_path(model_name)
    tag_resource_name = f"{package_path}/tags/{_PRODUCTION_TAG}"

    try:
        tag = client.get_tag(name=tag_resource_name)
    except Exception as exc:
        err = str(exc).lower()
        if "not found" in err or "404" in err:
            raise ValueError(
                f"No production version tagged for model '{model_name}'.\n"
                "Run promote_model_to_production() to tag a version as production."
            ) from exc
        raise

    production_version = tag.version.split("/")[-1]
    logger.info(
        "Production version for '%s': %s — downloading…",
        model_name, production_version,
    )
    return pull_model_from_registry(model_name, production_version, destination_dir)


# ============================================================
# MLFLOW INTEGRATION  — primary public entry point
# ============================================================

def push_after_mlflow_log(
    model_path: str,
    model_name: str,
    version: str,
    mlflow_run_id: str,
    horizon: int,
    model_type: str,
    metrics: Dict[str, float],
    performance_tier: str,
    auto_promote: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Push a model to GCP Artifact Registry immediately after MLflow logging.

    Designed to be called right after ``mlflow.xgboost.log_model()`` or
    ``mlflow.lightgbm.log_model()``.  Stores the MLflow ``run_id`` in the GCP
    metadata sidecar *and* writes GCP artifact coordinates back as MLflow run
    tags, creating a bidirectional link between the two tracking systems.

    This function **never raises** — if GCP is unavailable the error is logged
    as a warning and ``None`` is returned.  Training is not disrupted.

    Args:
        model_path:       Local path to the saved model file (e.g.
                          ``"models/xgboost_1h.joblib"``).
        model_name:       Logical model name (e.g. ``"xgboost_1h"``).
        version:          Version string; use :func:`make_version_string` for a
                          standardised format.
        mlflow_run_id:    Active MLflow run ID, typically obtained from
                          ``mlflow.active_run().info.run_id``.
        horizon:          Forecast horizon in hours: 1, 6, 12, or 24.
        model_type:       Model flavour: ``"xgboost"``, ``"lightgbm"``, or
                          ``"xgboost_tuned"``.
        metrics:          Evaluation metrics dict:
                          ``{"mae": float, "rmse": float, "r2": float}``.
        performance_tier: Quality tier string from
                          :func:`mlflow_config.get_performance_tier`:
                          ``"excellent"``, ``"good"``, ``"fair"``, or ``"poor"``.
        auto_promote:     When ``True`` *and* ``performance_tier`` is
                          ``"excellent"`` or ``"good"``, automatically move the
                          ``production`` tag to this version after upload.

    Returns:
        The result dict from :func:`push_model_to_registry` on success,
        or ``None`` if GCP is unavailable.

    Example (inside a training script)::

        with mlflow.start_run(run_name=run_name, nested=True) as run:
            # … train and evaluate …
            mlflow.xgboost.log_model(model, f"xgboost_{horizon}h")
            save_model(model, f"xgboost_{horizon}h")          # joblib to models/

            push_after_mlflow_log(
                model_path=os.path.join(MODELS_DIR, f"xgboost_{horizon}h.joblib"),
                model_name=f"xgboost_{horizon}h",
                version=make_version_string("xgboost", horizon),
                mlflow_run_id=run.info.run_id,
                horizon=horizon,
                model_type="xgboost",
                metrics=test_metrics,
                performance_tier=get_performance_tier(test_metrics["mae"], horizon),
                auto_promote=True,
            )
    """
    import mlflow as _mlflow

    tracking_uri = _mlflow.get_tracking_uri() or ""
    mlflow_url   = (
        f"{tracking_uri}/#/runs/{mlflow_run_id}" if tracking_uri else None
    )

    metadata: Dict[str, Any] = {
        "mlflow_run_id":       mlflow_run_id,
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_run_url":      mlflow_url,
        "training_timestamp":  datetime.utcnow().isoformat() + "Z",
        "horizon":             horizon,
        "model_type":          model_type,
        "metrics":             metrics,
        "git_commit":          _get_git_commit(),
        "performance_tier":    performance_tier,
    }

    try:
        result = push_model_to_registry(
            model_path=model_path,
            model_name=model_name,
            version=version,
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning(
            "GCP Artifact Registry push skipped for '%s' v%s: %s: %s\n"
            "Training results are safe in MLflow — continuing without GCP registry.",
            model_name, version, type(exc).__name__, exc,
        )
        return None

    # Write GCP coordinates back to the MLflow run as tags
    try:
        _mlflow.set_tags({
            "gcp.registry.package":   _package_name(model_name),
            "gcp.registry.version":   version,
            "gcp.registry.location":  GCP_REGISTRY_LOCATION,
            "gcp.registry.repo":      GCP_REGISTRY_REPO,
            "gcp.registry.project":   GCP_PROJECT_ID,
        })
    except Exception as tag_exc:
        logger.warning(
            "Could not write GCP tags back to MLflow run %s: %s",
            mlflow_run_id, tag_exc,
        )

    # Auto-promote high-quality models to production
    if auto_promote and performance_tier in ("excellent", "good"):
        try:
            promote_model_to_production(model_name, version)
        except Exception as promo_exc:
            logger.warning(
                "Auto-promotion to production skipped for '%s' v%s: %s",
                model_name, version, promo_exc,
            )

    return result


# ============================================================
# CONVENIENCE HELPERS
# ============================================================

def make_version_string(model_type: str, horizon: int) -> str:
    """
    Generate a deterministic, chronologically-sortable version string.

    Format::

        {YYYYMMDD}_{HHMM}_{model_type}_{horizon}h

    Examples::

        make_version_string("xgboost", 1)        # "20240315_1423_xgboost_1h"
        make_version_string("xgboost_tuned", 24) # "20240315_1423_xgboost_tuned_24h"

    Args:
        model_type: Model flavour string (e.g. ``"xgboost"``, ``"lightgbm"``).
        horizon:    Forecast horizon in hours (1, 6, 12, or 24).

    Returns:
        Version string suitable for :func:`push_model_to_registry`.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    return f"{ts}_{model_type}_{horizon}h"


def _get_git_commit() -> str:
    """Return the short git SHA of HEAD, or ``'unknown'`` if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

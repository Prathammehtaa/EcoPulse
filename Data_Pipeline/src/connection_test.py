import os
from google.cloud import storage

# This looks for the ID on the specific computer running the code
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
client = storage.Client(project=project_id)

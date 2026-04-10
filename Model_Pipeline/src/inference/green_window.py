"""
EcoPulse Inference Pipeline — green_window.py
===============================================
WHAT IT DOES:
    Takes a 24-hour carbon intensity forecast and identifies the
    "green windows" — periods when carbon intensity is low enough
    that running workloads would minimize emissions.

WHY WE NEED IT:
    This is the CORE BUSINESS LOGIC of EcoPulse. The model predicts
    numbers (287.3 gCO2/kWh). This file turns those numbers into
    actionable advice ("run your job between 2-5 AM").

ANALOGY:
    The model is like a weather forecaster saying "it'll be 72°F at 3 PM."
    This file is like the app saying "great time for a run! 🏃"

USAGE:
    from inference.green_window import GreenWindowDetector
    
    detector = GreenWindowDetector()
    windows = detector.find_green_windows(forecast_df)
    # → [{"start": "2025-11-15 02:00", "end": "05:00", "avg_intensity": 187.3}, ...]
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GreenWindow:
    """
    Represents a period of low carbon intensity.
    
    Think of it like a "happy hour" for the electricity grid —
    this is when it's cheapest (in carbon terms) to run your workload.
    """
    start_time: str              # When the window opens
    end_time: str                # When the window closes
    duration_hours: int          # How long the window lasts
    avg_intensity: float         # Average carbon intensity during window
    min_intensity: float         # Lowest point in the window
    max_intensity: float         # Highest point in the window


class GreenWindowDetector:
    """
    Identifies low-carbon windows from a carbon intensity forecast.
    
    HOW IT DECIDES WHAT'S "GREEN":
        Option 1 (default): Percentile-based threshold
            - The bottom 25% of predicted values are "green"
            - Adapts automatically to different grid zones
            - PJM (coal-heavy) might have threshold at 300 gCO2/kWh
            - PACW (hydro-heavy) might have threshold at 100 gCO2/kWh
        
        Option 2: Fixed threshold
            - User sets a hard limit like 200 gCO2/kWh
            - Useful for compliance: "we only run below 200"
    """

    def __init__(
        self,
        method: str = "percentile",
        percentile: float = 25.0,
        fixed_threshold: Optional[float] = None,
    ):
        """
        Args:
            method: "percentile" (adaptive) or "fixed" (hard limit)
            percentile: Bottom X% is green (default: 25th percentile)
            fixed_threshold: Hard gCO2/kWh limit (only used if method="fixed")
        """
        self.method = method
        self.percentile = percentile
        self.fixed_threshold = fixed_threshold

    def _compute_threshold(self, intensities: np.ndarray) -> float:
        """
        Compute the green/not-green threshold.
        
        Percentile method: if intensities are [100, 200, 300, 400],
        the 25th percentile is ~175. Anything below 175 is "green."
        """
        if self.method == "fixed" and self.fixed_threshold is not None:
            return self.fixed_threshold
        return np.percentile(intensities, self.percentile)

    def find_green_windows(
        self, forecast_df: pd.DataFrame
    ) -> Dict:
        """
        Find green windows in a forecast.
        
        THIS IS THE MAIN FUNCTION.
        
        Args:
            forecast_df: DataFrame with columns:
                - datetime: hourly timestamps
                - predicted_carbon_intensity: model predictions
                - zone (optional): grid zone
        
        Returns:
            Dict with:
                - threshold: the green/not-green cutoff value
                - windows: list of GreenWindow objects
                - summary: overall statistics
                - hourly: per-hour green/red classification
        """
        df = forecast_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        intensities = df["predicted_carbon_intensity"].values
        threshold = self._compute_threshold(intensities)

        # Classify each hour as green or red
        df["is_green"] = df["predicted_carbon_intensity"] <= threshold

        # Find contiguous green windows
        # When is_green changes from False→True, a new window starts
        # When it changes from True→False, the window ends
        windows = []
        in_window = False
        window_start = None
        window_rows = []

        for i, row in df.iterrows():
            if row["is_green"] and not in_window:
                # Window starts
                in_window = True
                window_start = row["datetime"]
                window_rows = [row]
            elif row["is_green"] and in_window:
                # Window continues
                window_rows.append(row)
            elif not row["is_green"] and in_window:
                # Window ends
                in_window = False
                window_intensities = [r["predicted_carbon_intensity"]
                                      for r in window_rows]
                windows.append(GreenWindow(
                    start_time=str(window_start),
                    end_time=str(window_rows[-1]["datetime"]),
                    duration_hours=len(window_rows),
                    avg_intensity=round(np.mean(window_intensities), 2),
                    min_intensity=round(np.min(window_intensities), 2),
                    max_intensity=round(np.max(window_intensities), 2),
                ))
                window_rows = []

        # Close any window that extends to the end of the forecast
        if in_window and window_rows:
            window_intensities = [r["predicted_carbon_intensity"]
                                  for r in window_rows]
            windows.append(GreenWindow(
                start_time=str(window_start),
                end_time=str(window_rows[-1]["datetime"]),
                duration_hours=len(window_rows),
                avg_intensity=round(np.mean(window_intensities), 2),
                min_intensity=round(np.min(window_intensities), 2),
                max_intensity=round(np.max(window_intensities), 2),
            ))

        # Summary statistics
        green_hours = int(df["is_green"].sum())
        total_hours = len(df)
        avg_green = round(float(df.loc[df["is_green"],
                          "predicted_carbon_intensity"].mean()), 2) if green_hours > 0 else None
        avg_red = round(float(df.loc[~df["is_green"],
                        "predicted_carbon_intensity"].mean()), 2) if green_hours < total_hours else None

        co2_savings_pct = None
        if avg_green is not None and avg_red is not None and avg_red > 0:
            co2_savings_pct = round(((avg_red - avg_green) / avg_red) * 100, 1)

        return {
            "threshold_gco2_kwh": round(threshold, 2),
            "green_hours": green_hours,
            "total_hours": total_hours,
            "green_pct": round(green_hours / total_hours * 100, 1),
            "avg_green_intensity": avg_green,
            "avg_red_intensity": avg_red,
            "co2_savings_pct": co2_savings_pct,
            "windows": [vars(w) for w in windows],
            "hourly": df[["datetime", "predicted_carbon_intensity",
                          "is_green"]].to_dict("records"),
        }


class WorkloadScheduler:
    """
    Given a workload and a forecast, find the optimal time to run it.
    
    WHAT IT DOES:
        "I have a 3-hour ML training job. I can wait up to 12 hours.
         When should I start it to minimize carbon emissions?"
    
    HOW IT WORKS:
        1. Get the 24h forecast (predicted carbon intensity per hour)
        2. Slide a window of size = runtime across the forecast
        3. Find the window with the lowest average carbon intensity
        4. Compare against running immediately
        5. Return the recommendation with CO₂ savings
    
    EXAMPLE:
        Forecast: [400, 380, 350, 320, 280, 250, 220, 200, 190, 210, ...]
        Runtime: 3 hours
        
        Sliding window averages:
            Start hour 0: avg(400, 380, 350) = 376.7
            Start hour 1: avg(380, 350, 320) = 350.0
            ...
            Start hour 7: avg(200, 190, 210) = 200.0  ← BEST
        
        Recommendation: "Wait 7 hours, save 47% CO₂"
    """

    def find_optimal_schedule(
        self,
        forecast: List[Dict],
        runtime_hours: int,
        flexibility_hours: Optional[int] = None,
        energy_kwh: float = 100.0,
    ) -> Dict:
        """
        Find the best time to run a workload.
        
        Args:
            forecast: List of {"datetime": ..., "predicted_carbon_intensity": ...}
                      (from GreenWindowDetector.find_green_windows()["hourly"])
            runtime_hours: How long the workload takes (e.g., 3 hours)
            flexibility_hours: How long we can wait (default: full forecast)
            energy_kwh: Energy the workload consumes (for CO₂ calculation)
        
        Returns:
            Dict with recommendation:
                - recommended_start: when to start
                - hours_to_wait: how long to defer
                - expected_intensity: avg intensity during recommended window
                - immediate_intensity: avg intensity if running now
                - co2_saved_kg: kg of CO₂ avoided by waiting
                - co2_savings_pct: percentage reduction
        """
        if not forecast:
            raise ValueError("Empty forecast — cannot schedule")

        intensities = [h["predicted_carbon_intensity"] for h in forecast]
        datetimes = [h["datetime"] for h in forecast]

        max_start = len(intensities) - runtime_hours
        if flexibility_hours is not None:
            max_start = min(max_start, flexibility_hours)

        if max_start < 0:
            raise ValueError(
                f"Forecast too short ({len(intensities)}h) for "
                f"runtime ({runtime_hours}h)"
            )

        # Slide the window and find lowest average intensity
        best_start = 0
        best_avg = float("inf")

        for start in range(max_start + 1):
            window = intensities[start: start + runtime_hours]
            avg = np.mean(window)
            if avg < best_avg:
                best_avg = avg
                best_start = start

        # Compare with immediate execution
        immediate_window = intensities[0: runtime_hours]
        immediate_avg = np.mean(immediate_window)

        # CO₂ calculation
        # gCO₂/kWh × kWh = gCO₂, convert to kg
        immediate_co2_kg = (immediate_avg * energy_kwh) / 1000
        optimal_co2_kg = (best_avg * energy_kwh) / 1000
        co2_saved_kg = immediate_co2_kg - optimal_co2_kg
        savings_pct = (co2_saved_kg / immediate_co2_kg * 100
                       if immediate_co2_kg > 0 else 0)

        return {
            "recommended_start": str(datetimes[best_start]),
            "hours_to_wait": best_start,
            "expected_intensity_gco2_kwh": round(best_avg, 2),
            "immediate_intensity_gco2_kwh": round(immediate_avg, 2),
            "runtime_hours": runtime_hours,
            "energy_kwh": energy_kwh,
            "immediate_co2_kg": round(immediate_co2_kg, 3),
            "optimal_co2_kg": round(optimal_co2_kg, 3),
            "co2_saved_kg": round(co2_saved_kg, 3),
            "co2_savings_pct": round(savings_pct, 1),
            "recommendation": (
                f"Wait {best_start} hours — start at {datetimes[best_start]}. "
                f"Save {co2_saved_kg:.1f} kg CO₂ ({savings_pct:.1f}% reduction)."
                if best_start > 0
                else "Run now — this is already the optimal window."
            ),
        }
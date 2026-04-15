import { useMemo, useState } from "react";
import SimpleChart from "../components/SimpleChart";
import { forecast24h, getZoneDisplayName, regions } from "../mockData";

function toneForValue(value) {
  if (value < 160) return "green";
  if (value < 230) return "amber";
  return "red";
}

export default function DashboardPage({ username, workloadHistory, onNavigate, onApproveRecommended }) {
  const [zone, setZone] = useState("US-MIDA-PJM");
  const forecast = forecast24h[zone];
  const current = forecast[0];
  const lowWindow = Math.min(...forecast);
  const lowWindowIndex = forecast.indexOf(lowWindow);
  const savedToday = workloadHistory.reduce((sum, item) => sum + item.co2SavedKg, 0);
  const averageDelay =
    workloadHistory.length > 0
      ? (
          workloadHistory.reduce((sum, item) => sum + (item.recommendedDelayHours ?? 0), 0) /
          workloadHistory.length
        ).toFixed(1)
      : "0.0";
  const bestRegion = regions.reduce((best, item) => (item.intensity < best.intensity ? item : best), regions[0]);

  const recommendedWindow = useMemo(() => {
    const start = Math.max(lowWindowIndex - 1, 0);
    const end = Math.min(lowWindowIndex + 2, 23);

    const formatHour = (hour) => `${String(hour % 24).padStart(2, "0")}:00`;

    return {
      start,
      end,
      label: `${formatHour(start)} - ${formatHour(end)}`
    };
  }, [lowWindowIndex]);

  const activeRegionName = getZoneDisplayName(zone);
  const recommendedStart = useMemo(() => {
    const next = new Date();
    next.setMinutes(0, 0, 0);
    next.setHours(next.getHours() + recommendedWindow.start);
    return next.toLocaleString();
  }, [recommendedWindow]);

  const placementIndexes = useMemo(() => {
    const points = [];
    for (let i = recommendedWindow.start; i <= recommendedWindow.end; i += 1) {
      points.push(i);
    }
    return points;
  }, [recommendedWindow]);
  const workloadWindows = useMemo(() => {
    const now = new Date();
    const horizonEnd = new Date(now.getTime() + 24 * 60 * 60 * 1000);

    return workloadHistory
      .filter((item) => item.zone === zone && item.recommendedStart && item.runtimeHours)
      .map((item) => {
        const start = new Date(item.recommendedStart);
        const end = new Date(start.getTime() + Number(item.runtimeHours || 0) * 60 * 60 * 1000);

        if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end <= now || start >= horizonEnd) {
          return null;
        }

        const startIndex = Math.max(0, (start.getTime() - now.getTime()) / (60 * 60 * 1000));
        const endIndex = Math.min(24, (end.getTime() - now.getTime()) / (60 * 60 * 1000));

        return {
          startIndex,
          endIndex,
          label: `${item.name} (${item.runtimeHours}h)`
        };
      })
      .filter(Boolean);
  }, [workloadHistory, zone]);

  const regionCards = [
    { zone: "US-MIDA-PJM", note: "Medium - consider deferring" },
    { zone: "US-NW-PACW", note: "Low carbon - good window" }
  ].map((item) => {
    const region = regions.find((entry) => entry.zone === item.zone);
    return {
      label: getZoneDisplayName(item.zone),
      value: `${region.intensity}`,
      unit: "gCO2/kWh",
      note: item.note,
      tone: toneForValue(region.intensity)
    };
  });

  const cards = [
    ...regionCards,
    {
      label: "CO2 SAVED TODAY",
      value: `${savedToday.toFixed(1)}`,
      unit: "kg",
      note: "Tracked from approved schedules",
      tone: "green"
    },
    {
      label: "AVERAGE DELAY",
      value: `${averageDelay}`,
      unit: "hrs",
      note: "Average approved wait time",
      tone: "blue"
    },
    {
      label: "1H FORECAST",
      value: `${forecast[1]}`,
      unit: "gCO2",
      note: "Stable - good window",
      tone: toneForValue(forecast[1])
    },
    {
      label: "6H FORECAST",
      value: `${forecast[5]}`,
      unit: "gCO2",
      note: "Rising - consider deferring",
      tone: toneForValue(forecast[5])
    },
    {
      label: "24H FORECAST",
      value: `${forecast[23]}`,
      unit: "gCO2",
      note: "Late-night recovery",
      tone: toneForValue(forecast[23])
    },
    {
      label: "CO2 SAVED TILL NOW",
      value: `${savedToday.toFixed(1)}`,
      unit: "kg",
      note: `${workloadHistory.length} workloads optimized`,
      tone: "green"
    }
  ];

  const approveRecommendedWindow = () => {
    const immediateIntensity = forecast[0];
    const expectedIntensity = lowWindow;
    const energyKwh = 120;
    const runtimeHours = Math.max(recommendedWindow.end - recommendedWindow.start, 1);
    const immediateCo2Kg = Number(((immediateIntensity * energyKwh * runtimeHours) / 1000).toFixed(3));
    const optimalCo2Kg = Number(((expectedIntensity * energyKwh * runtimeHours) / 1000).toFixed(3));
    const co2SavedKg = Number(Math.max(0, immediateCo2Kg - optimalCo2Kg).toFixed(3));

    onApproveRecommended({
      name: `Recommended window - ${getZoneDisplayName(zone)}`,
      zone,
      recommendedStart,
      energyKwh,
      priorityHours: recommendedWindow.start,
      runtimeHours,
      recommendedDelayHours: recommendedWindow.start,
      expectedIntensity,
      immediateIntensity,
      co2SavedKg
    });
  };

  return (
    <div className="page-section">
      <div className="dashboard-topline">
        <h1>Dashboard</h1>
      </div>

      <section className="hero-banner">
        <div className="hero-banner-main">
          <div className="hero-banner-copy">
            <span className="hero-kicker">Live carbon overview</span>
            <div className="hero-copy-topline">
              <div className="hero-copy-intro">
                <h2>Hello{username ? `, ${username}` : ""}!</h2>
                <p>
                  {activeRegionName} is showing a cleaner operating window right now. Review the recommendation below
                  to place flexible work at a lower-carbon time.
                </p>
              </div>
              <article className="hero-insight-card">
                <span>Best live region</span>
                <strong>{getZoneDisplayName(bestRegion.zone)}</strong>
                <small>{bestRegion.intensity} gCO2/kWh right now</small>
              </article>
            </div>
          </div>

          <div className="hero-best-window">
            <div className="hero-best-window-label">Recommended green window</div>
            <div className="hero-best-window-time">{recommendedWindow.label}</div>
            <div className="hero-best-window-actions">
              <button className="primary-action inline" onClick={approveRecommendedWindow}>Schedule Workload</button>
            </div>
          </div>
        </div>
      </section>

      <section className="card-grid">
        {cards.map((card) => (
          <article key={card.label} className={`stat-card tone-${card.tone}`}>
            <div className="stat-label">{card.label}</div>
            <div className="stat-value-row">
              <strong>{card.value}</strong>
              <span>{card.unit}</span>
            </div>
            <div className="stat-note">{card.note}</div>
          </article>
        ))}
      </section>

      <section className="forecast-panel">
        <div className="forecast-panel-header">
          <div>
            <h3>24-hour carbon intensity forecast - {getZoneDisplayName(zone)}</h3>
            <div className="forecast-panel-controls">
              <p>Updated 4 min ago</p>
              <select value={zone} onChange={(event) => setZone(event.target.value)} className="forecast-select">
                {regions.map((item) => (
                  <option key={item.zone} value={item.zone}>
                    {item.name}
                  </option>
                ))}
              </select>
            </div>
            <span className="forecast-tag">
              Green shading marks lower-carbon windows. Placement markers show the suggested run interval.
            </span>
          </div>

          <div className="forecast-now">
            <strong>{current}</strong>
            <span>gCO2/kWh now</span>
          </div>
        </div>

        <SimpleChart values={forecast} placementIndexes={placementIndexes} workloadWindows={workloadWindows} />

        <div className="forecast-summary">
          Lowest 24h intensity: <strong>{lowWindow} gCO2/kWh</strong>
        </div>
      </section>
    </div>
  );
}

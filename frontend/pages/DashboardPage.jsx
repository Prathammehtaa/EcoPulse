import { useEffect, useMemo, useState } from "react";
import SimpleChart from "../components/SimpleChart";
import { getRegions, getForecast } from "../api";

function toneForValue(value) {
  if (value < 160) return "green";
  if (value < 230) return "amber";
  return "red";
}

export default function DashboardPage({ username, workloadHistory, onNavigate, onApproveRecommended }) {
  const [zone, setZone] = useState("US-MIDA-PJM");
  const [regions, setRegions] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch regions on mount
  useEffect(() => {
    getRegions()
      .then(setRegions)
      .catch((err) => {
        console.error("Failed to load regions:", err);
        setError("Could not load region data.");
      });
  }, []);

  // Fetch forecast whenever zone changes
  useEffect(() => {
    if (!zone) return;
    setLoading(true);
    getForecast(zone, 6)
      .then((data) => {
        setForecast(data.values);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load forecast:", err);
        setError("Could not load forecast.");
        setLoading(false);
      });
  }, [zone]);

  const current       = forecast[0] ?? 0;
  const lowWindow     = forecast.length > 0 ? Math.min(...forecast) : 0;
  const lowWindowIndex = forecast.length > 0 ? forecast.indexOf(lowWindow) : 0;
  const savedToday    = workloadHistory.reduce((sum, item) => sum + item.co2SavedKg, 0);
  const averageDelay  =
    workloadHistory.length > 0
      ? (workloadHistory.reduce((sum, item) => sum + (item.recommendedDelayHours ?? 0), 0) /
          workloadHistory.length).toFixed(1)
      : "0.0";

  const bestRegion = regions.length > 0
    ? regions.reduce((best, item) => (item.intensity < best.intensity ? item : best), regions[0])
    : { zone: "—", intensity: 0 };

  const recommendedWindow = useMemo(() => {
    const start = Math.max(lowWindowIndex - 1, 0);
    const end   = Math.min(lowWindowIndex + 2, 23);
    const formatHour = (h) => `${String(h % 24).padStart(2, "0")}:00`;
    return { start, end, label: `Tonight ${formatHour(start)} - ${formatHour(end)}` };
  }, [lowWindowIndex]);

  const recommendedStart = useMemo(() => {
    const next = new Date();
    next.setMinutes(0, 0, 0);
    next.setHours(next.getHours() + recommendedWindow.start);
    return next.toLocaleString();
  }, [recommendedWindow]);

  const placementIndexes = useMemo(() => {
    const points = [];
    for (let i = recommendedWindow.start; i <= recommendedWindow.end; i++) points.push(i);
    return points;
  }, [recommendedWindow]);

  const regionCards = [
    { zone: "US-MIDA-PJM", note: "Medium - consider deferring" },
    { zone: "US-NW-PACW",  note: "Low carbon - good window" },
  ].map((item) => {
    const region = regions.find((r) => r.zone === item.zone);
    return {
      label: item.zone,
      value: region ? `${region.intensity}` : "—",
      unit: "gCO2/kWh",
      note: item.note,
      tone: region ? toneForValue(region.intensity) : "gray",
    };
  });

  const cards = [
    ...regionCards,
    { label: "CO2 SAVED TODAY",   value: `${savedToday.toFixed(1)}`, unit: "kg",  note: "Tracked from approved schedules", tone: "green" },
    { label: "AVERAGE DELAY",     value: `${averageDelay}`,           unit: "hrs", note: "Average approved wait time",       tone: "blue"  },
    { label: "1H FORECAST",       value: `${forecast[1] ?? "—"}`,     unit: "gCO2", note: "Stable - good window",           tone: toneForValue(forecast[1] ?? 0) },
    { label: "6H FORECAST",       value: `${forecast[5] ?? "—"}`,     unit: "gCO2", note: "Rising - consider deferring",    tone: toneForValue(forecast[5] ?? 0) },
    { label: "24H FORECAST",      value: `${forecast[23] ?? "—"}`,    unit: "gCO2", note: "Late-night recovery",            tone: toneForValue(forecast[23] ?? 0) },
    { label: "CO2 SAVED TILL NOW",value: `${savedToday.toFixed(1)}`,  unit: "kg",  note: `${workloadHistory.length} workloads optimized`, tone: "green" },
  ];

  const approveRecommendedWindow = () => {
    const immediateIntensity = forecast[0];
    const expectedIntensity  = lowWindow;
    const energyKwh          = 120;
    const runtimeHours       = Math.max(recommendedWindow.end - recommendedWindow.start, 1);
    const immediateCo2Kg     = Number(((immediateIntensity * energyKwh * runtimeHours) / 1000).toFixed(3));
    const optimalCo2Kg       = Number(((expectedIntensity  * energyKwh * runtimeHours) / 1000).toFixed(3));
    const co2SavedKg         = Number(Math.max(0, immediateCo2Kg - optimalCo2Kg).toFixed(3));
    onApproveRecommended({
      name: `Recommended window - ${zone}`, zone, recommendedStart,
      energyKwh, priorityHours: recommendedWindow.start, runtimeHours,
      recommendedDelayHours: recommendedWindow.start,
      expectedIntensity, immediateIntensity, co2SavedKg,
    });
  };

  if (error) return <div className="page-section"><p style={{ color: "red" }}>{error}</p></div>;

  return (
    <div className="page-section">
      <div className="dashboard-topline"><h1>Dashboard</h1></div>

      <section className="hero-banner">
        <div className="hero-banner-main">
          <div className="hero-banner-copy">
            <h2>Good afternoon, {username}</h2>
            <p>Grid carbon is lower in {zone}. This is a strong window to run flexible workloads.</p>
            <div className="hero-insight-grid">
              <article className="hero-insight-card">
                <span>Best live region</span>
                <strong>{bestRegion.zone}</strong>
                <small>{bestRegion.intensity} gCO2/kWh right now</small>
              </article>
              <article className="hero-insight-card">
                <span>Carbon-aware action</span>
                <strong>Shift flexible jobs tonight</strong>
                <small>Use cleaner windows to reduce emissions without changing workload size.</small>
              </article>
            </div>
          </div>
          <div className="hero-best-window">
            <div className="hero-best-window-label">Recommended green window</div>
            <div className="hero-best-window-time">{recommendedWindow.label}</div>
            <div className="hero-best-window-note">
              Low-carbon interval identified by the XGBoost model. Real predictions from your trained pipeline.
            </div>
            <div className="hero-best-window-meta">
              <span>CO2 saved <strong>24.8 kg</strong></span>
              <span>Delay <strong>{averageDelay}h</strong></span>
              <span>SLA risk <strong>None</strong></span>
            </div>
            <div className="hero-best-window-actions">
              <button className="primary-action inline" onClick={approveRecommendedWindow}>Approve</button>
              <button className="ghost-action danger" type="button" onClick={() => onNavigate("scheduler")}>Deny</button>
            </div>
          </div>
        </div>
      </section>

      <section className="card-grid">
        {cards.map((card) => (
          <article key={card.label} className={`stat-card tone-${card.tone}`}>
            <div className="stat-label">{card.label}</div>
            <div className="stat-value-row"><strong>{card.value}</strong><span>{card.unit}</span></div>
            <div className="stat-note">{card.note}</div>
          </article>
        ))}
      </section>

      <section className="forecast-panel">
        <div className="forecast-panel-header">
          <div>
            <h3>24-hour carbon intensity forecast - {zone}</h3>
            <div className="forecast-panel-controls">
              <p>{loading ? "Loading..." : "Live from XGBoost model"}</p>
              <select value={zone} onChange={(e) => setZone(e.target.value)} className="forecast-select">
                {regions.map((r) => (
                  <option key={r.zone} value={r.zone}>{r.zone}</option>
                ))}
              </select>
            </div>
            <span className="forecast-tag">
              Green shading marks low-carbon windows | Red shading marks high-carbon windows
            </span>
          </div>
          <div className="forecast-now">
            <strong>{current}</strong>
            <span>gCO2/kWh now</span>
          </div>
        </div>
        {!loading && forecast.length > 0 && (
          <SimpleChart values={forecast} placementIndexes={placementIndexes} />
        )}
        {loading && <p style={{ padding: "1rem", color: "#666" }}>Loading forecast...</p>}
        <div className="forecast-summary">
          Lowest 24h intensity: <strong>{lowWindow} gCO2/kWh</strong>
        </div>
      </section>
    </div>
  );
}

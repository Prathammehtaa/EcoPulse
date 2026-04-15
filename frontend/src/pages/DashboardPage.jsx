import { useEffect, useMemo, useState } from "react";
import SimpleChart from "../components/SimpleChart";
import { getRegions, getForecast } from "../api";

function toneForValue(value) {
  if (value < 160) return "green";
  if (value < 230) return "amber";
  return "red";
}

const ZONE_LABELS = {
  "US-MIDA-PJM": "Northern Virginia Region",
  "US-NW-PACW": "Portland Oregon Region",
};

export default function DashboardPage({ username, workloadHistory, onNavigate, onApproveRecommended }) {
  const [zone, setZone] = useState("US-MIDA-PJM");
  const [regions, setRegions] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date().toLocaleTimeString());

  const allowedZones = ["US-MIDA-PJM", "US-NW-PACW"];

  const fetchData = () => {
    getRegions().then(setRegions).catch(console.error);
    getForecast(zone, 1)
      .then((data) => {
        setForecast(data.values);
        setLoading(false);
        setLastUpdated(new Date().toLocaleTimeString());
      })
      .catch(console.error);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [zone]);

  const current = forecast[0] ?? 0;
  const lowWindow = forecast.length > 0 ? Math.min(...forecast) : 0;
  const lowWindowIndex = forecast.length > 0 ? forecast.indexOf(lowWindow) : 0;
  const savedToday = workloadHistory.reduce((sum, item) => sum + item.co2SavedKg, 0);
  const averageDelay =
    workloadHistory.length > 0
      ? (workloadHistory.reduce((sum, item) => sum + (item.recommendedDelayHours ?? 0), 0) / workloadHistory.length).toFixed(1)
      : "0.0";

  const filteredRegions = regions.filter((r) => allowedZones.includes(r.zone));
  const bestRegion = filteredRegions.length > 0
    ? filteredRegions.reduce((best, item) => (item.intensity < best.intensity ? item : best), filteredRegions[0])
    : { zone: "—", intensity: 0 };

  const recommendedWindow = useMemo(() => {
    const start = Math.max(lowWindowIndex - 1, 0);
    const end = Math.min(lowWindowIndex + 2, 23);
    const formatHour = (hour) => `${String(hour % 24).padStart(2, "0")}:00`;
    return { start, end, label: `${formatHour(start)} - ${formatHour(end)}` };
  }, [lowWindowIndex]);

  const recommendedStart = useMemo(() => {
    const next = new Date();
    next.setMinutes(0, 0, 0);
    next.setHours(next.getHours() + recommendedWindow.start);
    return next.toLocaleString();
  }, [recommendedWindow]);

  const placementIndexes = useMemo(() => {
    const points = [];
    for (let i = recommendedWindow.start; i <= recommendedWindow.end; i += 1) points.push(i);
    return points;
  }, [recommendedWindow]);

  const regionCards = [
    { zone: "US-MIDA-PJM", note: "Medium - consider deferring" },
    { zone: "US-NW-PACW", note: "Low carbon - good window" }
  ].map((item) => {
    const region = regions.find((r) => r.zone === item.zone);
    return {
      label: ZONE_LABELS[item.zone],
      value: region ? `${region.intensity}` : "—",
      unit: "gCO2/kWh",
      note: item.note,
      tone: region ? toneForValue(region.intensity) : "gray",
    };
  });

  const cards = [
    ...regionCards,
    { label: "CO2 SAVED TODAY", value: `${savedToday.toFixed(1)}`, unit: "kg", note: "Tracked from approved schedules", tone: "green" },
    { label: "AVERAGE DELAY", value: `${averageDelay}`, unit: "hrs", note: "Average approved wait time", tone: "blue" },
    { label: "1H FORECAST", value: `${forecast[1] ?? "—"}`, unit: "gCO2", note: "Stable - good window", tone: toneForValue(forecast[1] ?? 0) },
    { label: "12H FORECAST", value: `${forecast[11] ?? "—"}`, unit: "gCO2", note: "Rising - consider deferring", tone: toneForValue(forecast[11] ?? 0) },
    { label: "24H FORECAST", value: `${forecast[23] ?? "—"}`, unit: "gCO2", note: "Late-night recovery", tone: toneForValue(forecast[23] ?? 0) },
    { label: "CO2 SAVED TILL NOW", value: `${savedToday.toFixed(1)}`, unit: "kg", note: `${workloadHistory.length} workloads optimized`, tone: "green" },
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
      name: `Recommended window - ${ZONE_LABELS[zone]}`, zone, recommendedStart,
      energyKwh, priorityHours: recommendedWindow.start, runtimeHours,
      recommendedDelayHours: recommendedWindow.start,
      expectedIntensity, immediateIntensity, co2SavedKg,
    });
  };

  return (
    <div className="page-section">
      <div className="dashboard-topline"><h1>Dashboard</h1></div>

      <section className="hero-banner">
        <div className="hero-banner-main">
          <div className="hero-banner-copy">
            <h2>Good Afternoon, {username}</h2>
              <p>Grid carbon is lower in {ZONE_LABELS[zone]}. This is a strong window to run flexible workloads.</p>
                <div style={{marginTop: "12px", display: "flex", alignItems: "center", gap: "10px"}}>
              <span style={{color: "var(--green-300)", fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em"}}>Best live region</span>
            <span style={{color: "white", fontWeight: "700", fontSize: "1.1rem"}}>{ZONE_LABELS[bestRegion.zone] ?? bestRegion.zone}</span>
          <span style={{color: "#d6f5e8", fontSize: "0.9rem"}}>· {bestRegion.intensity} gCO2/kWh</span>
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
            <div className="stat-value-row"><strong>{card.value}</strong><span>{card.unit}</span></div>
            <div className="stat-note">{card.note}</div>
          </article>
        ))}
      </section>

      <section className="forecast-panel">
        <div className="forecast-panel-header">
          <div>
            <h3>24-hour carbon intensity forecast - {ZONE_LABELS[zone]}</h3>
            <div className="forecast-panel-controls">
              <p>{loading ? "Loading..." : `Updated at ${lastUpdated}`}</p>
              <select value={zone} onChange={(e) => setZone(e.target.value)} className="forecast-select">
                {filteredRegions.map((item) => (
                  <option key={item.zone} value={item.zone}>{ZONE_LABELS[item.zone]}</option>
                ))}
              </select>
            </div>
            <span className="forecast-tag">
              Green shading marks low-carbon windows | Red shading marks high-carbon windows | Placement markers show suggested run slots
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
        {loading && <p style={{ padding: "1rem", color: "#9fe1cb" }}>Loading forecast...</p>}
        <div className="forecast-summary">
          Lowest 24h intensity: <strong>{lowWindow} gCO2/kWh</strong>
        </div>
      </section>
    </div>
  );
}
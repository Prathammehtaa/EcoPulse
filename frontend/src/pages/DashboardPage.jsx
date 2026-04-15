import { useEffect, useMemo, useState } from "react";
import SimpleChart from "../components/SimpleChart";
<<<<<<< HEAD
import { forecast24h, getZoneDisplayName, regions } from "../mockData";
=======
import { getRegions, getForecast } from "../api";
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80

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
<<<<<<< HEAD

    return {
      start,
      end,
      label: `${formatHour(start)} - ${formatHour(end)}`
    };
=======
    return { start, end, label: `Tonight ${formatHour(start)} - ${formatHour(end)}` };
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
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
    for (let i = recommendedWindow.start; i <= recommendedWindow.end; i += 1) points.push(i);
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
    const region = regions.find((r) => r.zone === item.zone);
    return {
<<<<<<< HEAD
      label: getZoneDisplayName(item.zone),
      value: `${region.intensity}`,
=======
      label: ZONE_LABELS[item.zone],
      value: region ? `${region.intensity}` : "—",
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
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
<<<<<<< HEAD
      name: `Recommended window - ${getZoneDisplayName(zone)}`,
      zone,
      recommendedStart,
      energyKwh,
      priorityHours: recommendedWindow.start,
      runtimeHours,
=======
      name: `Recommended window - ${ZONE_LABELS[zone]}`, zone, recommendedStart,
      energyKwh, priorityHours: recommendedWindow.start, runtimeHours,
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
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
<<<<<<< HEAD
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
=======
            <h2>Good afternoon, {username}</h2>
            <p>Grid carbon is lower in {ZONE_LABELS[zone]}. This is a strong window to run flexible workloads.</p>
            <div className="hero-insight-grid">
              <article className="hero-insight-card">
                <span>Best live region</span>
                <strong>{ZONE_LABELS[bestRegion.zone] ?? bestRegion.zone}</strong>
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
                <small>{bestRegion.intensity} gCO2/kWh right now</small>
              </article>
            </div>
          </div>
          <div className="hero-best-window">
            <div className="hero-best-window-label">Recommended green window</div>
            <div className="hero-best-window-time">{recommendedWindow.label}</div>
<<<<<<< HEAD
=======
            <div className="hero-best-window-note">
              Low-carbon interval identified by the XGBoost model.
            </div>
            <div className="hero-best-window-meta">
              <span>CO2 saved <strong>24.8 kg</strong></span>
              <span>Delay <strong>{averageDelay}h</strong></span>
              <span>SLA risk <strong>None</strong></span>
            </div>
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
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
<<<<<<< HEAD
            <h3>24-hour carbon intensity forecast - {getZoneDisplayName(zone)}</h3>
            <div className="forecast-panel-controls">
              <p>Updated 4 min ago</p>
              <select value={zone} onChange={(event) => setZone(event.target.value)} className="forecast-select">
                {regions.map((item) => (
                  <option key={item.zone} value={item.zone}>
                    {item.name}
                  </option>
=======
            <h3>24-hour carbon intensity forecast - {ZONE_LABELS[zone]}</h3>
            <div className="forecast-panel-controls">
              <p>{loading ? "Loading..." : `Updated at ${lastUpdated}`}</p>
              <select value={zone} onChange={(e) => setZone(e.target.value)} className="forecast-select">
                {filteredRegions.map((item) => (
                  <option key={item.zone} value={item.zone}>{ZONE_LABELS[item.zone]}</option>
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
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
<<<<<<< HEAD

        <SimpleChart values={forecast} placementIndexes={placementIndexes} workloadWindows={workloadWindows} />

=======
        {!loading && forecast.length > 0 && (
          <SimpleChart values={forecast} placementIndexes={placementIndexes} />
        )}
        {loading && <p style={{ padding: "1rem", color: "#9fe1cb" }}>Loading forecast...</p>}
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
        <div className="forecast-summary">
          Lowest 24h intensity: <strong>{lowWindow} gCO2/kWh</strong>
        </div>
      </section>
    </div>
  );
}
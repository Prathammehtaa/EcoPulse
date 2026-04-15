<<<<<<< HEAD
<<<<<<< HEAD
import { useMemo, useState } from "react";
import { forecast24h, getZoneDisplayName, regions } from "../mockData";
=======
import { useState, useEffect } from "react";
import { getRegions, predict as apiPredict } from "../api";
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
=======
import { useMemo, useState } from "react";
import { forecast24h, getZoneDisplayName, regions } from "../mockData";
>>>>>>> d8ba4f8b707174db92668c6ccf394e7bd956ea98

const ZONE_LABELS = {
  "US-MIDA-PJM": "Northern Virginia Region",
  "US-NW-PACW": "Portland Oregon Region",
};

export default function SchedulerPage({ workloadHistory, setWorkloadHistory }) {
  const [regions, setRegions] = useState([]);
  const [name, setName] = useState("");
  const [zone, setZone] = useState("US-MIDA-PJM");
  const [energy, setEnergy] = useState(120);
  const [runtime, setRuntime] = useState(4);
  const [horizon, setHorizon] = useState(1);
  const [priorityHours, setPriorityHours] = useState(6);
  const [pendingRecommendation, setPendingRec] = useState(null);
  const [whatIf, setWhatIf] = useState(null);
  const [loading, setLoading] = useState(false);
  const [whatIfLoading, setWhatIfLoading] = useState(false);

  const allowedZones = ["US-MIDA-PJM", "US-NW-PACW"];

  useEffect(() => {
    getRegions()
      .then((data) => setRegions(data.filter((r) => allowedZones.includes(r.zone))))
      .catch(console.error);
  }, []);

  useEffect(() => {
    setWhatIfLoading(true);
    apiPredict(zone, energy, runtime, horizon, priorityHours)
      .then((data) => { setWhatIf(data); setWhatIfLoading(false); })
      .catch((err) => { console.error("What-if failed:", err); setWhatIfLoading(false); });
  }, [zone, energy, runtime, horizon, priorityHours]);

  const submit = async () => {
    if (!name.trim()) return;
    setLoading(true);
    try {
      const result = await apiPredict(zone, energy, runtime, horizon, priorityHours);
      setPendingRec({ ...result, name, zone });
    } catch (err) {
      console.error("Predict failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const approveSchedule = () => {
    if (!pendingRecommendation) return;
    const record = {
      name: pendingRecommendation.name,
      zone: pendingRecommendation.zone,
      energyKwh: pendingRecommendation.energy_kwh,
      runtimeHours: pendingRecommendation.runtime_hours,
      horizon,
      priorityHours,
      co2SavedKg: pendingRecommendation.co2_saved_kg,
      recommendedDelayHours: pendingRecommendation.hours_to_wait,
      recommendedStart: pendingRecommendation.recommended_start,
      expectedIntensity: pendingRecommendation.expected_intensity_gco2_kwh,
      immediateIntensity: pendingRecommendation.immediate_intensity_gco2_kwh,
    };
    setWorkloadHistory([record, ...workloadHistory]);
    setPendingRec(null);
    setName("");
  };

  return (
    <div className="page-section">
      <div className="page-header">
        <h1>Workload Scheduler</h1>
      </div>

      <section className="surface-card">
        <h3>What-if simulator</h3>
        <div className="form-grid three">
          <label>
            Zone
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d8ba4f8b707174db92668c6ccf394e7bd956ea98
            <select value={zone} onChange={(event) => setZone(event.target.value)}>
              {regions.map((region) => (
                <option key={region.zone} value={region.zone}>
                  {region.name}
                </option>
<<<<<<< HEAD
=======
            <select value={zone} onChange={(e) => setZone(e.target.value)}>
              {regions.map((r) => (
                <option key={r.zone} value={r.zone}>{ZONE_LABELS[r.zone]}</option>
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
=======
>>>>>>> d8ba4f8b707174db92668c6ccf394e7bd956ea98
              ))}
            </select>
          </label>
          <label>
            Energy (kWh)
            <input type="number" value={energy} onChange={(e) => setEnergy(Number(e.target.value))} />
          </label>
          <label>
            Priority hours
            <input type="number" min="0" max="24" value={priorityHours}
              onChange={(e) => setPriorityHours(Number(e.target.value))} />
          </label>
        </div>
        <div className="mini-metrics">
          <div>
            <span>Expected intensity</span>
            <strong>{whatIfLoading ? "..." : `${whatIf?.expected_intensity_gco2_kwh ?? "—"} gCO2/kWh`}</strong>
          </div>
          <div>
            <span>CO2 saved vs now</span>
            <strong>{whatIfLoading ? "..." : `${whatIf?.co2_saved_kg ?? "—"} kg`}</strong>
          </div>
          <div>
            <span>Recommended wait</span>
            <strong>{whatIfLoading ? "..." : `${whatIf?.hours_to_wait ?? "—"}h`}</strong>
          </div>
        </div>
      </section>

      <section className="surface-card">
        <h3>Schedule a new workload</h3>
        <div className="form-grid two">
          <label>
            Workload name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="ML training job" />
          </label>
          <label>
            Zone
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d8ba4f8b707174db92668c6ccf394e7bd956ea98
            <select value={zone} onChange={(event) => setZone(event.target.value)}>
              {regions.map((region) => (
                <option key={region.zone} value={region.zone}>
                  {region.name}
                </option>
<<<<<<< HEAD
=======
            <select value={zone} onChange={(e) => setZone(e.target.value)}>
              {regions.map((r) => (
                <option key={r.zone} value={r.zone}>{ZONE_LABELS[r.zone]}</option>
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
=======
>>>>>>> d8ba4f8b707174db92668c6ccf394e7bd956ea98
              ))}
            </select>
          </label>
          <label>
            Energy (kWh)
            <input type="number" value={energy} onChange={(e) => setEnergy(Number(e.target.value))} />
          </label>
          <label>
            Runtime (hours)
            <input type="number" value={runtime} onChange={(e) => setRuntime(Number(e.target.value))} />
          </label>
          <label>
            Forecast horizon
            <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
              <option value={1}>1h</option>
              <option value={12}>12h</option>
              <option value={24}>24h</option>
            </select>
          </label>
          <label>
            Priority hours
            <input type="number" min="0" max="24" value={priorityHours}
              onChange={(e) => setPriorityHours(Number(e.target.value))} />
          </label>
        </div>
        <button className="primary-action inline" onClick={submit} disabled={loading}>
          {loading ? "Finding window..." : "Find green window"}
        </button>

        {pendingRecommendation && (
          <div className="result-box-react">
            <strong>Model recommendation</strong>
            <p>{pendingRecommendation.recommendation}</p>
            <div className="recommendation-grid">
              <div><span>Recommended start</span><strong>{pendingRecommendation.recommended_start}</strong></div>
              <div><span>Hours to wait</span><strong>{pendingRecommendation.hours_to_wait}h</strong></div>
              <div><span>Expected intensity</span><strong>{pendingRecommendation.expected_intensity_gco2_kwh} gCO2/kWh</strong></div>
              <div><span>Immediate intensity</span><strong>{pendingRecommendation.immediate_intensity_gco2_kwh} gCO2/kWh</strong></div>
              <div><span>Immediate CO2</span><strong>{pendingRecommendation.immediate_co2_kg} kg</strong></div>
              <div><span>Optimal CO2</span><strong>{pendingRecommendation.optimal_co2_kg} kg</strong></div>
              <div><span>CO2 saved</span><strong>{pendingRecommendation.co2_saved_kg} kg</strong></div>
              <div><span>Savings</span><strong>{pendingRecommendation.co2_savings_pct}%</strong></div>
            </div>
            <div className="recommendation-actions">
              <button className="primary-action inline" onClick={approveSchedule}>Schedule</button>
              <button className="ghost-action danger" onClick={() => setPendingRec(null)}>Deny</button>
            </div>
          </div>
        )}
      </section>

      <section className="surface-card">
        <h3>Scheduled workloads</h3>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Zone</th>
                <th>Start</th>
                <th>Energy</th>
                <th>Priority hours</th>
                <th>Runtime</th>
                <th>CO2 saved</th>
              </tr>
            </thead>
            <tbody>
              {workloadHistory.map((item, index) => (
                <tr key={`${item.name}-${index}`}>
                  <td>{item.name}</td>
<<<<<<< HEAD
<<<<<<< HEAD
                  <td>{getZoneDisplayName(item.zone)}</td>
=======
                  <td>{ZONE_LABELS[item.zone] ?? item.zone}</td>
>>>>>>> 974b4e31eab3bc79ec988f0e849161b6a5022e80
=======
                  <td>{getZoneDisplayName(item.zone)}</td>
>>>>>>> d8ba4f8b707174db92668c6ccf394e7bd956ea98
                  <td>{item.recommendedStart ?? "-"}</td>
                  <td>{item.energyKwh} kWh</td>
                  <td>{item.priorityHours ?? 0}h</td>
                  <td>{item.runtimeHours}h</td>
                  <td>{item.co2SavedKg} kg</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
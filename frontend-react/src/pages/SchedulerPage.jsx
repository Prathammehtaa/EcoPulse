import { useMemo, useState } from "react";
import { forecast24h, regions } from "../mockData";

function buildRecommendation(zone, energyKwh, runtimeHours, priorityHours, horizon) {
  const values = forecast24h[zone];
  const limit = Math.max(0, Math.min(Number(horizon) || 0, values.length - 1));
  const allowedWait = Math.max(0, Math.min(Number(priorityHours) || 0, limit));
  const datetimes = Array.from({ length: values.length }, (_, index) => {
    const next = new Date();
    next.setMinutes(0, 0, 0);
    next.setHours(next.getHours() + index);
    return next;
  });

  const immediateAvg = values[0];
  const searchWindow = values.slice(0, allowedWait + 1);
  const bestAvg = Math.min(...searchWindow);
  const bestStart = searchWindow.indexOf(bestAvg);
  const immediateCo2Kg = (immediateAvg * energyKwh * runtimeHours) / 1000;
  const optimalCo2Kg = (bestAvg * energyKwh * runtimeHours) / 1000;
  const co2SavedKg = Math.max(0, immediateCo2Kg - optimalCo2Kg);
  const savingsPct = immediateCo2Kg > 0 ? (co2SavedKg / immediateCo2Kg) * 100 : 0;

  return {
    recommended_start: datetimes[bestStart].toLocaleString(),
    hours_to_wait: bestStart,
    expected_intensity_gco2_kwh: Number(bestAvg.toFixed(2)),
    immediate_intensity_gco2_kwh: Number(immediateAvg.toFixed(2)),
    runtime_hours: runtimeHours,
    energy_kwh: energyKwh,
    immediate_co2_kg: Number(immediateCo2Kg.toFixed(3)),
    optimal_co2_kg: Number(optimalCo2Kg.toFixed(3)),
    co2_saved_kg: Number(co2SavedKg.toFixed(3)),
    co2_savings_pct: Number(savingsPct.toFixed(1)),
    recommendation:
      bestStart > 0
        ? `Wait ${bestStart} hours - start at ${datetimes[bestStart].toLocaleString()}. Save ${co2SavedKg.toFixed(1)} kg CO2 (${savingsPct.toFixed(1)}% reduction).`
        : "Run now - this is already the optimal window."
  };
}

export default function SchedulerPage({ workloadHistory, setWorkloadHistory }) {
  const [name, setName] = useState("");
  const [zone, setZone] = useState("US-MIDA-PJM");
  const [energy, setEnergy] = useState(120);
  const [runtime, setRuntime] = useState(4);
  const [horizon, setHorizon] = useState(6);
  const [priorityHours, setPriorityHours] = useState(6);
  const [pendingRecommendation, setPendingRecommendation] = useState(null);

  const whatIf = useMemo(
    () => buildRecommendation(zone, energy, runtime, priorityHours, horizon),
    [zone, energy, runtime, priorityHours, horizon]
  );

  const submit = () => {
    if (!name.trim()) return;
    const next = buildRecommendation(zone, energy, runtime, priorityHours, horizon);
    setPendingRecommendation({
      ...next,
      name,
      zone
    });
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
      immediateIntensity: pendingRecommendation.immediate_intensity_gco2_kwh
    };

    setWorkloadHistory([record, ...workloadHistory]);
    setPendingRecommendation(null);
    setName("");
  };

  const denySchedule = () => {
    setPendingRecommendation(null);
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
            <select value={zone} onChange={(event) => setZone(event.target.value)}>
              {regions.map((region) => (
                <option key={region.zone} value={region.zone}>
                  {region.zone}
                </option>
              ))}
            </select>
          </label>
          <label>
            Energy (kWh)
            <input type="number" value={energy} onChange={(event) => setEnergy(Number(event.target.value))} />
          </label>
          <label>
            Priority hours
            <input
              type="number"
              min="0"
              max="24"
              value={priorityHours}
              onChange={(event) => setPriorityHours(Number(event.target.value))}
            />
          </label>
        </div>
        <div className="mini-metrics">
          <div>
            <span>Expected intensity</span>
            <strong>{whatIf.expected_intensity_gco2_kwh} gCO2/kWh</strong>
          </div>
          <div>
            <span>CO2 saved vs now</span>
            <strong>{whatIf.co2_saved_kg} kg</strong>
          </div>
          <div>
            <span>Recommended wait</span>
            <strong>{whatIf.hours_to_wait}h</strong>
          </div>
        </div>
      </section>

      <section className="surface-card">
        <h3>Schedule a new workload</h3>
        <div className="form-grid two">
          <label>
            Workload name
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="ML training job" />
          </label>
          <label>
            Zone
            <select value={zone} onChange={(event) => setZone(event.target.value)}>
              {regions.map((region) => (
                <option key={region.zone} value={region.zone}>
                  {region.zone}
                </option>
              ))}
            </select>
          </label>
          <label>
            Energy (kWh)
            <input type="number" value={energy} onChange={(event) => setEnergy(Number(event.target.value))} />
          </label>
          <label>
            Runtime (hours)
            <input type="number" value={runtime} onChange={(event) => setRuntime(Number(event.target.value))} />
          </label>
          <label>
            Forecast horizon
            <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
              <option value={1}>1h</option>
              <option value={6}>6h</option>
              <option value={12}>12h</option>
              <option value={24}>24h</option>
            </select>
          </label>
          <label>
            Priority hours
            <input
              type="number"
              min="0"
              max="24"
              value={priorityHours}
              onChange={(event) => setPriorityHours(Number(event.target.value))}
            />
          </label>
        </div>
        <button className="primary-action inline" onClick={submit}>Find green window</button>

        {pendingRecommendation ? (
          <div className="result-box-react">
            <strong>Model recommendation</strong>
            <p>{pendingRecommendation.recommendation}</p>

            <div className="recommendation-grid">
              <div>
                <span>Recommended start</span>
                <strong>{pendingRecommendation.recommended_start}</strong>
              </div>
              <div>
                <span>Hours to wait</span>
                <strong>{pendingRecommendation.hours_to_wait}h</strong>
              </div>
              <div>
                <span>Expected intensity</span>
                <strong>{pendingRecommendation.expected_intensity_gco2_kwh} gCO2/kWh</strong>
              </div>
              <div>
                <span>Immediate intensity</span>
                <strong>{pendingRecommendation.immediate_intensity_gco2_kwh} gCO2/kWh</strong>
              </div>
              <div>
                <span>Immediate CO2</span>
                <strong>{pendingRecommendation.immediate_co2_kg} kg</strong>
              </div>
              <div>
                <span>Optimal CO2</span>
                <strong>{pendingRecommendation.optimal_co2_kg} kg</strong>
              </div>
              <div>
                <span>CO2 saved</span>
                <strong>{pendingRecommendation.co2_saved_kg} kg</strong>
              </div>
              <div>
                <span>Savings</span>
                <strong>{pendingRecommendation.co2_savings_pct}%</strong>
              </div>
            </div>

            <div className="recommendation-actions">
              <button className="primary-action inline" onClick={approveSchedule}>Schedule</button>
              <button className="ghost-action danger" type="button" onClick={denySchedule}>Deny</button>
            </div>
          </div>
        ) : null}
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
                  <td>{item.zone}</td>
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

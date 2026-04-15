export default function ImpactPage({ workloadHistory }) {
  const totalSaved = workloadHistory.reduce((sum, item) => sum + item.co2SavedKg, 0);
  const baselineWeekly = [62, 58, 71, 65, 60, 48, 52];
  const optimizedWeekly = [44, 40, 49, 45, 41, 33, 36];
  const baselineRunNow = workloadHistory.length > 0 ? (totalSaved + 18.4).toFixed(1) : "--";
  const optimizedRun = workloadHistory.length > 0 ? totalSaved.toFixed(1) : "--";
  const weekLabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const weeklyMax = Math.max(...baselineWeekly, ...optimizedWeekly);

  return (
    <div className="page-section">
      <div className="page-header">
        <h1>Impact & ESG Reporting</h1>
      </div>
      <section className="card-grid compact">
        <article className="stat-card tone-green">
          <div className="stat-label">TOTAL CO2 AVOIDED</div>
          <div className="stat-value-row"><strong>{totalSaved.toFixed(1)}</strong><span>kg</span></div>
        </article>
        <article className="stat-card tone-green">
          <div className="stat-label">TREES EQUIVALENT</div>
          <div className="stat-value-row"><strong>{(totalSaved / 13.3).toFixed(2)}</strong><span>trees</span></div>
        </article>
        <article className="stat-card tone-blue">
          <div className="stat-label">WORKLOADS OPTIMIZED</div>
          <div className="stat-value-row"><strong>{workloadHistory.length}</strong><span>runs</span></div>
        </article>
      </section>
      <section className="impact-visual-grid">
        <article className="surface-card impact-panel">
          <h3>Before vs after EcoPulse</h3>
          <div className="impact-bars">
            <div className="impact-bar-row">
              <span>Baseline (run now)</span>
              <div className="impact-bar-track">
                <div className="impact-bar baseline" style={{ width: workloadHistory.length > 0 ? "78%" : "0%" }} />
              </div>
              <strong>{baselineRunNow} kg</strong>
            </div>
            <div className="impact-bar-row">
              <span>EcoPulse optimized</span>
              <div className="impact-bar-track">
                <div className="impact-bar optimized" style={{ width: workloadHistory.length > 0 ? "61%" : "0%" }} />
              </div>
              <strong>{optimizedRun} kg</strong>
            </div>
          </div>
          <div className="impact-note">
            Schedule workloads to see savings here.
          </div>
        </article>

        <article className="surface-card impact-panel">
          <h3>Weekly CO2 trend</h3>
          <div className="chart-legend">
            <span><i className="legend-swatch slate" /> Baseline CO2</span>
            <span><i className="legend-swatch green" /> EcoPulse CO2</span>
          </div>
          <div className="weekly-chart">
            {weekLabels.map((label, index) => (
              <div className="weekly-group" key={label}>
                <div className="weekly-bars">
                  <div
                    className="weekly-bar baseline"
                    style={{ height: `${(baselineWeekly[index] / weeklyMax) * 140}px` }}
                    title={`Baseline ${baselineWeekly[index]} kg`}
                  />
                  <div
                    className="weekly-bar optimized"
                    style={{ height: `${(optimizedWeekly[index] / weeklyMax) * 140}px` }}
                    title={`EcoPulse ${optimizedWeekly[index]} kg`}
                  />
                </div>
                <div className="bar-label">{label}</div>
              </div>
            ))}
          </div>
        </article>
      </section>
      <section className="surface-card">
        <h3>Audit log</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Workload</th>
              <th>Zone</th>
              <th>Priority</th>
              <th>CO2 saved</th>
            </tr>
          </thead>
          <tbody>
            {workloadHistory.map((item, index) => (
              <tr key={`${item.name}-${index}`}>
                <td>{item.name}</td>
                <td>{item.zone}</td>
                <td>{item.priority}</td>
                <td>{item.co2SavedKg} kg</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

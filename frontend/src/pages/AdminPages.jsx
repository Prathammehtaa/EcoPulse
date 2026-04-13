import { driftRows, logs, metricsRows, shapRows, usersSeed } from "../mockData";
import { useState } from "react";

export function MetricsPage() {
  const bestMae = metricsRows[0].xgboostMae;
  const bestR2 = metricsRows[0].xgboostR2;
  const maeDelta = (metricsRows[0].baselineMae - metricsRows[0].xgboostMae).toFixed(2);
  const baselineAverage = (
    metricsRows.reduce((sum, row) => sum + row.baselineMae, 0) / metricsRows.length
  ).toFixed(2);
  const modelAverage = (
    metricsRows.reduce((sum, row) => sum + row.xgboostMae, 0) / metricsRows.length
  ).toFixed(2);
  const improvement = Math.round(((baselineAverage - modelAverage) / baselineAverage) * 100);

  return (
    <PageShell title="Model Performance Overview">
      <section className="card-grid compact metrics-kpis">
        <article className="stat-card tone-green">
          <div className="stat-label">BEST MAE (1H)</div>
          <div className="stat-value-row">
            <strong>{bestMae}</strong>
          </div>
          <div className="stat-note">-{maeDelta} vs baseline</div>
        </article>
        <article className="stat-card tone-blue">
          <div className="stat-label">BEST R2 (1H)</div>
          <div className="stat-value-row">
            <strong>{bestR2}</strong>
          </div>
          <div className="stat-note">Top short-horizon fit</div>
        </article>
        <article className="stat-card tone-green">
          <div className="stat-label">BEST RMSE (1H)</div>
          <div className="stat-value-row">
            <strong>11.2</strong>
          </div>
          <div className="stat-note">Lower is better</div>
        </article>
        <article className="stat-card tone-amber">
          <div className="stat-label">VS BASELINE AVG</div>
          <div className="stat-value-row">
            <strong>{improvement}%</strong>
          </div>
          <div className="stat-note">Average MAE improvement</div>
        </article>
      </section>

      <section className="surface-card">
        <h3>MAE - XGBoost vs LightGBM vs Baseline</h3>
        <div className="chart-legend">
          <span><i className="legend-swatch green" /> XGBoost MAE</span>
          <span><i className="legend-swatch blue" /> LightGBM MAE</span>
          <span><i className="legend-swatch slate" /> Baseline MAE</span>
        </div>
        <div className="bar-chart-grid">
          {metricsRows.map((row) => {
            const maxMae = Math.max(...metricsRows.flatMap((item) => [item.xgboostMae, item.lightgbmMae, item.baselineMae]));
            return (
              <div className="bar-group" key={row.horizon}>
                <div className="bar-stack">
                  <div className="bar green" style={{ height: `${(row.xgboostMae / maxMae) * 180}px` }} title={`XGBoost ${row.xgboostMae}`} />
                  <div className="bar blue" style={{ height: `${(row.lightgbmMae / maxMae) * 180}px` }} title={`LightGBM ${row.lightgbmMae}`} />
                  <div className="bar slate" style={{ height: `${(row.baselineMae / maxMae) * 180}px` }} title={`Baseline ${row.baselineMae}`} />
                </div>
                <div className="bar-label">{row.horizon}</div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="surface-card">
        <h3>R2 across horizons - XGBoost</h3>
        <div className="line-chart">
          {metricsRows.map((row) => (
            <div className="line-point-group" key={row.horizon}>
              <div className="line-point-value">{row.xgboostR2}</div>
              <div className="line-point-track">
                <div
                  className="line-point-fill"
                  style={{ height: `${((row.xgboostR2 - 0.75) / 0.25) * 160}px` }}
                />
              </div>
              <div className="bar-label">{row.horizon}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="surface-card">
        <h3>Full metrics table</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Horizon</th>
              <th>XGBoost MAE</th>
              <th>LightGBM MAE</th>
              <th>Baseline MAE</th>
              <th>XGBoost R2</th>
              <th>XGBoost RMSE</th>
              <th>XGBoost MAPE</th>
            </tr>
          </thead>
          <tbody>
            {metricsRows.map((row) => (
              <tr key={row.horizon}>
                <td>{row.horizon}</td>
                <td>{row.xgboostMae}</td>
                <td>{row.lightgbmMae}</td>
                <td>{row.baselineMae}</td>
                <td>{row.xgboostR2}</td>
                <td>{row.horizon === "1h" ? 11.2 : row.horizon === "6h" ? 14.7 : row.horizon === "12h" ? 19.4 : 24.1}</td>
                <td>{row.horizon === "1h" ? 4.1 : row.horizon === "6h" ? 5.8 : row.horizon === "12h" ? 7.3 : 9.1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </PageShell>
  );
}

export function ShapPage() {
  return (
    <PageShell title="SHAP Feature Importance & Bias Audit">
      <section className="surface-card">
        <h3>Top features</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Mean |SHAP|</th>
              <th>Direction</th>
            </tr>
          </thead>
          <tbody>
            {shapRows.map((row) => (
              <tr key={row.feature}>
                <td>{row.feature}</td>
                <td>{row.value}</td>
                <td>{row.direction}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </PageShell>
  );
}

export function DriftPage() {
  return (
    <PageShell title="Drift Monitoring">
      <section className="surface-card">
        <h3>Feature drift</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>PSI</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {driftRows.map((row) => (
              <tr key={row.feature}>
                <td>{row.feature}</td>
                <td>{row.psi}</td>
                <td>{row.status}</td>
                <td>{row.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </PageShell>
  );
}

export function ApiPage() {
  return (
    <PageShell title="API / Inference">
      <section className="surface-card">
        <h3>Available endpoints</h3>
        <ul className="endpoint-list">
          <li><strong>POST</strong> /predict</li>
          <li><strong>GET</strong> /forecast/{"{zone}"}</li>
          <li><strong>GET</strong> /regions</li>
          <li><strong>GET</strong> /health</li>
          <li><strong>POST</strong> /retrain</li>
          <li><strong>GET</strong> /metrics/{"{model}"}</li>
        </ul>
      </section>
    </PageShell>
  );
}

export function UsersPage() {
  const [users, setUsers] = useState(usersSeed);
  const [email, setEmail] = useState("");

  return (
    <PageShell title="Registered Users">
      <section className="surface-card">
        <h3>User table</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Initials</th>
              <th>Email</th>
              <th>Status</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.email}>
                <td>{user.initials}</td>
                <td>{user.email}</td>
                <td>{user.status}</td>
                <td>{user.role}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="inline-form">
          <input value={email} placeholder="new@example.com" onChange={(event) => setEmail(event.target.value)} />
          <button
            className="primary-action inline"
            onClick={() => {
              if (!email) return;
              setUsers([...users, { initials: email.slice(0, 2).toUpperCase(), email, status: "Active", role: "Operator" }]);
              setEmail("");
            }}
          >
            Add User
          </button>
        </div>
      </section>
    </PageShell>
  );
}

export function SettingsPage() {
  return (
    <PageShell title="Model & System Settings">
      <section className="surface-card">
        <h3>Configuration</h3>
        <div className="form-grid two">
          <label>Carbon threshold<input type="range" min="50" max="400" defaultValue="200" /></label>
          <label>Forecast horizon<input type="range" min="1" max="24" defaultValue="6" /></label>
          <label>Notification email<input defaultValue="ops@datacenter.com" /></label>
          <label>Slack webhook<input placeholder="https://hooks.slack.com/..." /></label>
        </div>
      </section>
    </PageShell>
  );
}

export function LogsPage() {
  return (
    <PageShell title="System Logs">
      <section className="surface-card">
        <h3>Recent events</h3>
        <div className="stack-list">
          {logs.map(([time, level, message]) => (
            <article className="log-row" key={`${time}-${message}`}>
              <span>{time}</span>
              <strong>{level}</strong>
              <p>{message}</p>
            </article>
          ))}
        </div>
      </section>
    </PageShell>
  );
}

function PageShell({ title, children }) {
  return (
    <div className="page-section">
      <div className="page-header">
        <h1>{title}</h1>
      </div>
      {children}
    </div>
  );
}

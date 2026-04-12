import { alerts } from "../mockData";

export default function AlertsPage() {
  return (
    <div className="page-section">
      <div className="page-header">
        <h1>Alerts & Reliability</h1>
      </div>
      <section className="surface-card">
        <h3>Active alerts</h3>
        <div className="stack-list">
          {alerts.map((alert) => (
            <article key={alert.title} className={`alert-row ${alert.type} ${alert.active ? "active" : "muted"}`}>
              <div>
                <strong>{alert.title}</strong>
                <p>{alert.detail}</p>
              </div>
              <span>{alert.time}</span>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

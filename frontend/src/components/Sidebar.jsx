import LogoMark from "./LogoMark";

export default function Sidebar({ role, currentPage, onNavigate, onLogout, username }) {
  const operatorItems = [
    ["dashboard", "Dashboard"],
    ["scheduler", "Workload Scheduler"]
  ];

  const adminItems = [
    ["metrics", "Model Metrics"],
    ["shap", "SHAP & Bias"],
    ["drift", "Drift Monitor"],
    ["api", "API / Inference"],
    ["users", "Users"],
    ["settings", "Settings"],
    ["logs", "System Logs"]
  ];

  const items = role === "admin" ? [...operatorItems, ...adminItems] : operatorItems;

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-logo">
          <LogoMark className="eco-logo" />
        </div>
        <div>
          <div className="brand-name">
  <span style={{color: "#4fc988"}}>Eco</span><span style={{color: "white"}}>Pulse</span>
          </div>
          <div className="brand-subtitle">Carbon-aware operations</div>
        </div>
      </div>
      <nav className="sidebar-nav">
        {items.map(([key, label]) => (
          <button
            key={key}
            className={`nav-button ${currentPage === key ? "active" : ""}`}
            onClick={() => onNavigate(key)}
          >
            <span>{label}</span>
            {key === "alerts" ? <span className="nav-badge">2</span> : null}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="profile-chip">
          <div className="profile-avatar">{username.slice(0, 2).toUpperCase()}</div>
          <div>
            <div className="profile-name">{username}</div>
            <div className="profile-role">{role === "admin" ? "Admin" : "Operator"}</div>
          </div>
        </div>
        <button className="logout-button" onClick={onLogout}>
          Sign out
        </button>
      </div>
    </aside>
  );
}

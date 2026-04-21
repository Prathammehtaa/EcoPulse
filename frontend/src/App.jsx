import { useEffect, useState } from "react";
import LoginPage from "./components/LoginPage";
import Sidebar from "./components/Sidebar";
import DashboardPage from "./pages/DashboardPage";
import SchedulerPage from "./pages/SchedulerPage";
import AlertsPage from "./pages/AlertsPage";
import LandingPage from "./pages/LandingPage"; 

import {
  ApiPage,
  DriftPage,
  LogsPage,
  MetricsPage,
  SettingsPage,
  ShapPage,
  UsersPage
} from "./pages/AdminPages";

const initialState = {
  loggedIn: false,
  username: "",
  role: "user",
  page: "dashboard",
  workloadHistory: [],
  showLanding: true
};

const STORAGE_KEY = "ecopulse-session-v1";

function loadStoredSession() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return initialState;
    const parsed = JSON.parse(raw);
    return {
      ...initialState,
      ...parsed,
      workloadHistory: Array.isArray(parsed.workloadHistory) ? parsed.workloadHistory : [],
      showLanding: true
    };
  } catch {
    return initialState;
  }
}

export default function App() {
  const [session, setSession] = useState(loadStoredSession);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }, [session]);

  // Show landing page first
  if (session.showLanding) {
    return (
      <LandingPage
        onSignIn={() => setSession((prev) => ({ ...prev, showLanding: false }))}
      />
    );
  }

  // Show login if not logged in
  if (!session.loggedIn) {
    return (
      <LoginPage
        onLogin={({ username, role }) =>
          setSession({
            ...initialState,
            loggedIn: true,
            username,
            role,
            showLanding: false,
            page: role === "admin" ? "metrics" : "dashboard"
          })
        }
      />
    );
  }

  const renderPage = () => {
    if (session.page === "dashboard") {
      return (
        <DashboardPage
          username={session.username}
          workloadHistory={session.workloadHistory}
          onNavigate={(page) => setSession((prev) => ({ ...prev, page }))}
          onApproveRecommended={(record) =>
            setSession((prev) => ({
              ...prev,
              workloadHistory: [record, ...prev.workloadHistory],
              page: "scheduler"
            }))
          }
        />
      );
    }
    if (session.page === "scheduler") {
      return (
        <SchedulerPage
          workloadHistory={session.workloadHistory}
          setWorkloadHistory={(workloadHistory) => setSession((prev) => ({ ...prev, workloadHistory }))}
        />
      );
    }
    if (session.page === "alerts") {
      return <AlertsPage />;
    }

    const adminPages = {
      metrics: <MetricsPage />,
      shap: <ShapPage />,
      drift: <DriftPage />,
      api: <ApiPage />,
      users: <UsersPage />,
      settings: <SettingsPage />,
      logs: <LogsPage />
    };
    return adminPages[session.page] ?? <MetricsPage />;
  };

  return (
    <div className="app-shell">
      <Sidebar
        role={session.role}
        currentPage={session.page}
        username={session.username}
        onNavigate={(page) => setSession((prev) => ({ ...prev, page }))}
        onLogout={() => {
          window.localStorage.removeItem(STORAGE_KEY);
          setSession({ ...initialState, showLanding: true });
        }}
      />
      <main className="app-content">{renderPage()}</main>
    </div>
  );
}
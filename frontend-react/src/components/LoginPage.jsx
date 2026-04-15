import { useState } from "react";
import LogoMark from "./LogoMark";

export default function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("user");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const submit = (event) => {
    event.preventDefault();
    if (!email || password !== "ecopulse") {
      setError("Use any email and password ecopulse.");
      return;
    }
    onLogin({
      username: email.split("@")[0],
      role
    });
  };

  return (
    <div className="login-shell">
      <section className="login-hero">
        <div className="login-hero-logo">
          <LogoMark className="eco-logo large" />
        </div>
        <h1>
          Eco<span>Pulse</span>
        </h1>
        <p>
          Intelligent carbon-aware orchestration with human in the loop decisions for greener,
          operationally safer data centers.
        </p>
      </section>

      <form className="login-form" onSubmit={submit}>
        <div className="login-copy">
          <h2>Sign in</h2>
          <p>Access your carbon decision support dashboard</p>
        </div>

        <div className="segmented">
          <button
            type="button"
            className={role === "user" ? "active" : ""}
            onClick={() => setRole("user")}
          >
            Operator / Engineer
          </button>
          <button
            type="button"
            className={role === "admin" ? "active" : ""}
            onClick={() => setRole("admin")}
          >
            Admin
          </button>
        </div>

        <div className="login-fields">
          <input
            type="email"
            placeholder=""
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          <div className="password-field">
            <input
              type={showPassword ? "text" : "password"}
              placeholder=""
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
            <button
              type="button"
              className="password-toggle"
              onClick={() => setShowPassword((current) => !current)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "🙈" : "👁"}
            </button>
          </div>
        </div>
        {error ? <div className="form-error">{error}</div> : null}
        <button className="primary-action" type="submit">
          Sign in to EcoPulse
        </button>
      </form>
    </div>
  );
}


import { useEffect, useRef } from "react";

// ─── Inline styles (mirrors the original CSS exactly) ────────────────────────
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

  :root {
    --g900:#061c11;--g800:#0f3d28;--g750:#0f4a30;--g700:#145233;--g600:#1a6b43;
    --g500:#22874f;--g400:#2da866;--g300:#4fc988;--g200:#86dba9;--g100:#bfedcf;
    --g50:#e8f8ef;--g25:#f3fcf6;
    --cream:#faf9f6;--ink:#0a1f12;
    --fn:'DM Sans',sans-serif;--fs:'DM Serif Display',serif;
  }

  /* NAV */
  .ep-nav {
    position:fixed;top:0;left:0;right:0;z-index:100;
    display:flex;align-items:center;justify-content:space-between;
    padding:18px 6%;
    background:rgba(250,249,246,0.92);
    backdrop-filter:blur(12px);
    border-bottom:1px solid rgba(15,61,40,0.1);
  }
  .ep-nav-logo { display:flex;align-items:center;gap:10px;cursor:pointer;border:none;background:none; }
  .ep-nav-logo-icon { width:34px;height:34px;display:flex;align-items:center;justify-content:center; }
  .ep-nav-logo-icon svg { width:18px;height:18px; }
  .ep-nav-brand { font-size:17px;font-weight:600;color:var(--g800);letter-spacing:-0.3px;font-family:var(--fn); }
  .ep-nav-links { display:flex;align-items:center;gap:32px; }
  .ep-nav-links button { font-size:14px;color:var(--g700);font-weight:500;opacity:0.8;transition:opacity .2s;cursor:pointer;background:none;border:none;font-family:var(--fn); }
  .ep-nav-links button:hover { opacity:1; }
  .ep-nav-cta { background:var(--g800);color:#fff;font-size:14px;font-weight:500;font-family:var(--fn);padding:9px 22px;border-radius:8px;border:none;cursor:pointer;transition:background .2s; }
  .ep-nav-cta:hover { background:var(--g700); }

  /* HERO */
  .ep-hero {
    min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
    padding:120px 6% 80px;position:relative;overflow:hidden;background:var(--g800);
  }
  .ep-hero-bg-grid {
    position:absolute;inset:0;
    background-image:linear-gradient(rgba(79,201,136,0.06) 1px,transparent 1px),linear-gradient(90deg,rgba(79,201,136,0.06) 1px,transparent 1px);
    background-size:60px 60px;
  }
  .ep-hero-orb {
    position:absolute;width:600px;height:600px;border-radius:50%;
    background:radial-gradient(circle,rgba(45,168,102,0.18) 0%,transparent 70%);
    top:50%;left:50%;transform:translate(-50%,-50%);
    animation:ep-pulse-orb 6s ease-in-out infinite;
  }
  @keyframes ep-pulse-orb { 0%,100%{transform:translate(-50%,-50%) scale(1);}50%{transform:translate(-50%,-50%) scale(1.1);} }
  .ep-hero-badge {
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(79,201,136,0.12);border:1px solid rgba(79,201,136,0.25);
    border-radius:100px;padding:6px 16px;
    font-size:12px;font-weight:500;color:var(--g200);
    letter-spacing:0.04em;text-transform:uppercase;margin-bottom:28px;
    animation:ep-fade-up 0.8s ease both;position:relative;
  }
  .ep-badge-dot { width:6px;height:6px;border-radius:50%;background:var(--g300);animation:ep-blink 2s ease infinite;display:inline-block; }
  @keyframes ep-blink { 0%,100%{opacity:1;}50%{opacity:0.3;} }
  @keyframes ep-fade-up { from{opacity:0;transform:translateY(24px);}to{opacity:1;transform:translateY(0);} }
  .ep-hero h1 {
    font-family:var(--fs);font-size:clamp(40px,6.5vw,78px);
    color:#fff;line-height:1.05;letter-spacing:-1px;
    text-align:center;max-width:820px;
    animation:ep-fade-up 0.8s 0.15s ease both;position:relative;
  }
  .ep-hero h1 em { color:var(--g300);font-style:italic; }
  .ep-hero-sub {
    font-size:clamp(17px,2vw,20px);color:var(--g100);
    text-align:center;max-width:640px;
    line-height:1.7;margin-top:24px;font-weight:300;
    animation:ep-fade-up 0.8s 0.3s ease both;position:relative;
  }
  .ep-hero-actions {
    display:flex;gap:14px;margin-top:42px;flex-wrap:wrap;justify-content:center;
    animation:ep-fade-up 0.8s 0.45s ease both;position:relative;
  }
  .ep-btn-primary {
    background:var(--g300);color:var(--g900);
    font-size:15px;font-weight:600;font-family:var(--fn);
    padding:13px 30px;border-radius:10px;border:none;cursor:pointer;
    transition:all .2s;display:inline-flex;align-items:center;gap:8px;
  }
  .ep-btn-primary:hover { background:#fff;transform:translateY(-1px); }
  .ep-btn-outline {
    background:transparent;color:var(--g100);
    font-size:15px;font-weight:500;font-family:var(--fn);
    padding:13px 30px;border-radius:10px;
    border:1px solid rgba(191,237,207,0.25);cursor:pointer;transition:all .2s;
  }
  .ep-btn-outline:hover { border-color:rgba(191,237,207,0.5);background:rgba(255,255,255,0.04); }

  /* SECTIONS */
  .ep-section { padding:96px 6%; }
  .ep-section-label { font-size:12px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--g600);margin-bottom:14px; }
  .ep-section-title { font-family:var(--fs);font-size:clamp(30px,4vw,48px);color:var(--ink);line-height:1.1;letter-spacing:-0.5px; }
  .ep-section-title em { color:var(--g600);font-style:italic; }
  .ep-section-body { font-size:17px;color:#4a5c4f;line-height:1.7;max-width:560px;margin-top:16px;font-weight:300; }

  /* HOW IT WORKS */
  .ep-how-wrap { background:var(--g25);border-radius:24px;padding:56px; }
  .ep-how-intro { font-size:17px;color:#4a5c4f;line-height:1.75;max-width:680px;margin-top:12px;font-weight:300; }
  .ep-how-steps-grid { display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:40px; }
  .ep-step { display:flex;gap:16px;padding:24px;background:#fff;border:1px solid rgba(15,61,40,0.1);border-radius:14px; }
  .ep-step-num { width:32px;height:32px;border-radius:50%;background:var(--g800);color:#fff;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px; }
  .ep-step-title { font-size:14px;font-weight:600;color:var(--g800);margin-bottom:5px; }
  .ep-step-desc { font-size:13px;color:#5a7a65;line-height:1.6; }

  /* BEFORE / AFTER */
  .ep-compare-wrap { margin-top:40px;display:grid;grid-template-columns:1fr auto 1fr;gap:0;align-items:stretch; }
  .ep-compare-col { border-radius:16px;overflow:hidden; }
  .ep-compare-col.ep-before { background:#fff;border:1px solid rgba(15,61,40,0.1); }
  .ep-compare-col.ep-after { background:var(--g800);border:1px solid var(--g600); }
  .ep-compare-head { padding:14px 20px;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;display:flex;align-items:center;gap:8px; }
  .ep-before .ep-compare-head { background:var(--g25);color:#6b7a6e;border-bottom:1px solid rgba(15,61,40,0.08); }
  .ep-after .ep-compare-head { background:var(--g700);color:var(--g200);border-bottom:1px solid rgba(255,255,255,0.08); }
  .ep-compare-head-dot { width:8px;height:8px;border-radius:50%;display:inline-block; }
  .ep-compare-body { padding:20px; }
  .ep-compare-row { display:flex;align-items:center;justify-content:space-between;padding:11px 0;border-bottom:1px solid rgba(15,61,40,0.07); }
  .ep-after .ep-compare-row { border-bottom-color:rgba(255,255,255,0.06); }
  .ep-compare-row:last-child { border-bottom:none; }
  .ep-cr-label { font-size:13px;color:#5a7a65; }
  .ep-after .ep-cr-label { color:var(--g200); }
  .ep-cr-val { font-size:13px;font-weight:600;color:var(--ink); }
  .ep-after .ep-cr-val { color:#fff; }
  .ep-cr-val.ep-bad { color:#c0392b; }
  .ep-cr-val.ep-good { color:var(--g400); }
  .ep-compare-divider { display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 20px;gap:8px; }
  .ep-compare-arrow { width:36px;height:36px;border-radius:50%;background:var(--g800);color:var(--g300);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700; }
  .ep-compare-vs { font-size:10px;font-weight:700;letter-spacing:0.1em;color:#6b7a6e;text-transform:uppercase; }

  /* IMPACT */
  .ep-impact-section { background:var(--g800);padding:96px 6%; }
  .ep-impact-section .ep-section-label { color:var(--g300); }
  .ep-imp-headline { font-family:Georgia,'Times New Roman',serif;font-size:clamp(32px,5vw,58px);font-weight:400;color:#fff;line-height:1.15;letter-spacing:-0.5px;margin:14px 0 20px; }
  .ep-imp-headline em { color:var(--g300);font-style:italic; }
  .ep-imp-sub { font-size:16px;color:var(--g200);font-weight:300;line-height:1.6;max-width:480px; }
  .ep-imp-stats { display:flex;align-items:center;gap:32px;margin:52px 0 56px;flex-wrap:wrap; }
  .ep-imp-stat { background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:32px 36px; }
  .ep-imp-stat.ep-highlight { background:rgba(79,201,136,0.1);border-color:rgba(79,201,136,0.3); }
  .ep-imp-num { font-family:Georgia,'Times New Roman',serif;font-size:clamp(48px,7vw,80px);font-weight:400;color:#fff;line-height:1; }
  .ep-imp-num span { font-size:0.45em;color:var(--g300);margin-left:3px;vertical-align:super; }
  .ep-imp-stat.ep-highlight .ep-imp-num { color:var(--g300); }
  .ep-imp-desc { font-size:14px;color:var(--g200);margin-top:10px;line-height:1.5; }
  .ep-imp-muted { font-size:12px;color:var(--g300);opacity:0.7; }
  .ep-imp-arrow { font-size:28px;color:var(--g400);flex-shrink:0; }
  .ep-imp-scale { border-top:1px solid rgba(255,255,255,0.1);padding-top:48px; }
  .ep-imp-scale-intro { font-family:Georgia,'Times New Roman',serif;font-size:18px;color:var(--g100);font-style:italic;margin-bottom:32px; }
  .ep-imp-scale-items { display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(255,255,255,0.08);border-radius:14px;overflow:hidden; }
  .ep-imp-scale-item { background:rgba(255,255,255,0.03);padding:28px 24px;text-align:center; }
  .ep-imp-scale-num { font-family:Georgia,'Times New Roman',serif;font-size:clamp(28px,3.5vw,40px);font-weight:400;color:#fff;line-height:1;margin-bottom:8px; }
  .ep-imp-scale-label { font-size:12px;color:var(--g200);line-height:1.4; }

  /* ZONES */
  .ep-zones-section { background:var(--g25); }
  .ep-zones-grid { display:grid;grid-template-columns:1fr 1fr;gap:48px;align-items:center;margin-top:48px; }
  .ep-zone-cards { display:flex;flex-direction:column;gap:12px; }
  .ep-zone-card { display:flex;align-items:center;gap:16px;background:#fff;border:1px solid rgba(15,61,40,0.1);border-radius:12px;padding:16px 20px; }
  .ep-zone-swatch { width:10px;height:40px;border-radius:4px;flex-shrink:0; }
  .ep-zone-name { font-size:14px;font-weight:600;color:var(--g800); }
  .ep-zone-sub { font-size:12px;color:#7a9a85;margin-top:2px; }
  .ep-zone-badge { margin-left:auto;font-size:11px;font-weight:600;padding:4px 12px;border-radius:100px; }
  .ep-badge-green { background:var(--g50);color:var(--g600); }
  .ep-badge-yellow { background:#fffbeb;color:#92400e; }
  .ep-zones-panel { background:var(--g800);border-radius:18px;padding:32px;position:relative;overflow:hidden; }
  .ep-zones-panel-label { font-size:11px;color:var(--g200);letter-spacing:0.06em;text-transform:uppercase;margin-bottom:24px; }
  .ep-zone-bar-row { margin-bottom:16px; }
  .ep-zone-bar-head { display:flex;justify-content:space-between;margin-bottom:6px; }
  .ep-zone-bar-head span:first-child { font-size:13px;color:var(--g100); }
  .ep-zone-bar-head span:last-child { font-size:13px;font-weight:600;color:#fff; }
  .ep-zone-bar-track { background:rgba(255,255,255,0.1);border-radius:4px;height:8px; }
  .ep-zone-bar-fill { height:8px;border-radius:4px; }
  .ep-zone-pills { display:flex;gap:6px;flex-wrap:wrap;margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.1); }
  .ep-zone-pill { font-size:11px;padding:4px 10px;border-radius:100px; }

  /* CTA */
  .ep-cta-section { background:var(--g800);text-align:center;padding:100px 6%;position:relative;overflow:hidden; }
  .ep-cta-orb { position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(79,201,136,0.15) 0%,transparent 70%);top:50%;left:50%;transform:translate(-50%,-50%); }
  .ep-cta-section h2 { font-family:var(--fs);font-size:clamp(32px,5vw,56px);color:#fff;position:relative;line-height:1.1; }
  .ep-cta-section h2 em { color:var(--g300);font-style:italic; }
  .ep-cta-section p { font-size:18px;color:var(--g100);margin:18px auto 40px;max-width:480px;font-weight:300;position:relative; }

  /* FOOTER */
  .ep-footer { background:var(--g900);padding:40px 6% 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:20px; }
  .ep-footer-brand { display:flex;align-items:center;gap:10px; }
  .ep-footer-brand span { font-size:15px;font-weight:600;color:var(--g200); }
  .ep-footer-links { display:flex;gap:24px; }
  .ep-footer-links button { font-size:13px;color:var(--g200);opacity:0.6;transition:opacity .2s;cursor:pointer;background:none;border:none;font-family:var(--fn); }
  .ep-footer-links button:hover { opacity:1; }
  .ep-footer-copy { font-size:12px;color:var(--g200);opacity:0.4;width:100%;margin-top:12px; }

  /* SCROLL REVEAL */
  .ep-reveal { opacity:1;transform:translateY(0);transition:opacity 0.7s ease,transform 0.7s ease; }
  .ep-reveal.ep-hidden { opacity:0;transform:translateY(28px); }
  .ep-reveal.ep-visible { opacity:1;transform:translateY(0); }

  /* RESPONSIVE */
  @media(max-width:900px) {
    .ep-how-steps-grid,.ep-zones-grid { grid-template-columns:1fr; }
    .ep-imp-stats { flex-direction:column;gap:16px; }
    .ep-imp-scale-items { grid-template-columns:1fr 1fr; }
    .ep-compare-wrap { grid-template-columns:1fr; }
  }
  @media(max-width:600px) {
    .ep-nav-links { display:none; }
    .ep-how-wrap { padding:40px 24px; }
    .ep-imp-scale-items { grid-template-columns:1fr 1fr; }
  }
`;

// ─── Logo SVG ────────────────────────────────────────────────────────────────
function Logo({ size = 34 }) {
  return <img src="/ecopulse_logo_v2.svg" width={size} height={size} alt="EcoPulse" style={{ display: "block" }} />;
}

// ─── Scroll reveal hook ───────────────────────────────────────────────────────
function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll(".ep-reveal");
    els.forEach((el) => el.classList.add("ep-hidden"));
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.remove("ep-hidden");
            e.target.classList.add("ep-visible");
          }
        });
      },
      { threshold: 0.06 }
    );
    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);
}

// ─── Nav ─────────────────────────────────────────────────────────────────────
function Nav({ onNav, onSignIn }) {
  return (
    <nav className="ep-nav">
      <button className="ep-nav-logo" onClick={() => onNav("hero")}>
        <div className="ep-nav-logo-icon"><Logo /></div>
        <span className="ep-nav-brand">EcoPulse</span>
      </button>
      <div className="ep-nav-links">
        <button onClick={() => onNav("how")}>How it works</button>
        <button onClick={() => onNav("impact")}>Our impact</button>
        <button onClick={() => onNav("zones")}>Where we work</button>
      </div>
      <button className="ep-nav-cta" onClick={() => onSignIn()}>Sign in</button>
    </nav>
  );
}

// ─── Page 1: Hero ────────────────────────────────────────────────────────────
function Hero({ onNav }) {
  return (
    <section className="ep-hero" id="hero">
      <div className="ep-hero-bg-grid" />
      <div className="ep-hero-orb" />
      <div className="ep-hero-badge" style={{ position: "relative" }}>
        <span className="ep-badge-dot" />
        Making electricity smarter, one task at a time
      </div>
      <h1 style={{ position: "relative" }}>
        Not all electricity is equally clean..<br/>Ecopulse finds the<em> cleanest moment </em>to run your work. 
      </h1>
      <p className="ep-hero-sub" style={{ position: "relative" }}>
        Every hour, the electricity coming from the grid gets greener or dirtier — depending on whether the wind is blowing or the sun is shining. EcoPulse watches those shifts and tells computers exactly when to do their heavy work, so less pollution ends up in the atmosphere.
      </p>
      <div className="ep-hero-actions" style={{ position: "relative" }}>
        <button className="ep-btn-primary" onClick={() => onNav("how")}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 2v12M2 8l6 6 6-6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Show me how
        </button>
        <button className="ep-btn-outline" onClick={() => onNav("impact")}>See the impact →</button>
      </div>
    </section>
  );
}

// ─── Page 2: How it works ────────────────────────────────────────────────────
const steps = [
  { n: 1, title: "We watch the grid every hour", desc: "Live readings of how clean or dirty the electricity is, plus weather affecting solar and wind." },
  { n: 2, title: "We check the data for problems", desc: "We catch gaps and errors before making any predictions, keeping results fair and reliable." },
  { n: 3, title: "We forecast the cleanest hours ahead", desc: "Two systems cross-check each other to predict grid cleanliness up to 24 hours ahead." },
  { n: 4, title: "We say: wait — do it then instead", desc: "EcoPulse picks the cleanest upcoming window. Same deadline. Far less pollution." },
];

const beforeRows = [
  { label: "Task runs at", val: "Right now", cls: "ep-bad" },
  { label: "Grid cleanliness", val: "Unknown", cls: "ep-bad" },
  { label: "Carbon emitted", val: "218 gCO₂/kWh", cls: "ep-bad" },
  { label: "Completed by", val: "2:00 PM", cls: "" },
  { label: "Emissions avoided", val: "None", cls: "ep-bad" },
];

const afterRows = [
  { label: "Task runs at", val: "+6h window", cls: "ep-good" },
  { label: "Grid cleanliness", val: "Forecasted clean", cls: "ep-good" },
  { label: "Carbon emitted", val: "84 gCO₂/kWh", cls: "ep-good" },
  { label: "Completed by", val: "8:00 PM", cls: "" },
  { label: "Emissions avoided", val: "61% less CO₂", cls: "ep-good" },
];

function HowItWorks() {
  return (
    <section className="ep-section" id="how">
      <div className="ep-how-wrap ep-reveal">
        <div className="ep-section-label">How it works</div>
        <div className="ep-section-title">
          Like a <em>weather forecast</em> — but for clean energy
        </div>
        <p className="ep-how-intro">
          EcoPulse checks the power grid before big computer tasks run — and finds the windows where the least pollution will be produced. Same work, cleaner result.
        </p>

        <div className="ep-how-steps-grid">
          {steps.map((s) => (
            <div className="ep-step" key={s.n}>
              <div className="ep-step-num">{s.n}</div>
              <div>
                <div className="ep-step-title">{s.title}</div>
                <div className="ep-step-desc">{s.desc}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Before / After */}
        <div className="ep-compare-wrap">
          <div className="ep-compare-col ep-before">
            <div className="ep-compare-head">
              <span className="ep-compare-head-dot" style={{ background: "#e07070" }} />
              Without EcoPulse
            </div>
            <div className="ep-compare-body">
              {beforeRows.map((r) => (
                <div className="ep-compare-row" key={r.label}>
                  <span className="ep-cr-label">{r.label}</span>
                  <span className={`ep-cr-val ${r.cls}`}>{r.val}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="ep-compare-divider">
            <div className="ep-compare-arrow">→</div>
            <div className="ep-compare-vs">vs</div>
          </div>

          <div className="ep-compare-col ep-after">
            <div className="ep-compare-head">
              <span className="ep-compare-head-dot" style={{ background: "#4fc988" }} />
              With EcoPulse
            </div>
            <div className="ep-compare-body">
              {afterRows.map((r) => (
                <div className="ep-compare-row" key={r.label}>
                  <span className="ep-cr-label">{r.label}</span>
                  <span className={`ep-cr-val ${r.cls}`}>{r.val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── Page 3: Impact ──────────────────────────────────────────────────────────
const scaleItems = [
  { num: "389t", label: "CO₂ avoided per year" },
  { num: "85", label: "cars off the road" },
  { num: "6,400", label: "trees worth of absorption" },
  { num: "$0", label: "cost to implement" },
];

function Impact() {
  return (
    <section className="ep-impact-section ep-section" id="impact">
      <div className="ep-reveal">
        <div className="ep-section-label">Real-world impact</div>
        <div className="ep-imp-headline">
          One decision.<br />Seven hours of waiting.<br /><em>29% less pollution.</em>
        </div>
        <p className="ep-imp-sub">We ran a real test. One computer job, scheduled smarter. Here's what happened.</p>
      </div>

      <div className="ep-imp-stats ep-reveal">
        <div className="ep-imp-stat">
          <div className="ep-imp-num">134<span>kg</span></div>
          <div className="ep-imp-desc">
            CO₂ if run at 6 PM<br />
            <span className="ep-imp-muted">dirty grid, Virginia</span>
          </div>
        </div>
        <div className="ep-imp-arrow">→</div>
        <div className="ep-imp-stat ep-highlight">
          <div className="ep-imp-num">95<span>kg</span></div>
          <div className="ep-imp-desc">
            CO₂ if run at 1 AM<br />
            <span className="ep-imp-muted">clean grid, same job</span>
          </div>
        </div>
      </div>

      <div className="ep-imp-scale ep-reveal">
        <p className="ep-imp-scale-intro">Do that across a mid-size data center for a year:</p>
        <div className="ep-imp-scale-items">
          {scaleItems.map((item) => (
            <div className="ep-imp-scale-item" key={item.num}>
              <div className="ep-imp-scale-num">{item.num}</div>
              <div className="ep-imp-scale-label">{item.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── Page 4: Grid Zones ──────────────────────────────────────────────────────
function Zones() {
  return (
    <section className="ep-section ep-zones-section" id="zones">
      <div className="ep-reveal">
        <div className="ep-section-label">Where we work</div>
        <div className="ep-section-title">
          Two major US power regions,<br /><em>monitored live</em>
        </div>
        <p className="ep-section-body">
          Different parts of the country generate electricity in very different ways. EcoPulse currently covers two major US regions — giving operators real scheduling intelligence across both.
        </p>

        <div className="ep-zones-grid" style={{ marginTop: 40 }}>
          <div className="ep-zone-cards">
            {[
              { color: "#22874f", name: "Mid-Atlantic (PJM)", sub: "Heavy mix of coal, gas & nuclear · High variation throughout the day", badge: "Moderate", badgeCls: "ep-badge-yellow", opacity: 1 },
              { color: "#4fc988", name: "Pacific Northwest (PACW)", sub: "Dominated by hydropower & wind · Naturally cleaner baseline", badge: "Clean", badgeCls: "ep-badge-green", opacity: 1 },
              { color: "#d1d5db", name: "More regions coming", sub: "California, Texas, Midwest — planned expansion", badge: "Roadmap", badgeCls: "", opacity: 0.5 },
            ].map((z) => (
              <div className="ep-zone-card" key={z.name} style={{ opacity: z.opacity }}>
                <div className="ep-zone-swatch" style={{ background: z.color }} />
                <div>
                  <div className="ep-zone-name">{z.name}</div>
                  <div className="ep-zone-sub">{z.sub}</div>
                </div>
                <span className={`ep-zone-badge ${z.badgeCls}`} style={z.badgeCls ? {} : { background: "#f3f4f6", color: "#6b7280" }}>
                  {z.badge}
                </span>
              </div>
            ))}
          </div>

          <div className="ep-zones-panel">
            <div className="ep-zones-panel-label">How clean is each region — 24 hour average</div>
            <div className="ep-zone-bar-row">
              <div className="ep-zone-bar-head">
                <span>Mid-Atlantic (PJM)</span><span>Moderate carbon</span>
              </div>
              <div className="ep-zone-bar-track">
                <div className="ep-zone-bar-fill" style={{ background: "#2da866", width: "65%" }} />
              </div>
            </div>
            <div className="ep-zone-bar-row">
              <div className="ep-zone-bar-head">
                <span>Pacific Northwest (PACW)</span><span>Low carbon ✓</span>
              </div>
              <div className="ep-zone-bar-track">
                <div className="ep-zone-bar-fill" style={{ background: "#4fc988", width: "38%" }} />
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#86dba9", marginBottom: 12, marginTop: 16, paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.1)" }}>
              What the cleanliness levels mean
            </div>
            <div className="ep-zone-pills">
              <span className="ep-zone-pill" style={{ background: "rgba(79,201,136,0.25)", color: "#86dba9" }}>🟢 Very clean — schedule now</span>
              <span className="ep-zone-pill" style={{ background: "rgba(45,168,102,0.2)", color: "#86dba9" }}>🟡 Okay — wait if possible</span>
              <span className="ep-zone-pill" style={{ background: "rgba(220,101,67,0.2)", color: "#fca5a5" }}>🔴 High pollution — delay</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── CTA ─────────────────────────────────────────────────────────────────────
function CTA({ onSignIn }) {
  return (
    <section className="ep-cta-section" id="cta">
      <div className="ep-cta-orb" />
      <h2><em>Ready to run compute cleaner?</em></h2>
      <p>Sign in to start scheduling smarter and see real-time carbon forecasts for your region.</p>
      <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap", position: "relative" }}>
        <button className="ep-btn-primary" onClick={() => onSignIn()}>
          Log in to EcoPulse →
        </button>
      </div>
    </section>
  );
}

// ─── Footer ──────────────────────────────────────────────────────────────────
function Footer({ onNav }) {
  return (
    <footer className="ep-footer">
      <div className="ep-footer-brand">
        <div className="ep-nav-logo-icon" style={{ width: 28, height: 28, borderRadius: 8 }}>
          <Logo size={28} />
        </div>
        <span>EcoPulse</span>
      </div>
      <div className="ep-footer-links">
        <button onClick={() => onNav("how")}>How it works</button>
        <button onClick={() => onNav("impact")}>Impact</button>
        <button onClick={() => onNav("zones")}>Grid zones</button>
      </div>
    </footer>
  );
}

// ─── Root component ───────────────────────────────────────────────────────────
export default function EcoPulseLanding({ onSignIn }) {
  useReveal();

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <>
      <style>{styles}</style>
      <Nav onNav={scrollTo} onSignIn={onSignIn} />
      <Hero onNav={scrollTo} />
      <HowItWorks />
      <Impact />
      <Zones />
      <CTA onSignIn={onSignIn} />
      <Footer onNav={scrollTo} />
    </>
  );
}
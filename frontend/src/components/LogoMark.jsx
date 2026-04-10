export default function LogoMark({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 68 68"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="34" cy="34" r="32" fill="#0a2e1e" />
      <ellipse cx="34" cy="50" rx="17" ry="6" fill="rgba(0,0,0,.25)" />
      <path
        d="M34 11C34 11 18 23 18 37C18 47.5 25.7 56 34 56C42.3 56 50 47.5 50 37C50 23 34 11 34 11Z"
        fill="#4fc988"
      />
      <path
        d="M34 11C34 11 27 24 27 37C27 47.5 30.5 56 34 56"
        fill="#2da866"
        opacity=".5"
      />
      <path d="M34 56L34 31" stroke="#0a2e1e" strokeWidth="2.8" strokeLinecap="round" />
      <path d="M34 45C34 45 26 41 21 32" stroke="#0a2e1e" strokeWidth="2" strokeLinecap="round" />
      <path d="M34 38C34 38 42 34 47 25" stroke="#0a2e1e" strokeWidth="2" strokeLinecap="round" />
      <circle cx="34" cy="11" r="3" fill="#bfedcf" />
      <circle cx="21" cy="32" r="2.2" fill="#bfedcf" />
      <circle cx="47" cy="25" r="2.2" fill="#bfedcf" />
      <path
        d="M23 60Q34 66 45 60"
        stroke="#22874f"
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}

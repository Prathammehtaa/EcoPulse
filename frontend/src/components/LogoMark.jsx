export default function LogoMark({ className = "" }) {
  return (
    <svg
  className={className}
  viewBox="0 0 120 128"
  fill="none"
  aria-hidden="true"
>
    >
      <circle cx="60" cy="60" r="58" fill="#0f3d28"/>
      <path d="M60 18C60 18 38 34 38 54C38 67 47 77 60 77C73 77 82 67 82 54C82 34 60 18 60 18Z" fill="#4fc988"/>
      <path d="M60 18C60 18 52 35 51 54C50 65 54 74 60 77" fill="#2da866" opacity=".55"/>
      <clipPath id="lcp">
        <path d="M60 18C60 18 38 34 38 54C38 67 47 77 60 77C73 77 82 67 82 54C82 34 60 18 60 18Z"/>
      </clipPath>
      <path d="M38 54 L44 54 L47 48 L51 62 L54 44 L57 64 L59.5 54 L60.5 54 L63 49 L67 61 L70 54 L82 54"
        stroke="#0a1a0e" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" fill="none"
        clipPath="url(#lcp)"/>
      <circle cx="60" cy="18" r="3" fill="#bfedcf"/>
      <line x1="60" y1="77" x2="60" y2="87" stroke="#2da866" strokeWidth="0.7" strokeLinecap="round"/>
      <line x1="56" y1="83" x2="64" y2="83" stroke="#2da866" strokeWidth="0.7" strokeLinecap="round"/>
      <path d="M48 93 Q60 100 72 93" stroke="#4fc988" strokeWidth="2.2" fill="none" strokeLinecap="round"/>
    </svg>
  );
}
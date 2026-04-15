function buildPath(values, width, height, padding) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = Math.max(max - min, 1);
  return values
    .map((value, index) => {
      const x = padding + (index * (width - padding * 2)) / (values.length - 1);
      const y = height - padding - ((value - min) / range) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
}

function pointY(value, min, max, height, padding) {
  return height - padding - ((value - min) / Math.max(max - min, 1)) * (height - padding * 2);
}

function buildAreaPath(values, indexes, width, height, padding, min, max) {
  if (!indexes.length) return "";
  const points = indexes.map((index) => {
    const x = padding + (index * (width - padding * 2)) / (values.length - 1);
    const y = pointY(values[index], min, max, height, padding);
    return { x, y };
  });

  const bottom = height - padding;
  const first = points[0];
  const last = points[points.length - 1];

  return [
    `M ${first.x} ${bottom}`,
    `L ${first.x} ${first.y}`,
    ...points.slice(1).map((point) => `L ${point.x} ${point.y}`),
    `L ${last.x} ${bottom}`,
    "Z"
  ].join(" ");
}

export default function SimpleChart({ values, placementIndexes = [] }) {
  const width = 860;
  const height = 240;
  const padding = 34;
  const path = buildPath(values, width, height, padding);
  const max = Math.max(...values);
  const min = Math.min(...values);
  const tickCount = 7;
  const ticks = Array.from({ length: tickCount }, (_, index) => {
    const ratio = index / (tickCount - 1);
    return Math.round(max - (max - min) * ratio);
  });
  const labels = ["Now", "2h", "4h", "6h", "8h", "10h", "12h", "14h", "16h", "18h", "20h", "22h", "24h"];
  const greenIndexes = values.map((value, index) => (value <= 160 ? index : -1)).filter((index) => index !== -1);
  const redIndexes = values.map((value, index) => (value >= 220 ? index : -1)).filter((index) => index !== -1);
  const greenAreaPath = buildAreaPath(values, greenIndexes, width, height, padding, min, max);
  const redAreaPath = buildAreaPath(values, redIndexes, width, height, padding, min, max);

  return (
    <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Forecast chart">
      {greenAreaPath ? <path d={greenAreaPath} className="chart-area-green" /> : null}
      {redAreaPath ? <path d={redAreaPath} className="chart-area-red" /> : null}

      {ticks.map((tick, index) => {
        const y = pointY(tick, min, max, height, padding);
        return <line key={`grid-${index}`} x1={padding} x2={width - padding} y1={y} y2={y} className="chart-grid" />;
      })}
      {ticks.map((tick) => {
        const y = pointY(tick, min, max, height, padding);
        return (
          <text key={`tick-${tick}`} x={6} y={y + 4} className="chart-axis-text">
            {tick}
          </text>
        );
      })}

      {placementIndexes.map((index) => {
        const x = padding + (index * (width - padding * 2)) / (values.length - 1);
        const y = pointY(values[index], min, max, height, padding);
        return (
          <g key={`placement-${index}`}>
            <line x1={x} x2={x} y1={padding} y2={height - padding} className="placement-line" />
            <circle cx={x} cy={y} r="7" className="placement-dot" />
          </g>
        );
      })}

      <path d={path} className="chart-line-shadow" />
      <path d={path} className="chart-line" />

      {values.map((value, index) => {
        const x = padding + (index * (width - padding * 2)) / (values.length - 1);
        const y = pointY(value, min, max, height, padding);
        return <circle key={index} cx={x} cy={y} r="4" className="chart-dot" />;
      })}

      {labels.map((label, index) => {
        const x = padding + (index * (width - padding * 2)) / (labels.length - 1);
        return (
          <text key={label} x={x} y={height - 8} textAnchor="middle" className="chart-axis-text">
            {label}
          </text>
        );
      })}
    </svg>
  );
}

import { useMemo } from "react";

interface SparklineProps {
  /** Series in ascending time order. Missing samples become gaps. */
  values: (number | null | undefined)[];
  width?: number;
  height?: number;
  strokeWidth?: number;
  /** Color token (text-* class). Defaults to accent. */
  colorClass?: string;
  /** Optional accessibility label. */
  label?: string;
}

/**
 * Minimal viewBox-scaled SVG sparkline. No library — points are mapped into
 * a fixed [0, 1] coordinate space and the outer SVG scales via viewBox. Null
 * / undefined samples split the polyline so the chart doesn't interpolate
 * across gaps in polling.
 */
export function Sparkline({
  values,
  width = 320,
  height = 48,
  strokeWidth = 1.5,
  colorClass = "text-accent",
  label,
}: SparklineProps) {
  const paths = useMemo(() => buildPaths(values), [values]);

  if (paths.length === 0) {
    return (
      <svg
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        style={{ width, height }}
        aria-label={label}
      />
    );
  }

  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      style={{ width, height }}
      className={colorClass}
      aria-label={label}
      role="img"
    >
      {paths.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth / Math.min(width, height)}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

function buildPaths(values: (number | null | undefined)[]): string[] {
  const finite = values
    .map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));
  const present = finite.filter((v): v is number => v !== null);
  if (present.length < 2) return [];

  const min = Math.min(...present);
  const max = Math.max(...present);
  const range = max - min || 1;

  // Map x into [0.01, 0.99] to keep stroke inside the viewBox, and y so
  // that larger values are higher (SVG y grows downward).
  const xStep = values.length > 1 ? 0.98 / (values.length - 1) : 0;
  const paths: string[] = [];
  let current: string[] = [];

  finite.forEach((v, i) => {
    if (v === null) {
      if (current.length >= 2) paths.push(current.join(" "));
      current = [];
      return;
    }
    const x = 0.01 + i * xStep;
    const y = 0.98 - ((v - min) / range) * 0.96;
    current.push(`${current.length === 0 ? "M" : "L"} ${x.toFixed(4)} ${y.toFixed(4)}`);
  });
  if (current.length >= 2) paths.push(current.join(" "));

  return paths;
}

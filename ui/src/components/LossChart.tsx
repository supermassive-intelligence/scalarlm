import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";

import type { TrainingHistoryPoint } from "@/api/training";
import { readJson, writeJson } from "@/lib/preferences";

interface LossChartProps {
  history: TrainingHistoryPoint[];
  maxSteps?: number;
}

// TensorBoard-style EMA: smooth[i] = w * smooth[i-1] + (1 - w) * raw[i]
// with debiasing for the warmup. w=0 returns the raw series.
const SMOOTH_KEY = "loss-chart.smoothing";
const SMOOTH_MAX = 0.95;

/**
 * Step-vs-loss chart for a training job's history[]. Hand-rolled SVG so we
 * don't pull a charting library into the bundle just for this. Supports
 * linear / log-y toggle, per-epoch vertical dividers, and hover crosshair
 * with a label that reports (step, loss, epoch, elapsed).
 */
export function LossChart({ history, maxSteps }: LossChartProps) {
  const [logY, setLogY] = useState(false);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [smoothing, setSmoothing] = useState<number>(() =>
    clamp(readJson<number>(SMOOTH_KEY, 0), 0, SMOOTH_MAX),
  );

  useEffect(() => {
    writeJson(SMOOTH_KEY, smoothing);
  }, [smoothing]);

  const {
    paths,
    smoothedPaths,
    epochMarkers,
    loMax,
    loMin,
    xMin,
    xMax,
    smoothedSeries,
  } = useMemo(
    () => computeChart(history, logY, smoothing),
    [history, logY, smoothing],
  );

  if (history.length < 2) {
    return (
      <div className="flex h-48 items-center justify-center rounded-md border border-border-subtle bg-bg/50 text-sm text-fg-muted">
        Waiting for training samples…
      </div>
    );
  }

  const width = 800;
  const height = 220;
  const padL = 40;
  const padR = 8;
  const padT = 8;
  const padB = 24;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xForIdx = (i: number) =>
    padL + (history.length === 1 ? 0 : (i / (history.length - 1)) * plotW);

  const hovered = hoverIdx !== null ? history[hoverIdx] : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="text-xs text-fg-muted">
          {history.length} samples · step {xMin} → {xMax}
          {maxSteps ? ` / ${maxSteps}` : ""}
        </div>
        <label
          className="ml-auto flex items-center gap-2 text-xs text-fg-muted"
          title="Exponential moving average. 0 = raw, 0.95 = heavily smoothed."
        >
          smoothing
          <input
            type="range"
            min={0}
            max={SMOOTH_MAX}
            step={0.01}
            value={smoothing}
            onChange={(e) => setSmoothing(Number(e.target.value))}
            aria-label="Loss smoothing"
            className="w-32 accent-accent"
          />
          <span className="w-10 text-right font-mono">
            {smoothing.toFixed(2)}
          </span>
        </label>
        <label className="flex items-center gap-2 text-xs text-fg-muted">
          <input
            type="checkbox"
            checked={logY}
            onChange={(e) => setLogY(e.target.checked)}
            className="h-3 w-3 accent-accent"
          />
          log y
        </label>
      </div>
      <div className="relative rounded-md border border-border-subtle bg-bg/50 p-2">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full"
          onMouseMove={(e) => {
            const svg = e.currentTarget;
            const rect = svg.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * width;
            const frac = (x - padL) / plotW;
            if (frac < 0 || frac > 1) {
              setHoverIdx(null);
              return;
            }
            const i = Math.round(frac * (history.length - 1));
            setHoverIdx(Math.max(0, Math.min(history.length - 1, i)));
          }}
          onMouseLeave={() => setHoverIdx(null)}
        >
          <rect x={padL} y={padT} width={plotW} height={plotH} fill="transparent" />
          {/* Epoch dividers */}
          {epochMarkers.map(({ i, epoch }) => (
            <line
              key={`ep-${i}`}
              x1={xForIdx(i)}
              x2={xForIdx(i)}
              y1={padT}
              y2={padT + plotH}
              stroke="rgb(var(--color-border))"
              strokeWidth={1}
              strokeDasharray="3 3"
            >
              <title>epoch {epoch}</title>
            </line>
          ))}
          {/* Loss line: when smoothing > 0, dim the raw line and draw
              the smoothed line on top (TensorBoard convention). When
              smoothing = 0, the smoothed series equals the raw series
              and we only draw one bold line. */}
          {paths.map((d, i) => (
            <path
              key={`raw-${i}`}
              d={d}
              fill="none"
              stroke="rgb(var(--color-accent))"
              strokeWidth={smoothing > 0 ? 1 : 1.5}
              strokeOpacity={smoothing > 0 ? 0.35 : 1}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
          {smoothing > 0 &&
            smoothedPaths.map((d, i) => (
              <path
                key={`smooth-${i}`}
                d={d}
                fill="none"
                stroke="rgb(var(--color-accent))"
                strokeWidth={1.75}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          {/* Hover crosshair */}
          {hoverIdx !== null && (
            <line
              x1={xForIdx(hoverIdx)}
              x2={xForIdx(hoverIdx)}
              y1={padT}
              y2={padT + plotH}
              stroke="rgb(var(--color-fg-muted))"
              strokeWidth={1}
            />
          )}
          {/* Y-axis labels */}
          <text
            x={padL - 4}
            y={padT + 8}
            textAnchor="end"
            fontSize={10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {formatLoss(loMax)}
          </text>
          <text
            x={padL - 4}
            y={padT + plotH - 2}
            textAnchor="end"
            fontSize={10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {formatLoss(loMin)}
          </text>
          {/* X-axis labels */}
          <text
            x={padL}
            y={height - 6}
            fontSize={10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {xMin}
          </text>
          <text
            x={width - padR}
            y={height - 6}
            textAnchor="end"
            fontSize={10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {xMax}
          </text>
        </svg>
        {hovered && (
          <div
            className={clsx(
              "pointer-events-none absolute top-2 rounded-md border border-border-subtle bg-bg-card px-2 py-1 font-mono text-[11px] shadow-lg",
            )}
            style={{
              left: `${(xForIdx(hoverIdx!) / width) * 100}%`,
              transform: "translateX(8px)",
            }}
          >
            <div>
              <span className="text-fg-subtle">step </span>
              <span className="text-fg">{hovered.step}</span>
            </div>
            <div>
              <span className="text-fg-subtle">loss </span>
              <span className="text-fg">{formatLoss(hovered.loss)}</span>
            </div>
            {smoothing > 0 && smoothedSeries[hoverIdx!] !== null && (
              <div>
                <span className="text-fg-subtle">smoothed </span>
                <span className="text-fg">
                  {formatLoss(smoothedSeries[hoverIdx!]!)}
                </span>
              </div>
            )}
            {hovered.epoch !== undefined && (
              <div>
                <span className="text-fg-subtle">epoch </span>
                <span className="text-fg">{hovered.epoch}</span>
              </div>
            )}
            {hovered.time !== undefined && (
              <div>
                <span className="text-fg-subtle">t </span>
                <span className="text-fg">{hovered.time.toFixed(1)}s</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface ChartComputation {
  paths: string[];
  smoothedPaths: string[];
  smoothedSeries: (number | null)[];
  epochMarkers: { i: number; epoch: number }[];
  loMin: number;
  loMax: number;
  xMin: number;
  xMax: number;
}

function computeChart(
  history: TrainingHistoryPoint[],
  logY: boolean,
  smoothing: number,
): ChartComputation {
  // useMemo runs before the early-return guard in LossChart, so this function
  // must tolerate an empty / single-sample history without crashing. The
  // returned shape is discarded in those cases — the guard renders the
  // "Waiting for training samples…" placeholder instead.
  if (history.length === 0) {
    return {
      paths: [],
      smoothedPaths: [],
      smoothedSeries: [],
      epochMarkers: [],
      loMin: 0,
      loMax: 0,
      xMin: 0,
      xMax: 0,
    };
  }

  const width = 800;
  const height = 220;
  const padL = 40;
  const padR = 8;
  const padT = 8;
  const padB = 24;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  // Transform loss values (handle log scale). A non-positive loss drops out
  // of the log-scale plot; we keep it as a gap in the polyline.
  const transform = (loss: number) =>
    logY ? (loss > 0 ? Math.log10(loss) : null) : loss;

  const rawLosses = history.map((h) => h.loss);
  const smoothedSeries = applyEma(rawLosses, smoothing);

  const ys = [
    ...rawLosses.map(transform),
    ...smoothedSeries.map((v) => (v === null ? null : transform(v))),
  ].filter((v): v is number => v !== null && Number.isFinite(v));

  const loMin = Math.min(...ys);
  const loMax = Math.max(...ys);
  const range = loMax - loMin || 1;

  const xForIdx = (i: number) =>
    padL + (history.length === 1 ? 0 : (i / (history.length - 1)) * plotW);
  const yForLoss = (loss: number | null) => {
    if (loss === null) return null;
    const t = transform(loss);
    if (t === null) return null;
    return padT + plotH - ((t - loMin) / range) * plotH;
  };

  const buildPaths = (series: (number | null)[]): string[] => {
    const out: string[] = [];
    let current: string[] = [];
    series.forEach((v, i) => {
      const y = yForLoss(v);
      if (y === null) {
        if (current.length >= 2) out.push(current.join(" "));
        current = [];
        return;
      }
      const x = xForIdx(i);
      current.push(
        `${current.length === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`,
      );
    });
    if (current.length >= 2) out.push(current.join(" "));
    return out;
  };

  const paths = buildPaths(rawLosses);
  const smoothedPaths = smoothing > 0 ? buildPaths(smoothedSeries) : [];

  const epochMarkers: { i: number; epoch: number }[] = [];
  let lastEpoch: number | undefined;
  history.forEach((h, i) => {
    if (h.epoch !== undefined && lastEpoch !== undefined && h.epoch !== lastEpoch) {
      epochMarkers.push({ i, epoch: h.epoch });
    }
    lastEpoch = h.epoch;
  });

  return {
    paths,
    smoothedPaths,
    smoothedSeries,
    epochMarkers,
    loMin: logY ? Math.pow(10, loMin) : loMin,
    loMax: logY ? Math.pow(10, loMax) : loMax,
    xMin: history[0].step,
    xMax: history[history.length - 1].step,
  };
}

/**
 * TensorBoard-style EMA with debiasing.
 *
 *   smoothed[i] = w * smoothed[i-1] + (1 - w) * raw[i]
 *
 * The naive recurrence is biased toward zero for the first few
 * samples (the seed is 0). TensorBoard divides by `1 - w^(i+1)` to
 * correct this — without it, smoothing=0.95 makes the curve start
 * visibly below the raw values. Non-finite samples produce a `null`
 * in the output but don't reset the state — same convention as the
 * raw-line gap handling.
 */
export function applyEma(
  values: number[],
  weight: number,
): (number | null)[] {
  if (weight <= 0) {
    return values.map((v) => (Number.isFinite(v) ? v : null));
  }
  const out: (number | null)[] = new Array(values.length).fill(null);
  let acc = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      out[i] = null;
      continue;
    }
    acc = weight * acc + (1 - weight) * v;
    n += 1;
    const debias = 1 - Math.pow(weight, n);
    out[i] = debias > 0 ? acc / debias : v;
  }
  return out;
}

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function formatLoss(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

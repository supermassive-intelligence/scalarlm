/**
 * Service-log tailer. Hits GET /v1/health/logs/{service_name}, which
 * serves newline-delimited JSON lines of the shape
 *
 *     {"line": "...", "line_number": N}\n
 *
 * The generator finishes at EOF rather than tailing forever, so we
 * reconnect from `line_number + 1` after a short delay.
 *
 * Large log files (100k+ lines): the initial connect passes `tail=N`
 * so the server seeks to the last N lines instead of dumping the full
 * history. After the first response the client resumes from
 * `starting_line_number=<last+1>` as usual.
 *
 * Mirrors tailTrainingLogs in api/training.ts; both share
 * streamNdjsonLogOnce for the actual parse work.
 */

import { getApiConfig } from "./config";
import { streamNdjsonLogOnce, type LogLine } from "./training";

export type ServiceName = "api" | "vllm" | "megatron";

export type LogStreamStatus =
  | "connecting"
  | "streaming"
  | "reconnecting"
  | "closed";

export interface ServiceLogTailOptions {
  service: ServiceName;
  signal: AbortSignal;
  onLine: (line: LogLine) => void;
  onStatus?: (status: LogStreamStatus) => void;
  /** Delay between reconnects once the generator hits EOF. Default 2 s. */
  tailReopenDelayMs?: number;
  /**
   * Number of trailing lines the first connect requests via `tail=N`.
   * Subsequent reconnects resume from `starting_line_number=<last+1>`.
   * Default 5000, which covers the recent history operators care about
   * without downloading megabytes of old output.
   */
  initialTail?: number;
}

export async function tailServiceLogs({
  service,
  signal,
  onLine,
  onStatus,
  tailReopenDelayMs = 2_000,
  initialTail = 5_000,
}: ServiceLogTailOptions): Promise<void> {
  let lineNumber = 0;
  let firstPass = true;

  while (!signal.aborted) {
    onStatus?.(firstPass ? "connecting" : "reconnecting");
    try {
      const { api_base } = getApiConfig();
      const params = new URLSearchParams();
      if (firstPass && initialTail > 0) {
        params.set("tail", String(initialTail));
      } else {
        params.set("starting_line_number", String(lineNumber));
      }
      const url = `${api_base}/health/logs/${encodeURIComponent(service)}?${params}`;
      lineNumber = await streamNdjsonLogOnce(url, lineNumber, signal, (ln) => {
        onStatus?.("streaming");
        onLine(ln);
      });
      firstPass = false;
    } catch (err) {
      if (signal.aborted || (err as { name?: string }).name === "AbortError")
        break;
      // Fall through to the reconnect delay; service logs tail forever,
      // so transient server failures just retry.
    }

    onStatus?.("reconnecting");
    await sleep(tailReopenDelayMs, signal);
  }

  onStatus?.("closed");
}

/**
 * One-shot fetch of a fixed slice of a log file — the scrollback path.
 * Requests `limit` lines starting at `startingLineNumber` and returns
 * the parsed array in order. Does not follow the tail.
 */
export async function fetchServiceLogRange(
  service: ServiceName,
  startingLineNumber: number,
  limit: number,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const { api_base } = getApiConfig();
  const params = new URLSearchParams({
    starting_line_number: String(Math.max(0, startingLineNumber)),
    limit: String(limit),
  });
  const url = `${api_base}/health/logs/${encodeURIComponent(service)}?${params}`;
  const out: LogLine[] = [];
  await streamNdjsonLogOnce(
    url,
    startingLineNumber,
    signal ?? new AbortController().signal,
    (ln) => out.push(ln),
  );
  return out;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
}

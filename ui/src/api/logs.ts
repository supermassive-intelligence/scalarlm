/**
 * Service-log tailer. Hits GET /v1/health/logs/{service_name}, which
 * serves newline-delimited JSON lines of the shape
 *
 *     {"line": "...", "line_number": N}\n
 *
 * The generator finishes at EOF rather than tailing forever, so we
 * reconnect from `line_number + 1` after a short delay.
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
}

export async function tailServiceLogs({
  service,
  signal,
  onLine,
  onStatus,
  tailReopenDelayMs = 2_000,
}: ServiceLogTailOptions): Promise<void> {
  let lineNumber = 0;

  while (!signal.aborted) {
    onStatus?.(lineNumber === 0 ? "connecting" : "reconnecting");
    try {
      const { api_base } = getApiConfig();
      const url =
        `${api_base}/health/logs/${encodeURIComponent(service)}` +
        `?starting_line_number=${lineNumber}`;
      lineNumber = await streamNdjsonLogOnce(url, lineNumber, signal, (ln) => {
        onStatus?.("streaming");
        onLine(ln);
      });
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

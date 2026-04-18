import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

/**
 * Response shape of GET /v1/generate/metrics.
 *
 * Note: `total_time` and the derived rates (token/s, request/s, flop/s) are
 * utilization-adjusted — the denominator is the accumulated time the queue
 * was non-empty, not wall-clock time. The UI surfaces this caveat in a
 * tooltip on the throughput card.
 */
export interface GenerateMetrics {
  queue_depth: number;
  requests: number;
  tokens: number;
  total_time: number;
  "token/s": number;
  "request/s": number;
  "flop/s": number;
}

export function useGenerateMetrics() {
  return useQuery<GenerateMetrics>({
    queryKey: ["generate-metrics"],
    queryFn: () => apiFetch<GenerateMetrics>("/generate/metrics"),
    refetchInterval: 3_000,
    staleTime: 0,
  });
}

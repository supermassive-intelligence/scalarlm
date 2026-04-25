/**
 * Publish-to-HuggingFace API surface.
 *
 * Phase 0 lands the checkpoint listing only — the modal needs it to
 * populate its dropdown, and the read endpoint is harmless to land
 * before the publish flow exists. The publish mutation + SSE consumer
 * (Phase 1+) will live alongside this hook in subsequent commits.
 *
 * See ui/docs/publish-to-hf.md for the full design.
 */

import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

export interface CheckpointEntry {
  name: string;
  step: number;
  /** Float seconds since epoch — same convention as TrainingJobStatus.start_time. */
  mtime: number;
}

interface CheckpointsResponse {
  checkpoints: CheckpointEntry[];
}

/**
 * GET /v1/megatron/train/{job_hash}/checkpoints
 *
 * Returns every `checkpoint_<step>.pt` in the job directory, newest
 * step first. Empty list while the job is still QUEUED or has no
 * step-checkpoints written yet.
 */
export function useCheckpoints(jobHash: string | undefined) {
  return useQuery<CheckpointEntry[]>({
    queryKey: ["checkpoints", jobHash],
    enabled: Boolean(jobHash),
    queryFn: async () =>
      (
        await apiFetch<CheckpointsResponse>(
          `/megatron/train/${jobHash}/checkpoints`,
        )
      ).checkpoints,
    staleTime: 30_000,
  });
}

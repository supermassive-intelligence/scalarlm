import { useGenerateMetrics } from "@/api/metrics";
import { Card } from "@/components/Card";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { Sparkline } from "@/components/Sparkline";
import { Stat } from "@/components/Stat";
import { formatNumber } from "@/lib/format";
import { useRollingBuffer } from "@/lib/useRollingBuffer";

interface ThroughputCardProps {
  /** Max samples retained for the sparkline. Governed by the page's time-range control. */
  sparklineCapacity: number;
}

const UTILIZATION_TOOLTIP =
  "Throughput while the queue was non-empty. Not wall-clock throughput. " +
  "See docs/inference-queue.md §6.2.";

export function ThroughputCard({ sparklineCapacity }: ThroughputCardProps) {
  const { data, error, refetch, isPending } = useGenerateMetrics();
  const tokenHistory = useRollingBuffer(data?.["token/s"] ?? null, sparklineCapacity);

  return (
    <Card title="Inference throughput" subtitle="GET /v1/generate/metrics">
      {isPending && !data ? (
        <ThroughputSkeleton />
      ) : error ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : data ? (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat
              label="Token/s"
              value={formatNumber(data["token/s"])}
              hint={UTILIZATION_TOOLTIP}
            />
            <Stat
              label="Request/s"
              value={formatNumber(data["request/s"])}
              hint={UTILIZATION_TOOLTIP}
            />
            <Stat
              label="Flop/s"
              value={formatNumber(data["flop/s"])}
              hint={UTILIZATION_TOOLTIP}
            />
            <Stat
              label="Queue depth"
              value={formatNumber(data.queue_depth)}
              hint="Outstanding inference requests"
            />
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between text-xs text-fg-muted">
              <span>token/s history</span>
              <span className="font-mono">{tokenHistory.length} samples</span>
            </div>
            <div className="w-full overflow-hidden rounded-md border border-border-subtle bg-bg/50 p-2">
              <Sparkline
                values={tokenHistory}
                width={640}
                height={56}
                label="token/s over time"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 border-t border-border-subtle pt-4 sm:grid-cols-3">
            <Stat label="Requests completed" value={formatNumber(data.requests)} />
            <Stat label="Tokens completed" value={formatNumber(data.tokens)} />
            <Stat
              label="Effective seconds"
              value={formatNumber(data.total_time)}
              hint="Accumulated time the queue was non-empty"
            />
          </div>
          <p className="text-xs leading-relaxed text-fg-subtle">
            <span className="font-semibold text-fg-muted">Note.</span>{" "}
            {UTILIZATION_TOOLTIP}
          </p>
        </div>
      ) : null}
    </Card>
  );
}

function ThroughputSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="flex flex-col gap-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-7 w-20" />
          </div>
        ))}
      </div>
      <Skeleton className="h-16 w-full" />
    </div>
  );
}

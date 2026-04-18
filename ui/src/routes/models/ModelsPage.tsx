import { PageHeader } from "@/components/PageHeader";
import { Placeholder } from "@/components/Placeholder";

export function ModelsPage() {
  return (
    <>
      <PageHeader title="Models" subtitle="Base model and registered adapters" />
      <Placeholder title="Model list" phase="5">
        <p>
          Lists <code className="font-mono">/v1/models</code>, showing the base
          model and every auto-registered post-training adapter with nickname
          editing.
        </p>
      </Placeholder>
    </>
  );
}

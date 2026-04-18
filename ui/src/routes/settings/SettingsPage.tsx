import { useState, useSyncExternalStore } from "react";

import { getApiConfig } from "@/api/config";
import { useHealth } from "@/api/health";
import { Card } from "@/components/Card";
import { ConfirmDestructive } from "@/components/ConfirmDestructive";
import { CopyButton } from "@/components/CopyButton";
import { PageHeader } from "@/components/PageHeader";
import { StatusDot } from "@/components/StatusDot";
import {
  getThemeMode,
  resolveTheme,
  setThemeMode,
  subscribeTheme,
  type ThemeMode,
} from "@/stores/theme";

export function SettingsPage() {
  const api = getApiConfig();
  const { data: health } = useHealth();
  const [askClear, setAskClear] = useState(false);

  // Deployment-level URL is whatever the browser was loaded from. The api_base
  // is relative ("/v1"), so strip it to get the hostname an external SDK would
  // point at.
  const externalUrl =
    api.api_base === "/v1"
      ? window.location.origin
      : api.api_base.replace(/\/v1\/?$/, "");

  const clearLocal = () => {
    try {
      localStorage.clear();
    } catch {
      // ignore
    }
    try {
      const req = indexedDB.deleteDatabase("scalarlm-chat");
      req.onblocked = () => {
        /* ignore — will delete on next app restart */
      };
    } catch {
      // ignore
    }
    setAskClear(false);
    // Hard reload so hooks refresh against the cleared store.
    window.location.reload();
  };

  return (
    <>
      <PageHeader title="Settings" subtitle="Deployment info and local preferences" />
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-6 py-6">
        <Card title="Deployment">
          <dl className="flex flex-col gap-2 text-sm">
            <Row
              label="API base URL"
              value={
                <span className="flex items-center gap-2 font-mono text-xs">
                  <span>{externalUrl}</span>
                  <CopyButton value={externalUrl} label="Copy" />
                </span>
              }
            />
            <Row
              label="Version"
              value={
                <span className="font-mono text-xs">{api.version}</span>
              }
            />
            <Row
              label="Default model"
              value={
                <span className="font-mono text-xs">{api.default_model}</span>
              }
            />
            {health && (
              <Row
                label="Health"
                value={
                  <div className="flex flex-wrap gap-1.5">
                    {health.components.map((c) => (
                      <StatusDot key={c.name} state={c.state} label={c.name} />
                    ))}
                  </div>
                }
              />
            )}
          </dl>
        </Card>

        <Card
          title="SDK usage"
          subtitle="Drive this deployment from Python"
        >
          <div className="flex flex-col gap-2">
            <pre className="overflow-x-auto rounded-md border border-border-subtle bg-bg p-3 font-mono text-xs leading-relaxed text-fg">
              {sdkSnippet(externalUrl)}
            </pre>
            <div>
              <CopyButton
                value={sdkSnippet(externalUrl)}
                label="Copy snippet"
              />
            </div>
          </div>
        </Card>

        <Card title="Appearance">
          <ThemeSetting />
        </Card>

        <Card title="Local data" className="border-danger/30">
          <div className="flex flex-col gap-3 text-sm">
            <p className="text-fg-muted">
              Conversations live in IndexedDB, and preferences / aliases in
              localStorage. Both are strictly client-side; the server has no
              copy. Clearing wipes chat history, aliases, and submit-form
              defaults.
            </p>
            <button
              type="button"
              onClick={() => setAskClear(true)}
              className="w-fit rounded-md border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger hover:border-danger hover:bg-danger/20"
            >
              Clear all local data
            </button>
          </div>
        </Card>
      </div>

      <ConfirmDestructive
        open={askClear}
        title="Clear all local data"
        description="Deletes every conversation, every alias, and every saved form default. The page will reload. This cannot be undone."
        confirmationText="clear"
        actionLabel="Clear"
        onConfirm={clearLocal}
        onClose={() => setAskClear(false)}
      />
    </>
  );
}

function Row({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border-subtle py-1.5 last:border-b-0">
      <dt className="text-xs uppercase tracking-wider text-fg-subtle">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function sdkSnippet(apiUrl: string): string {
  return [
    `import scalarlm`,
    `scalarlm.api_url = ${JSON.stringify(apiUrl)}`,
    ``,
    `llm = scalarlm.SupermassiveIntelligence()`,
    `results = llm.generate(prompts=["hello"])`,
    `print(results)`,
  ].join("\n");
}

const THEME_OPTIONS: { value: ThemeMode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "Follow OS preference" },
  { value: "light", label: "Light", hint: "" },
  { value: "dark", label: "Dark", hint: "" },
];

function ThemeSetting() {
  const mode = useSyncExternalStore(subscribeTheme, getThemeMode, getThemeMode);
  const resolved = resolveTheme(mode);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <div>
        <div className="text-fg">Theme</div>
        <div className="text-xs text-fg-muted">
          Currently rendering in{" "}
          <span className="font-mono">{resolved}</span> mode.
        </div>
      </div>
      <div className="flex items-center gap-1">
        {THEME_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => setThemeMode(opt.value)}
            className={
              mode === opt.value
                ? "rounded-md bg-bg-hover px-3 py-1 text-xs text-fg ring-1 ring-border"
                : "rounded-md px-3 py-1 text-xs text-fg-muted hover:bg-bg-card hover:text-fg"
            }
            title={opt.hint || undefined}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

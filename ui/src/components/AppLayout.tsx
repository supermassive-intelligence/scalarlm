import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

import { useKeyboardShortcuts } from "@/lib/useKeyboardShortcuts";

import { KeyboardCheatsheet } from "./KeyboardCheatsheet";

const navItems = [
  { to: "/chat", label: "Chat" },
  { to: "/train", label: "Train" },
  { to: "/metrics", label: "Metrics" },
  { to: "/models", label: "Models" },
];

export function AppLayout() {
  const [cheatsheetOpen, setCheatsheetOpen] = useState(false);
  useKeyboardShortcuts({ onShowCheatsheet: () => setCheatsheetOpen(true) });

  return (
    <div className="flex h-full flex-col">
      <TopBar onShowCheatsheet={() => setCheatsheetOpen(true)} />
      <main className="min-h-0 flex-1 overflow-auto">
        <Outlet />
      </main>
      <KeyboardCheatsheet
        open={cheatsheetOpen}
        onClose={() => setCheatsheetOpen(false)}
      />
    </div>
  );
}

function TopBar({ onShowCheatsheet }: { onShowCheatsheet: () => void }) {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-border-subtle bg-bg px-4">
      <div className="flex items-center gap-6">
        <NavLink to="/" className="text-sm font-semibold tracking-tight">
          ScalarLM
        </NavLink>
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                clsx(
                  "rounded-md px-3 py-1.5 text-sm transition-colors",
                  isActive
                    ? "bg-bg-hover text-fg"
                    : "text-fg-muted hover:bg-bg-card hover:text-fg",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={onShowCheatsheet}
          className="rounded-md px-2 py-1.5 text-xs text-fg-muted hover:bg-bg-card hover:text-fg"
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts (?)"
        >
          ?
        </button>
        <NavLink
          to="/settings"
          className="rounded-md px-2 py-1.5 text-sm text-fg-muted hover:bg-bg-card hover:text-fg"
          aria-label="Settings"
          title="Settings"
        >
          ⚙
        </NavLink>
      </div>
    </header>
  );
}

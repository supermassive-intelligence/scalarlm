/**
 * Sampling-parameter store (currently just `temperature`).
 *
 * Default is 0 for deterministic output — the user can bump it from the
 * settings page. Temperature is read from this store at send-time by the
 * chat view, so a slider change takes effect on the next message without
 * needing the view to re-render.
 *
 * Mirrors the shape of stores/theme.ts: a typed value + get/set/subscribe
 * for useSyncExternalStore.
 */

import { readJson, writeJson } from "@/lib/preferences";

const KEY = "sampling.temperature";

export const TEMPERATURE_MIN = 0;
export const TEMPERATURE_MAX = 2;
export const TEMPERATURE_DEFAULT = 0;

const listeners = new Set<() => void>();

let current: number = sanitize(
  readJson<number>(KEY, TEMPERATURE_DEFAULT),
);

function sanitize(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return TEMPERATURE_DEFAULT;
  if (n < TEMPERATURE_MIN) return TEMPERATURE_MIN;
  if (n > TEMPERATURE_MAX) return TEMPERATURE_MAX;
  return n;
}

export function getTemperature(): number {
  return current;
}

export function setTemperature(value: number): void {
  const next = sanitize(value);
  if (next === current) return;
  current = next;
  writeJson(KEY, next);
  for (const l of listeners) {
    try {
      l();
    } catch {
      // continue
    }
  }
}

export function subscribeTemperature(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

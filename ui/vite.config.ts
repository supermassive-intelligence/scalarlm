import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production bundle is served by FastAPI at /app/. Hashed assets live under
// /app/assets/ and are cached indefinitely; index.html is served with no-cache.
// In dev, Vite runs on :5173 and proxies API traffic to a running ScalarLM pod.
export default defineConfig(({ mode }) => ({
  base: "/app/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
    emptyOutDir: true,
    sourcemap: mode === "development",
  },
  server: {
    port: 5173,
    proxy: {
      "/v1": {
        target: process.env.SCALARLM_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
      "/app/api-config.json": {
        target: process.env.SCALARLM_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
}));

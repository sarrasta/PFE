import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy only — in production, nginx (see Dockerfile/nginx.conf)
// reverse-proxies /api to the backend container instead.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET || "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});

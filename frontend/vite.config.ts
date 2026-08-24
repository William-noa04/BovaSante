import path from "node:path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "client", "src") } },
  root: path.resolve(import.meta.dirname, "client"),
  envDir: import.meta.dirname,
  build: { outDir: path.resolve(import.meta.dirname, "dist"), emptyOutDir: true },
  server: { host: true, port: 3000 },
});

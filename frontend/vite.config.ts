import { defineConfig } from "vite";

export default defineConfig({
  root: "./",
  server: {
    port: 5173,
    open: false
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/setupTests.ts"
  }
});

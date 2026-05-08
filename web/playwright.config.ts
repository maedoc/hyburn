import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  retries: 0,
  use: {
    baseURL: "http://localhost:8080",
  },
  webServer: {
    command: "python3 -m http.server 8080",
    port: 8080,
    reuseExistingServer: true,
    cwd: ".",
  },
});

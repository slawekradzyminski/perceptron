import { initApp } from "./app";

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initApp());
} else {
  initApp();
}

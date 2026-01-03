import { useEffect } from "react";

type HotkeysConfig = {
  onStep: () => void;
  onReset: () => void;
  enabled?: boolean;
};

export function useHotkeys({ onStep, onReset, enabled = true }: HotkeysConfig) {
  useEffect(() => {
    if (!enabled) return () => undefined;
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      const key = event.key.toLowerCase();
      if (key === "s") onStep();
      if (key === "r") onReset();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onStep, onReset]);
}

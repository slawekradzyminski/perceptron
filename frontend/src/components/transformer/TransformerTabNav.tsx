import type { TransformerTab } from "./types";

type TransformerTabNavProps = {
  activeTab: TransformerTab;
  onTabChange: (tab: TransformerTab) => void;
};

export function TransformerTabNav({
  activeTab,
  onTabChange,
}: TransformerTabNavProps) {
  return (
    <div className="tab-nav">
      <button
        className={activeTab === "tokenize" ? "active" : ""}
        onClick={() => onTabChange("tokenize")}
      >
        Tokenization
      </button>
      <button
        className={activeTab === "trace" ? "active" : ""}
        onClick={() => onTabChange("trace")}
      >
        Block Trace
      </button>
      <button
        className={activeTab === "attention" ? "active" : ""}
        onClick={() => onTabChange("attention")}
      >
        Attention
      </button>
      <button
        className={activeTab === "logit-lens" ? "active" : ""}
        onClick={() => onTabChange("logit-lens")}
      >
        Logit Lens
      </button>
      <button
        className={activeTab === "kv-cache" ? "active" : ""}
        onClick={() => onTabChange("kv-cache")}
      >
        KV Cache
      </button>
      <button
        className={activeTab === "generate" ? "active" : ""}
        onClick={() => onTabChange("generate")}
      >
        Generation
      </button>
      <button
        className={activeTab === "scale" ? "active" : ""}
        onClick={() => onTabChange("scale")}
      >
        Model Scale
      </button>
    </div>
  );
}

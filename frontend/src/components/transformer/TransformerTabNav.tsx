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

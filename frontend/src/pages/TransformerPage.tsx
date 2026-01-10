import { useEffect, useState, useCallback } from "react";
import { useTransformerApi } from "../hooks/transformer/useTransformerApi";
import { TransformerEducation } from "../components/education/TransformerEducation";
import { TransformerInputSection } from "../components/transformer/TransformerInputSection";
import { TransformerTabNav } from "../components/transformer/TransformerTabNav";
import { TransformerTokenizeTab } from "../components/transformer/TransformerTokenizeTab";
import { TransformerTraceTab } from "../components/transformer/TransformerTraceTab";
import { TransformerChatTab } from "../components/transformer/TransformerChatTab";
import { TransformerScaleTab } from "../components/transformer/TransformerScaleTab";
import type { TransformerTab } from "../components/transformer/types";
import "../styles/transformer.css";
import "../styles/education.css";

export function TransformerPage({ apiBase }: { apiBase: string }) {
  const {
    tokens,
    embedInfo,
    trace,
    chatMessages,
    streamingContent,
    ollamaStatus,
    scaleModels,
    comparison,
    growth,
    error,
    loading,
    generating,
    tokenize,
    getEmbedInfo,
    getTrace,
    sendChat,
    resetChat,
    fetchOllamaStatus,
    fetchScaleModels,
    compareModels,
    fetchGrowth,
  } = useTransformerApi(apiBase);

  const [inputText, setInputText] = useState("The quick brown fox jumps");
  const [activeTab, setActiveTab] = useState<TransformerTab>("tokenize");

  useEffect(() => {
    void fetchScaleModels();
    void fetchGrowth();
    void fetchOllamaStatus();
  }, [fetchScaleModels, fetchGrowth, fetchOllamaStatus]);

  const handleTokenize = useCallback(() => {
    void tokenize(inputText);
    void getEmbedInfo(inputText);
  }, [tokenize, getEmbedInfo, inputText]);

  const handleTrace = useCallback(
    (nBlocks: number) => {
      void getTrace(inputText, nBlocks);
    },
    [getTrace, inputText],
  );

  const handleCompare = useCallback(
    (model1: string, model2: string) => {
      void compareModels(model1, model2);
    },
    [compareModels],
  );

  return (
    <section className="panel transformer-panel">
      <header className="transformer-header">
        <h2>Transformer Flow</h2>
        <p className="subtitle">Understanding GPT — From text to tokens to predictions</p>
      </header>

      <div className="transformer-grid">
        <TransformerInputSection
          inputText={inputText}
          onInputChange={setInputText}
        />

        <TransformerTabNav activeTab={activeTab} onTabChange={setActiveTab} />

        <div className="tab-content">
          {activeTab === "tokenize" && (
            <TransformerTokenizeTab
              inputText={inputText}
              loading={loading}
              tokens={tokens}
              embedInfo={embedInfo}
              onTokenize={handleTokenize}
            />
          )}

          {activeTab === "trace" && (
            <TransformerTraceTab
              inputText={inputText}
              loading={loading}
              trace={trace}
              onTrace={handleTrace}
            />
          )}

          {activeTab === "generate" && (
            <TransformerChatTab
              ollamaStatus={ollamaStatus}
              chatMessages={chatMessages}
              streamingContent={streamingContent}
              generating={generating}
              error={error}
              onSend={sendChat}
              onReset={resetChat}
            />
          )}

          {activeTab === "scale" && (
            <TransformerScaleTab
              scaleModels={scaleModels}
              comparison={comparison}
              growth={growth}
              loading={loading}
              onCompare={handleCompare}
            />
          )}
        </div>
      </div>

      <TransformerEducation />

      {error && <p className="transformer-error">{error}</p>}
    </section>
  );
}

import { useEffect, useState, useCallback } from "react";
import { useLocation, useNavigate } from "react-router";
import { useTransformerApi } from "../hooks/transformer/useTransformerApi";
import { useGlassBoxApi } from "../hooks/transformer/useGlassBoxApi";
import { TransformerEducation } from "../components/education/TransformerEducation";
import { TransformerInputSection } from "../components/transformer/TransformerInputSection";
import { TransformerTabNav } from "../components/transformer/TransformerTabNav";
import { TransformerTokenizeTab } from "../components/transformer/TransformerTokenizeTab";
import { TransformerTraceTab } from "../components/transformer/TransformerTraceTab";
import { TransformerChatTab } from "../components/transformer/TransformerChatTab";
import { TransformerScaleTab } from "../components/transformer/TransformerScaleTab";
import { AttentionTab } from "../components/transformer/AttentionTab";
import { LogitLensTab } from "../components/transformer/LogitLensTab";
import { KVCacheTab } from "../components/transformer/KVCacheTab";
import type { TransformerTab } from "../components/transformer/types";
import "../styles/transformer.css";
import "../styles/education.css";

const TAB_ROUTES: Record<string, TransformerTab> = {
  "tokenization": "tokenize",
  "block-trace": "trace",
  "attention": "attention",
  "logit-lens": "logit-lens",
  "kv-cache": "kv-cache",
  "generation": "generate",
  "model-scale": "scale",
};

const TAB_TO_ROUTE: Record<TransformerTab, string> = {
  "tokenize": "tokenization",
  "trace": "block-trace",
  "attention": "attention",
  "logit-lens": "logit-lens",
  "kv-cache": "kv-cache",
  "generate": "generation",
  "scale": "model-scale",
};

export function TransformerPage({ apiBase }: { apiBase: string }) {
  const location = useLocation();
  const navigate = useNavigate();

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
    error: transformerError,
    loading: transformerLoading,
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

  const {
    attention,
    logitLens,
    kvCache,
    error: glassBoxError,
    loading: glassBoxLoading,
    fetchAttention,
    fetchLogitLens,
    fetchKvCache,
  } = useGlassBoxApi(apiBase);

  const [inputText, setInputText] = useState("The American flag is red, white, and");

  // Parse tab from URL path
  const pathParts = location.pathname.split("/");
  const tabSlug = pathParts[2] || "tokenization";
  const activeTab: TransformerTab = TAB_ROUTES[tabSlug] || "tokenize";

  const setActiveTab = useCallback((tab: TransformerTab) => {
    const route = TAB_TO_ROUTE[tab];
    navigate(`/transformer/${route}`);
  }, [navigate]);

  // Combined loading and error states
  const loading = transformerLoading || glassBoxLoading;
  const error = transformerError || glassBoxError;

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

  const handleFetchAttention = useCallback(
    (text: string) => {
      void fetchAttention(text);
    },
    [fetchAttention],
  );

  const handleFetchLogitLens = useCallback(
    (text: string, topK?: number) => {
      void fetchLogitLens(text, topK);
    },
    [fetchLogitLens],
  );

  const handleFetchKvCache = useCallback(
    (config: Parameters<typeof fetchKvCache>[0]) => {
      void fetchKvCache(config);
    },
    [fetchKvCache],
  );

  return (
    <section className="panel transformer-panel">
      <header className="transformer-header">
        <h2>Transformer Flow</h2>
        <p className="subtitle">Understanding GPT — From text to tokens to predictions</p>
      </header>

      <div className="transformer-grid">
        <TransformerTabNav activeTab={activeTab} onTabChange={setActiveTab} />

        <div className="tab-content">
          {activeTab === "tokenize" && (
            <>
              <TransformerInputSection
                inputText={inputText}
                onInputChange={setInputText}
              />
              <TransformerTokenizeTab
                inputText={inputText}
                loading={loading}
                tokens={tokens}
                embedInfo={embedInfo}
                onTokenize={handleTokenize}
              />
            </>
          )}

          {activeTab === "trace" && (
            <TransformerTraceTab
              inputText={inputText}
              loading={loading}
              trace={trace}
              onTrace={handleTrace}
            />
          )}

          {activeTab === "attention" && (
            <AttentionTab
              inputText={inputText}
              onInputChange={setInputText}
              loading={loading}
              attention={attention}
              onFetchAttention={handleFetchAttention}
            />
          )}

          {activeTab === "logit-lens" && (
            <LogitLensTab
              inputText={inputText}
              onInputChange={setInputText}
              loading={loading}
              logitLens={logitLens}
              onFetchLogitLens={handleFetchLogitLens}
            />
          )}

          {activeTab === "kv-cache" && (
            <KVCacheTab
              loading={loading}
              kvCache={kvCache}
              onFetchKvCache={handleFetchKvCache}
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

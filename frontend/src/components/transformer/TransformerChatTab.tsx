import { useState } from "react";
import type {
  ChatMessage,
  OllamaStatus,
} from "../../hooks/transformer/useTransformerApi";

type TransformerChatTabProps = {
  ollamaStatus: OllamaStatus | null;
  chatMessages: ChatMessage[];
  streamingContent: string;
  generating: boolean;
  error: string | null;
  onSend: (message: string) => void;
  onReset: () => void;
};

export function TransformerChatTab({
  ollamaStatus,
  chatMessages,
  streamingContent,
  generating,
  error,
  onSend,
  onReset,
}: TransformerChatTabProps) {
  const [chatInput, setChatInput] = useState("");
  const ollamaOnline =
    ollamaStatus?.ok === true && ollamaStatus?.model_available === true;

  const handleSend = () => {
    if (chatInput.trim()) {
      onSend(chatInput);
      setChatInput("");
    }
  };

  return (
    <div className="chat-section">
      <div className="ollama-status-row">
        <div className={`ollama-status ${ollamaOnline ? "online" : "offline"}`}>
          <span className={`status-dot ${ollamaOnline ? "ok" : "bad"}`} />
          {ollamaOnline ? "Ollama connected" : "Ollama offline"}
        </div>
        {ollamaOnline && (
          <>
            <div className="model-info-badge llama">
              <span className="model-label">Model:</span>
              <span className="model-name">{ollamaStatus?.model}</span>
            </div>
            {chatMessages.length > 0 && (
              <button onClick={onReset} className="reset-btn">
                Reset Chat
              </button>
            )}
          </>
        )}
      </div>

      {!ollamaOnline && (
        <div className="ollama-offline-notice">
          <div className="offline-icon">⚡</div>
          <h3>Ollama Required</h3>
          <p>
            This tab requires Ollama to be running locally for live LLM chat.
          </p>
          <div className="offline-instructions">
            <p>
              <strong>To get started:</strong>
            </p>
            <ol>
              <li>Install Ollama from <code>ollama.ai</code></li>
              <li>Run <code>ollama serve</code> in a terminal</li>
              <li>Pull the model: <code>ollama pull llama3.2:1b</code></li>
              <li>Refresh this page</li>
            </ol>
            <p>
              <strong>Or with Docker:</strong>
            </p>
            <code>docker run -d -p 11434:11434 ollama/ollama</code>
          </div>
        </div>
      )}

      {ollamaOnline && (
        <div className="chat-container">
          <div className="chat-messages">
            {chatMessages.length === 0 && !generating && (
              <div className="chat-empty">
                <div className="empty-icon">💬</div>
                <p>Start a conversation with {ollamaStatus?.model}</p>
                <p className="empty-hint">
                  Try asking a question or giving a prompt
                </p>
              </div>
            )}

            {chatMessages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.role}`}>
                <div className="message-label">
                  {msg.role === "user" ? "You" : ollamaStatus?.model}
                </div>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}

            {generating && streamingContent && (
              <div className="chat-message assistant streaming">
                <div className="message-label">{ollamaStatus?.model}</div>
                <div className="message-content">
                  {streamingContent}
                  <span className="cursor">▊</span>
                </div>
              </div>
            )}

            {generating && !streamingContent && (
              <div className="chat-message assistant streaming">
                <div className="message-label">{ollamaStatus?.model}</div>
                <div className="message-content typing">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              </div>
            )}

            {error && (
              <div className="chat-error">
                <strong>⚠️ Error:</strong> {error}
              </div>
            )}
          </div>

          <div className="chat-input-area">
            <textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
              rows={2}
              disabled={generating}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <button
              onClick={handleSend}
              disabled={generating || !chatInput.trim()}
              className="send-btn"
            >
              {generating ? "..." : "Send"}
            </button>
          </div>

          <p className="chat-hint">
            Responses stream token-by-token. The model maintains conversation
            history.
          </p>
        </div>
      )}
    </div>
  );
}

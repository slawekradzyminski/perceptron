export function TransformerEducation() {
  return (
    <div className="education-panel transformer-education">
      <h3>Transformers: Attention Is All You Need</h3>

      <div className="education-section">
        <h4>📜 The Paper That Changed Everything (2017)</h4>
        <p>
          In June 2017, Vaswani et al. at Google published "Attention Is All You Need,"
          introducing the <strong>Transformer</strong> architecture. It replaced recurrent
          networks with pure attention, enabling massive parallelization.
        </p>
        <p>
          This paper is the foundation of GPT, BERT, T5, LLaMA, and every major language model.
          It also revolutionized vision (ViT), audio (Whisper), and protein folding (AlphaFold 2).
        </p>
      </div>

      <div className="education-section">
        <h4>🎯 Self-Attention: The Core Innovation</h4>
        <p>
          Self-attention lets each token "look at" every other token in the sequence.
          For each token, we compute:
        </p>
        <p className="formula">
          Attention(Q, K, V) = softmax(QKᵀ / √d) × V
        </p>
        <ul>
          <li><strong>Query (Q):</strong> "What am I looking for?"</li>
          <li><strong>Key (K):</strong> "What do I contain?"</li>
          <li><strong>Value (V):</strong> "What information do I pass along?"</li>
        </ul>
        <p>
          The √d scaling prevents dot products from becoming too large before softmax.
        </p>
      </div>

      <div className="education-section">
        <h4>🎭 Multi-Head Attention</h4>
        <p>
          Instead of one attention operation, Transformers use multiple <strong>heads</strong>
          (typically 8-96). Each head learns different relationship patterns:
        </p>
        <ul>
          <li>Some heads track syntactic dependencies</li>
          <li>Some track semantic similarity</li>
          <li>Some handle coreference (pronouns → nouns)</li>
          <li>Some attend to positional patterns</li>
        </ul>
        <p>Outputs are concatenated and linearly projected.</p>
      </div>

      <div className="education-section">
        <h4>📍 Positional Encoding</h4>
        <p>
          Attention is permutation-invariant — it doesn't know token order. We add
          positional encodings (sinusoidal or learned) to embeddings:
        </p>
        <p className="formula">
          PE(pos, 2i) = sin(pos / 10000^(2i/d))
        </p>
        <p className="formula">
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        </p>
        <p>
          Modern variants use Rotary Position Embeddings (RoPE) or ALiBi for better
          length generalization.
        </p>
      </div>

      <div className="education-section">
        <h4>🏗️ The Full Architecture</h4>
        <p>A Transformer block contains:</p>
        <ul>
          <li><strong>Multi-Head Attention:</strong> Token interactions</li>
          <li><strong>Layer Normalization:</strong> Stabilizes training</li>
          <li><strong>Feed-Forward Network:</strong> 2-layer MLP per position</li>
          <li><strong>Residual Connections:</strong> x + Sublayer(x) for gradient flow</li>
        </ul>
        <p>
          GPT-3 stacks 96 such blocks. The original paper used 6.
        </p>
      </div>

      <div className="education-section">
        <h4>📈 Scaling Laws</h4>
        <p>
          Kaplan et al. (2020) discovered that Transformer performance follows power laws:
        </p>
        <p className="formula">
          L ∝ N^(-0.076) × D^(-0.095) × C^(-0.050)
        </p>
        <p>
          where L is loss, N is parameters, D is data, and C is compute. This predictability
          enabled planning models like GPT-4 years in advance.
        </p>
      </div>
    </div>
  );
}

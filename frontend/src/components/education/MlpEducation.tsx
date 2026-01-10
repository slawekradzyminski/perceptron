export function MlpEducation() {
  return (
    <div className="education-panel mlp-education">
      <h3>Multi-Layer Perceptrons & Backpropagation</h3>

      <div className="education-section">
        <h4>📜 The Breakthrough (1986)</h4>
        <p>
          While the idea of multi-layer networks existed since the 1960s, training them was
          considered intractable. In 1986, Rumelhart, Hinton, and Williams published their
          famous paper showing how <strong>backpropagation</strong> could efficiently train
          deep networks. This revived the field of neural networks.
        </p>
        <p>
          The key insight: the chain rule of calculus allows gradients to flow backward
          through layers, enabling each weight to be adjusted based on its contribution to the error.
        </p>
      </div>

      <div className="education-section">
        <h4>🏗️ Architecture</h4>
        <p>An MLP consists of:</p>
        <ul>
          <li><strong>Input Layer:</strong> Raw features (e.g., pixel values)</li>
          <li><strong>Hidden Layer(s):</strong> Learn intermediate representations</li>
          <li><strong>Output Layer:</strong> Final predictions (classes or values)</li>
        </ul>
        <p>
          Each neuron computes: output = activation(Σ wᵢxᵢ + b). The hidden layers are what
          make MLPs "universal function approximators" — they can learn any continuous function.
        </p>
      </div>

      <div className="education-section">
        <h4>⚡ Activation Functions</h4>
        <p>Non-linear activations are essential — without them, stacking layers would be pointless:</p>
        <ul>
          <li><strong>Sigmoid:</strong> σ(x) = 1/(1+e⁻ˣ) — historical, causes vanishing gradients</li>
          <li><strong>Tanh:</strong> tanh(x) — zero-centered, still vanishes</li>
          <li><strong>ReLU:</strong> max(0, x) — modern default, avoids vanishing gradient</li>
          <li><strong>GELU:</strong> x·Φ(x) — smooth ReLU, used in Transformers</li>
        </ul>
      </div>

      <div className="education-section">
        <h4>🔙 Backpropagation Algorithm</h4>
        <ol>
          <li><strong>Forward Pass:</strong> Compute predictions layer by layer</li>
          <li><strong>Compute Loss:</strong> Measure how wrong we are</li>
          <li><strong>Backward Pass:</strong> Apply chain rule to get ∂Loss/∂w for every weight</li>
          <li><strong>Update:</strong> w ← w − η × ∂Loss/∂w</li>
        </ol>
        <p>
          The elegance is that each layer only needs to know the gradient coming from above
          and its own local derivatives. This scales to networks with billions of parameters.
        </p>
      </div>

      <div className="education-section">
        <h4>🎯 Why Hidden Layers Matter</h4>
        <p>
          A single hidden layer can represent XOR: it creates two linear boundaries whose
          combination produces the non-linear decision surface. Each neuron acts as a
          "feature detector," and the output layer combines these features.
        </p>
        <p>
          This is the fundamental principle of deep learning: hierarchical feature extraction.
        </p>
      </div>
    </div>
  );
}

export function TinyGpsEducation() {
  return (
    <div className="education-panel tinygps-education">
      <h3>TinyGPS: Backpropagation in Practice</h3>

      <div className="education-section">
        <h4>🗺️ The Problem: Learning Geographic Patterns</h4>
        <p>
          TinyGPS demonstrates how neural networks can learn to classify locations.
          Given latitude and longitude coordinates, the network learns to predict
          which city a point belongs to — a non-linear classification problem
          that requires learning complex decision boundaries.
        </p>
      </div>

      <div className="education-section">
        <h4>🔙 Backpropagation: The Chain Rule in Action</h4>
        <p>
          Backpropagation is just the chain rule applied systematically through a computational graph.
          For a two-layer network predicting ŷ:
        </p>
        <p className="formula">
          h = σ(W₁x + b₁)  →  ŷ = W₂h + b₂
        </p>
        <p>To update W₁, we need ∂Loss/∂W₁. The chain rule gives:</p>
        <p className="formula">
          ∂Loss/∂W₁ = ∂Loss/∂ŷ × ∂ŷ/∂h × ∂h/∂W₁
        </p>
        <p>
          Each term is a local derivative. We compute them going forward, then multiply
          going backward. This is the essence of backpropagation.
        </p>
      </div>

      <div className="education-section">
        <h4>📍 Positional Encoding</h4>
        <p>
          Raw coordinates (x, y) have limited expressiveness. Networks struggle to learn
          high-frequency patterns. <strong>Positional encoding</strong> transforms coordinates
          using sinusoidal functions at multiple frequencies:
        </p>
        <p className="formula">
          PE(pos) = [sin(pos/10000^(0)), cos(pos/10000^(0)), sin(pos/10000^(2/d)), ...]
        </p>
        <p>
          This technique, now famous from Transformers, gives the network multiple
          "views" of each position at different scales.
        </p>
      </div>

      <div className="education-section">
        <h4>⚡ The Gradient Flow</h4>
        <p>Watch the backpropagation visualization to see:</p>
        <ul>
          <li><strong>Forward pass:</strong> Input → Hidden activations → Output → Loss</li>
          <li><strong>Backward pass:</strong> Loss gradient → Output layer gradients → Hidden layer gradients</li>
          <li><strong>Weight update:</strong> Each weight moves opposite to its gradient</li>
        </ul>
        <p>
          The magnitude of gradients matters: if they're too small (vanishing) or too large
          (exploding), training fails. This is why activation functions and initialization matter.
        </p>
      </div>

      <div className="education-section">
        <h4>🎯 Why This Architecture Works</h4>
        <p>
          The hidden layer learns to partition the 2D coordinate space into regions.
          Each hidden neuron defines a linear boundary. The output layer combines
          these boundaries into complex shapes that match city territories.
        </p>
        <p>
          This is feature learning: the network automatically discovers useful
          intermediate representations that weren't explicitly programmed.
        </p>
      </div>
    </div>
  );
}

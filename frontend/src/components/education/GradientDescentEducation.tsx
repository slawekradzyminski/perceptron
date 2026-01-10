export function GradientDescentEducation() {
  return (
    <div className="education-panel gd-education">
      <h3>Gradient Descent: The Engine of Learning</h3>

      <div className="education-section">
        <h4>📜 A Mathematical Foundation (1847)</h4>
        <p>
          <strong>Gradient descent</strong> was first described by Augustin-Louis Cauchy in 1847,
          long before computers existed. The idea is simple: to minimize a function, repeatedly
          take small steps in the direction of steepest decrease.
        </p>
        <p>
          This same algorithm, with various modifications, trains every neural network today —
          from the perceptron to GPT-4. It's the universal optimization engine of machine learning.
        </p>
      </div>

      <div className="education-section">
        <h4>⛰️ The Loss Landscape</h4>
        <p>
          Imagine the loss function as a mountainous terrain. Each point represents a possible
          set of weights, and the height is the loss (error). We want to find the lowest valley.
        </p>
        <p>
          In reality, this landscape has millions of dimensions (one per weight) and is highly
          non-convex, with many local minima, saddle points, and flat regions.
        </p>
      </div>

      <div className="education-section">
        <h4>📐 The Update Rule</h4>
        <p>The gradient points uphill. We step in the opposite direction:</p>
        <p className="formula">
          w ← w − η × ∇L(w)
        </p>
        <p>
          where ∇L is the gradient (vector of all partial derivatives) and η is the learning rate.
          This is the core equation — everything else is optimization.
        </p>
      </div>

      <div className="education-section">
        <h4>🎚️ The Learning Rate Dilemma</h4>
        <p>Learning rate (η) is the most important hyperparameter:</p>
        <ul>
          <li><strong>η = 0.1–1:</strong> Often too large — loss may explode</li>
          <li><strong>η = 0.01:</strong> Common for SGD — stable but slow</li>
          <li><strong>η = 0.001:</strong> Default for Adam — works for most problems</li>
          <li><strong>η = 0.0001:</strong> Fine-tuning pre-trained models</li>
        </ul>
        <p>
          Modern optimizers like Adam adapt the learning rate per-parameter, reducing
          the need for manual tuning.
        </p>
      </div>

      <div className="education-section">
        <h4>🎲 Stochastic Gradient Descent (SGD)</h4>
        <p>
          Computing gradients over the entire dataset is expensive. <strong>SGD</strong> uses
          random samples (or mini-batches), trading accuracy for speed. The noise actually helps:
        </p>
        <ul>
          <li>Escapes shallow local minima</li>
          <li>Provides implicit regularization</li>
          <li>Enables online learning from streaming data</li>
        </ul>
      </div>

      <div className="education-section">
        <h4>🔬 Advanced Optimizers</h4>
        <p>Beyond vanilla SGD:</p>
        <ul>
          <li><strong>Momentum:</strong> Accumulates velocity to smooth updates</li>
          <li><strong>RMSprop:</strong> Adapts learning rate per parameter</li>
          <li><strong>Adam:</strong> Combines momentum + RMSprop (most popular)</li>
          <li><strong>AdamW:</strong> Adam with decoupled weight decay (current best practice)</li>
        </ul>
      </div>
    </div>
  );
}

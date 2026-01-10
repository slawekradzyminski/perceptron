export function LmsEducation() {
  return (
    <div className="education-panel lms-education">
      <h3>LMS: The Widrow-Hoff Algorithm</h3>

      <div className="education-section">
        <h4>📜 Historical Context (1960)</h4>
        <p>
          Bernard Widrow and Ted Hoff developed the <strong>Least Mean Squares (LMS)</strong> algorithm
          at Stanford in 1960, just two years after the perceptron. Unlike Rosenblatt's discrete approach,
          they used continuous optimization — the same principle that powers modern deep learning.
        </p>
        <p>
          Their ADALINE (Adaptive Linear Neuron) machine was one of the first neural networks
          to find commercial use, applied in echo cancellation for telephone lines — a technology
          still used in every phone call you make today.
        </p>
      </div>

      <div className="education-section">
        <h4>🎯 The Objective: Minimize Squared Error</h4>
        <p>LMS minimizes the mean squared error between predictions and targets:</p>
        <p className="formula">
          E = ½ × (target − prediction)²
        </p>
        <p>
          Unlike the perceptron (which only updates on mistakes), LMS updates after every sample,
          always pushing predictions closer to targets. The ½ is a convenience factor that
          simplifies the derivative.
        </p>
      </div>

      <div className="education-section">
        <h4>⚡ The Delta Rule</h4>
        <p>The gradient of the error with respect to weights gives the update rule:</p>
        <p className="formula">
          w ← w + η × (target − prediction) × input
        </p>
        <p>
          This is <strong>gradient descent</strong> in its simplest form. The learning rate η
          controls step size. Too large and you overshoot; too small and learning is slow.
        </p>
      </div>

      <div className="education-section">
        <h4>🌊 Stochastic vs Batch Learning</h4>
        <p>
          LMS is a <strong>stochastic</strong> (online) algorithm — it updates after each sample.
          This has advantages:
        </p>
        <ul>
          <li>Memory efficient (no need to store the full dataset)</li>
          <li>Can adapt to changing data in real-time</li>
          <li>Often escapes local minima due to noise</li>
        </ul>
        <p>
          Modern deep learning uses mini-batch gradient descent, a hybrid approach that
          averages gradients over small batches (32-512 samples) before updating.
        </p>
      </div>

      <div className="education-section">
        <h4>🔗 Foundation of Deep Learning</h4>
        <p>
          LMS introduced the core principle underlying all neural network training:
          <strong>compute the error, take the gradient, update in the opposite direction</strong>.
          When generalized to multiple layers via the chain rule, this becomes backpropagation.
        </p>
      </div>
    </div>
  );
}

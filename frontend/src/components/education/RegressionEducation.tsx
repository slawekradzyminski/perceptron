export function RegressionEducation() {
  return (
    <div className="education-panel regression-education">
      <h3>Neural Network Regression</h3>

      <div className="education-section">
        <h4>📈 From Classification to Continuous Prediction</h4>
        <p>
          While classification assigns discrete labels, <strong>regression</strong> predicts
          continuous values: house prices, temperature, stock returns. The same neural network
          architecture works — we just change the output layer and loss function.
        </p>
        <p>
          Linear regression (ŷ = mx + b) is the simplest case: one input, one output,
          no hidden layers. Adding layers allows learning non-linear relationships.
        </p>
      </div>

      <div className="education-section">
        <h4>🎯 Loss Functions for Regression</h4>
        <p><strong>Mean Squared Error (MSE)</strong> is the most common:</p>
        <p className="formula">
          MSE = (1/n) × Σ(prediction − target)²
        </p>
        <p>
          Squaring penalizes large errors heavily. A prediction off by 10 contributes 100
          to the loss, while off by 1 contributes only 1.
        </p>
        <p><strong>Mean Absolute Error (L1)</strong> is more robust to outliers:</p>
        <p className="formula">
          MAE = (1/n) × Σ|prediction − target|
        </p>
      </div>

      <div className="education-section">
        <h4>🔧 Output Layer Design</h4>
        <p>For regression, the output layer typically uses:</p>
        <ul>
          <li><strong>Linear (identity):</strong> No activation — allows any output value</li>
          <li><strong>ReLU:</strong> For non-negative outputs (prices, counts)</li>
          <li><strong>Sigmoid:</strong> For bounded outputs between 0 and 1</li>
        </ul>
      </div>

      <div className="education-section">
        <h4>🌊 Universal Function Approximation</h4>
        <p>
          A neural network with one hidden layer and enough neurons can approximate any
          continuous function to arbitrary precision (Cybenko, 1989). This is the
          <strong>Universal Approximation Theorem</strong>.
        </p>
        <p>
          Watch the visualization: as training progresses, the network's curve bends to
          fit the data points. More hidden neurons = more flexibility = tighter fit.
        </p>
      </div>

      <div className="education-section">
        <h4>⚠️ Overfitting</h4>
        <p>
          A network with too many parameters can <em>memorize</em> training points rather
          than learning the underlying pattern. Signs of overfitting:
        </p>
        <ul>
          <li>Training loss much lower than validation loss</li>
          <li>Predictions are perfect on training data but wild on new data</li>
          <li>The learned curve wiggles excessively between points</li>
        </ul>
        <p>
          Solutions include: regularization (L2), dropout, early stopping, or simply
          using fewer parameters.
        </p>
      </div>
    </div>
  );
}

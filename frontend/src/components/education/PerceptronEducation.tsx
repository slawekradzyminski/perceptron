export function PerceptronEducation() {
  return (
    <div className="education-panel perceptron-education">
      <h3>How the Perceptron Works</h3>

      <div className="education-section">
        <h4>🧠 The Birth of Neural Networks (1958)</h4>
        <p>
          The <strong>perceptron</strong> was invented by Frank Rosenblatt at Cornell in 1958,
          inspired by the biological neuron. It was the first algorithm that could <em>learn from data</em>,
          adjusting its weights based on mistakes. The New York Times famously predicted it would lead to
          machines that "walk, talk, see, write, reproduce itself and be conscious of its existence."
        </p>
        <p>
          While those predictions were premature, the perceptron introduced core ideas still used today:
          weighted sums, thresholds, and iterative learning.
        </p>
      </div>

      <div className="education-section">
        <h4>⚙️ The Mathematics</h4>
        <p>The perceptron computes:</p>
        <p className="formula">
          score = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
        </p>
        <p className="formula">
          prediction = sign(score)
        </p>
        <p>
          If the score is positive, predict +1. If negative, predict -1. The weights (w) determine
          how much each input (x) contributes. The bias (b) shifts the decision threshold.
        </p>
      </div>

      <div className="education-section">
        <h4>📐 Geometric Interpretation</h4>
        <p>
          The weights define a <strong>decision boundary</strong> — a hyperplane that divides
          the input space. In 2D, this is a line: w₁x₁ + w₂x₂ + b = 0. Points on one side
          are classified as +1, the other side as -1.
        </p>
        <p>
          Learning moves this boundary until it correctly separates the classes (if possible).
        </p>
      </div>

      <div className="education-section">
        <h4>🔄 The Perceptron Learning Rule</h4>
        <p>When the perceptron makes a mistake:</p>
        <p className="formula">
          w ← w + η · y · x
        </p>
        <p className="formula">
          b ← b + η · y
        </p>
        <p>
          where η is the learning rate and y is the correct label. This nudges the boundary
          toward classifying that point correctly. The <strong>Perceptron Convergence Theorem</strong>
          guarantees this converges in finite steps if the data is linearly separable.
        </p>
      </div>

      <div className="education-section">
        <h4>⚠️ The XOR Problem and AI Winter</h4>
        <p>
          In 1969, Minsky and Papert proved that perceptrons cannot learn XOR — a simple function
          where the output is true when inputs differ. This requires a non-linear boundary.
        </p>
        <p>
          This limitation, combined with overhyped expectations, triggered the first "AI winter."
          Research funding dried up until multi-layer networks (MLPs) and backpropagation
          solved the problem in the 1980s.
        </p>
      </div>
    </div>
  );
}

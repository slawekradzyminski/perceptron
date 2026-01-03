export function MlpMathCard() {
  return (
    <div className="mlp-math">
      <h3>MLP (2-layer) training</h3>
      <p><strong>Hidden:</strong> h = tanh(W₁x + b₁)</p>
      <p><strong>Output:</strong> z = W₂h + b₂</p>
      <p><strong>Probability:</strong> p̂ = σ(z)</p>
      <p><strong>Loss:</strong> L = −(y log p̂ + (1−y) log(1−p̂))</p>
      <p><strong>Update:</strong> W ← W − η ∇W L</p>
    </div>
  );
}

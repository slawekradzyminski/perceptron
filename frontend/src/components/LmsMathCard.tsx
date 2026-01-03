export function LmsMathCard() {
  return (
    <div className="lms-math">
      <h3>Update rule</h3>
      <p><strong>Prediction:</strong> ŷ = w·x + b</p>
      <p><strong>Error:</strong> e = y − ŷ</p>
      <p><strong>Update:</strong> w ← w + η e x, b ← b + η e</p>
      <p><strong>Gradient:</strong> ∂E/∂wᵢ = −2 e xᵢ, ∂E/∂b = −2 e</p>
    </div>
  );
}

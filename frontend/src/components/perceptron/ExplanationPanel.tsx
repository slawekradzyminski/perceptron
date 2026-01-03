export function ExplanationPanel() {
  return (
    <section className="panel explain">
      <h2>What happens on each step</h2>
      <div className="math">
        <p><strong>Score:</strong> <span>s = w · x + b</span></p>
        <p><strong>Prediction:</strong> <span>ŷ = sign(s)</span></p>
        <p><strong>Mistake?</strong> <span>y · s ≤ 0</span></p>
        <p><strong>Update (only if mistake):</strong> <span>w ← w + η y x, b ← b + η y</span></p>
      </div>
      <p className="detail">
        The input grid shows the switches (x). The weight grid shows the dials (w). The contribution grid
        shows each cell’s product xᵢ·wᵢ before the sum. The score meter is the sum of all contributions plus bias.
      </p>
      <p className="detail">
        For XOR, no single line separates the classes, so the boundary keeps moving without converging.
      </p>
    </section>
  );
}

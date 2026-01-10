export function ExplanationPanel() {
  return (
    <section className="panel explain">
      <p className="detail">
        The input grid shows the switches (x). The weight grid shows the dials (w). The contribution grid
        shows each cell's product xᵢ·wᵢ before the sum. The score meter is the sum of all contributions plus bias.
      </p>
      <p className="detail">
        For XOR, no single line separates the classes, so the boundary keeps moving without converging.
      </p>
    </section>
  );
}

export function DeepEducation() {
  return (
    <div className="education-panel deep-education">
      <h3>Deep Neural Networks: The Power of Depth</h3>

      <div className="education-section">
        <h4>📜 The Deep Learning Revolution (2012)</h4>
        <p>
          While multi-layer networks existed for decades, the "deep learning revolution" began
          in 2012 when AlexNet (8 layers) crushed the ImageNet competition. The key enablers:
          massive data (ImageNet), GPU computing, and better training techniques (ReLU, dropout).
        </p>
        <p>
          Today's largest models have hundreds of layers and billions of parameters.
        </p>
      </div>

      <div className="education-section">
        <h4>🌀 Why Depth Matters</h4>
        <p>
          Each layer learns increasingly abstract features. In image recognition:
        </p>
        <ul>
          <li><strong>Layer 1:</strong> Edges and colors</li>
          <li><strong>Layer 2-3:</strong> Textures and patterns</li>
          <li><strong>Layer 4-5:</strong> Parts (eyes, wheels, windows)</li>
          <li><strong>Final layers:</strong> Objects and concepts</li>
        </ul>
        <p>
          This hierarchical feature learning is why deep networks excel — they automatically
          discover the right representations.
        </p>
      </div>

      <div className="education-section">
        <h4>📐 Geometry of Learning</h4>
        <p>
          Each layer transforms the input space, "folding" it to make classification easier.
          Watch the boundary visualization: early layers create simple bends, while later
          layers combine them into complex decision surfaces.
        </p>
        <p>
          Mathematically, a 2-layer network can represent XOR. But representing a spiral
          requires more layers — depth provides compositional power.
        </p>
      </div>

      <div className="education-section">
        <h4>🎚️ Learning Rate for Deep Networks</h4>
        <p>
          Deeper networks require <strong>smaller learning rates</strong>. Why?
        </p>
        <ul>
          <li>Gradients must flow through many layers (chain rule multiplications)</li>
          <li>Large updates in early layers cause cascading changes</li>
          <li>The loss landscape is more complex with more parameters</li>
        </ul>
        <p>
          Typical values: η = 0.1 for shallow networks, η = 0.001 for deep networks,
          η = 0.0001 for fine-tuning.
        </p>
      </div>

      <div className="education-section">
        <h4>⚠️ Challenges of Depth</h4>
        <p><strong>Vanishing gradients:</strong> Gradients shrink as they backpropagate, making early layers learn slowly.</p>
        <p><strong>Solutions:</strong></p>
        <ul>
          <li><strong>ReLU activation:</strong> Gradient is 1 for positive inputs (no shrinking)</li>
          <li><strong>Batch normalization:</strong> Stabilizes layer outputs</li>
          <li><strong>Residual connections:</strong> Skip connections that let gradients flow directly</li>
          <li><strong>Careful initialization:</strong> Xavier or He initialization</li>
        </ul>
      </div>

      <div className="education-section">
        <h4>🎯 The Universal Approximation Perspective</h4>
        <p>
          A single wide layer can approximate any function (Cybenko, 1989). But deeper networks
          are exponentially more parameter-efficient — they share computations hierarchically.
        </p>
        <p>
          This is why we prefer depth over width: a 10-layer network with 100 neurons per layer
          (10K parameters) often outperforms a 1-layer network with 10K neurons.
        </p>
      </div>
    </div>
  );
}

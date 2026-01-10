export function ConvEducation() {
  return (
    <div className="conv-education">
      <h3>How Convolutional Neural Networks Work</h3>

      <div className="education-section">
        <h4>🔍 What is a Convolution?</h4>
        <p>
          A <strong>convolution</strong> is a mathematical operation that slides a small matrix
          (called a <em>kernel</em> or <em>filter</em>) across an image. At each position,
          it computes a weighted sum of the overlapping pixels. This creates an
          <em>activation map</em> that shows where the kernel's pattern appears in the image.
        </p>
      </div>

      <div className="education-section">
        <h4>🎯 Weight Sharing: The Key Insight</h4>
        <p>
          Unlike traditional neural networks where every connection has its own weight,
          CNNs use the <strong>same weights everywhere</strong>. This means:
        </p>
        <ul>
          <li><strong>Fewer parameters:</strong> A 3×3 kernel only has 9 weights, no matter how big the image is</li>
          <li><strong>Translation invariance:</strong> The network can detect a feature (like an edge) regardless of where it appears</li>
          <li><strong>Efficient learning:</strong> The network learns general patterns, not position-specific memorization</li>
        </ul>
      </div>

      <div className="education-section">
        <h4>📐 Padding and Stride</h4>
        <p>
          <strong>Padding</strong> adds zeros around the image border, allowing the kernel to
          process edge pixels and keep the output size equal to the input.
        </p>
        <p>
          <strong>Stride</strong> controls how far the kernel moves between positions.
          Stride 2 means the kernel skips every other pixel, reducing the output size by half.
        </p>
      </div>

      <div className="education-section">
        <h4>🏗️ Building Feature Hierarchies</h4>
        <p>
          Real CNNs stack multiple convolutional layers. Each layer builds on the previous:
        </p>
        <ul>
          <li><strong>Layer 1:</strong> Detects simple features like edges and gradients</li>
          <li><strong>Layer 2:</strong> Combines edges into corners, textures, and curves</li>
          <li><strong>Layer 3+:</strong> Recognizes complex patterns like eyes, wheels, or letters</li>
        </ul>
        <p>
          By the final layers, the network has transformed raw pixels into abstract concepts
          that can distinguish between a cat and a dog, or recognize handwritten digits.
        </p>
      </div>

      <div className="education-section">
        <h4>🧮 The Math Behind Each Cell</h4>
        <p>
          For each position in the output activation map, the calculation is:
        </p>
        <ol>
          <li><strong>Extract patch:</strong> Take the local region (receptive field) from the input</li>
          <li><strong>Multiply:</strong> Compute element-wise product with kernel weights</li>
          <li><strong>Sum:</strong> Add all products together to get the activation value</li>
        </ol>
        <p>
          Hover over the activation map above to see this calculation step-by-step!
        </p>
      </div>
    </div>
  );
}

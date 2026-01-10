export function AlexNetEducation() {
  return (
    <div className="education-panel alexnet-education">
      <h3>AlexNet: The Deep Learning Breakthrough</h3>

      <div className="education-section">
        <h4>📜 The ImageNet Moment (2012)</h4>
        <p>
          In September 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton entered
          AlexNet into the ImageNet Large Scale Visual Recognition Challenge. It achieved
          a top-5 error rate of <strong>15.3%</strong> — the runner-up had 26.2%.
        </p>
        <p>
          This wasn't incremental progress; it was a paradigm shift. Within a year, nearly
          every computer vision paper used deep learning. Hinton's student Ilya Sutskever
          would later co-found OpenAI.
        </p>
      </div>

      <div className="education-section">
        <h4>🏗️ The Architecture</h4>
        <p>AlexNet has 8 learnable layers:</p>
        <ul>
          <li><strong>Conv1:</strong> 96 filters, 11×11, stride 4 → detects edges</li>
          <li><strong>Conv2:</strong> 256 filters, 5×5 → textures and patterns</li>
          <li><strong>Conv3-5:</strong> 384, 384, 256 filters, 3×3 → complex features</li>
          <li><strong>FC6-7:</strong> 4096 neurons each → high-level reasoning</li>
          <li><strong>FC8:</strong> 1000 outputs (ImageNet classes)</li>
        </ul>
        <p>Total: ~60 million parameters, trained on 1.2 million images.</p>
      </div>

      <div className="education-section">
        <h4>💡 Key Innovations</h4>
        <ul>
          <li>
            <strong>ReLU Activation:</strong> f(x) = max(0, x). Six times faster to train
            than tanh. Avoids vanishing gradients.
          </li>
          <li>
            <strong>Dropout (p=0.5):</strong> Randomly zeros half of neurons during training,
            forcing the network to learn redundant representations.
          </li>
          <li>
            <strong>GPU Training:</strong> Split the network across two GTX 580 GPUs.
            Training took 5-6 days — unfeasible on CPUs.
          </li>
          <li>
            <strong>Data Augmentation:</strong> Random crops, flips, color jittering
            artificially expanded the training set 2048×.
          </li>
          <li>
            <strong>Local Response Normalization:</strong> Lateral inhibition inspired
            by neuroscience (later replaced by batch normalization).
          </li>
        </ul>
      </div>

      <div className="education-section">
        <h4>🔍 What Each Layer Sees</h4>
        <p>
          <strong>Layer 1 filters</strong> are interpretable: Gabor-like edge detectors
          at various orientations, plus color-opponent blobs. These emerge automatically
          from the data — no hand-engineering required.
        </p>
        <p>
          <strong>Deeper layers</strong> respond to increasingly abstract concepts:
          textures → parts → objects. By Layer 5, neurons respond to faces, text, or specific object parts.
        </p>
      </div>

      <div className="education-section">
        <h4>🎓 Legacy and Transfer Learning</h4>
        <p>
          AlexNet's features transfer remarkably well. The penultimate layer (FC7)
          provides a 4096-dimensional "universal image descriptor" useful for many tasks.
        </p>
        <p>
          This insight led to modern practice: pre-train on ImageNet, fine-tune on your task.
          Even today's models (ResNet, ViT) build on AlexNet's foundation.
        </p>
      </div>
    </div>
  );
}

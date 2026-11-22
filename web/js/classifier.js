// MLP minimo: capas densas con ReLU y softmax final.
// Los pesos vienen de ai_engine/training/train_static_classifier.py

export class StaticClassifier {
  constructor(model) {
    this.labels = model.labels;
    this.layers = model.layers; // cada capa: { W: [salida][entrada], b: [salida] }
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`No se pudo cargar el modelo: ${url}`);
    return new StaticClassifier(await response.json());
  }

  predict(features) {
    let activation = Array.from(features);

    this.layers.forEach((layer, index) => {
      const next = layer.W.map((row, i) => {
        let sum = layer.b[i];
        for (let j = 0; j < row.length; j++) sum += row[j] * activation[j];
        return sum;
      });
      const isLast = index === this.layers.length - 1;
      activation = isLast ? next : next.map((v) => Math.max(0, v));
    });

    // softmax estable
    const max = Math.max(...activation);
    const exps = activation.map((v) => Math.exp(v - max));
    const total = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / total);

    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: this.labels[best], confidence: probs[best], probs };
  }
}

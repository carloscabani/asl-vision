// Convierte predicciones frame a frame (ruidosas) en letras confirmadas.
// Una letra se acepta cuando domina la ventana reciente y se mantuvo
// el tiempo suficiente. Para repetir la misma letra hay que bajar la mano
// (o cambiar de letra) y volver a hacer la sena.

export class LetterStabilizer {
  constructor({ windowSize = 12, minConfidence = 0.8, holdMs = 700 } = {}) {
    this.windowSize = windowSize;
    this.minConfidence = minConfidence;
    this.holdMs = holdMs;
    this.reset();
  }

  reset() {
    this.window = [];
    this.candidate = null;
    this.candidateSince = 0;
    this.locked = null; // letra ya emitida mientras la mano sigue igual
  }

  // Llamar cuando no hay mano: permite volver a emitir la misma letra.
  handLost() {
    this.window = [];
    this.candidate = null;
    this.locked = null;
  }

  // Devuelve { letter, progress, emitted }
  update(prediction, now) {
    this.window.push(prediction.confidence >= this.minConfidence ? prediction.label : null);
    if (this.window.length > this.windowSize) this.window.shift();

    const counts = new Map();
    for (const l of this.window) if (l) counts.set(l, (counts.get(l) || 0) + 1);

    let top = null;
    let topCount = 0;
    for (const [l, c] of counts) if (c > topCount) { top = l; topCount = c; }

    const dominant = this.window.length >= this.windowSize / 2 && topCount >= this.windowSize * 0.7;
    const letter = dominant ? top : null;

    if (letter !== this.candidate) {
      this.candidate = letter;
      this.candidateSince = now;
      if (letter !== this.locked) this.locked = null;
    }

    if (!letter) return { letter: null, progress: 0, emitted: null };

    const progress = Math.min(1, (now - this.candidateSince) / this.holdMs);
    if (progress >= 1 && this.locked !== letter) {
      this.locked = letter;
      return { letter, progress: 1, emitted: letter };
    }
    return { letter, progress: this.locked === letter ? 1 : progress, emitted: null };
  }
}

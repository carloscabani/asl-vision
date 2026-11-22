// Replica EXACTA de LandmarkExtractor (Python): trasladar respecto a la muneca
// y dividir entre la distancia 3D maxima. Si cambias uno, cambia el otro.

export function extractFeatures(landmarks) {
  const wrist = landmarks[0];
  const pts = landmarks.map((p) => [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]);

  let scale = 0;
  for (const [x, y, z] of pts) {
    scale = Math.max(scale, Math.hypot(x, y, z));
  }
  if (scale > 0) {
    for (const p of pts) {
      p[0] /= scale;
      p[1] /= scale;
      p[2] /= scale;
    }
  }
  return Float32Array.from(pts.flat());
}

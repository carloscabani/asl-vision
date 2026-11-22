import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
import { extractFeatures } from "./features.js";
import { StaticClassifier } from "./classifier.js";
import { LetterStabilizer } from "./stabilizer.js";
import { OVERRIDES } from "./translation-overrides.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL_URL = "../models/hand_landmarker.task";
const CLASSIFIER_URL = "./models/static_mlp.json";

// Pares de landmarks para dibujar el esqueleto de la mano
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
];

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  status: document.getElementById("status"),
  letter: document.getElementById("letter"),
  confidence: document.getElementById("confidence"),
  progress: document.getElementById("progress"),
  text: document.getElementById("text"),
  lang: document.getElementById("lang"),
  source: document.getElementById("source"),
  translated: document.getElementById("translated"),
  note: document.getElementById("note"),
  speak: document.getElementById("speak"),
  space: document.getElementById("space"),
  back: document.getElementById("back"),
  clear: document.getElementById("clear"),
};
const ctx = els.canvas.getContext("2d");

let text = "";
function setText(value) {
  text = value;
  els.text.textContent = text || "…";
}

// --- Traduccion en el dispositivo (Translator API de Chrome/Edge de escritorio) ---
const translators = new Map();

async function getTranslator(source, target) {
  if (!("Translator" in self)) return null;
  const key = `${source}>${target}`;
  if (translators.has(key)) return translators.get(key);

  const options = { sourceLanguage: source, targetLanguage: target };
  if ((await Translator.availability(options)) === "unavailable") return null;

  // create() descarga el paquete de idioma la primera vez; debe ocurrir
  // dentro de un clic (el boton "Hablar" lo es).
  const translator = await Translator.create({
    ...options,
    monitor(m) {
      m.addEventListener("downloadprogress", (e) => {
        els.note.textContent = `Descargando idioma… ${Math.round(e.loaded * 100)}%`;
      });
    },
  });
  translators.set(key, translator);
  return translator;
}

async function translate(raw, source, target) {
  const override = OVERRIDES[`${source}>${target}`]?.[raw];
  if (override) return override;
  const translator = await getTranslator(source, target);
  return translator ? translator.translate(raw) : null;
}

async function speak() {
  // En minusculas: el deletreo sale en mayusculas y un traductor
  // tomaria "HOLA" como sigla en vez de como palabra.
  const raw = text.trim().toLowerCase();
  if (!raw) return;

  const source = els.source.value;            // "es" | "en"
  const voiceLang = els.lang.value;           // "es-ES" | "en-US"
  const target = voiceLang.slice(0, 2);

  let toSpeak = raw;
  let spokenLang = voiceLang;
  els.translated.textContent = "";
  els.note.textContent = "";

  if (source !== target) {
    try {
      const result = await translate(raw, source, target);
      if (result) {
        toSpeak = result;
        els.translated.textContent = `→ ${result}`;
      } else {
        throw new Error("unsupported");
      }
    } catch (error) {
      console.warn("Traduccion no disponible", error);
      els.note.textContent =
        "Traducción no disponible (requiere Chrome o Edge de escritorio). Se lee sin traducir.";
      spokenLang = source === "es" ? "es-ES" : "en-US";
    } 
  }

  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(toSpeak);
  utterance.lang = spokenLang;
  speechSynthesis.speak(utterance);
}

els.speak.onclick = speak;
els.space.onclick = () => setText(text + " ");
els.back.onclick = () => setText(text.slice(0, -1));
els.clear.onclick = () => setText("");

function draw(landmarks) {
  const { width, height } = els.canvas;
  ctx.strokeStyle = "#00e676";
  ctx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function main() {
  els.status.textContent = "Cargando modelos…";
  const [vision, classifier] = await Promise.all([
    FilesetResolver.forVisionTasks(WASM_URL),
    StaticClassifier.load(CLASSIFIER_URL),
  ]);

  const createLandmarker = (delegate) =>
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

  let landmarker;
  try {
    landmarker = await createLandmarker("GPU");
  } catch (error) {
    console.warn("GPU no disponible, usando CPU", error);
    landmarker = await createLandmarker("CPU");
  }

  els.status.textContent = "Pidiendo cámara…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
  });
  els.video.srcObject = stream;
  await els.video.play();
  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;
  els.status.textContent = "Listo. Muestra una mano.";

  const stabilizer = new LetterStabilizer();
  let lastVideoTime = -1;

  function loop() {
    if (els.video.currentTime !== lastVideoTime) {
      lastVideoTime = els.video.currentTime;

      // Se dibuja el video ESPEJADO en el canvas y se detecta sobre ese canvas,
      // igual que en la recoleccion de datos (cv2.flip(frame, 1)).
      ctx.save();
      ctx.translate(els.canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
      ctx.restore();

      const now = performance.now();
      const result = landmarker.detectForVideo(els.canvas, now);

      if (result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        draw(landmarks);

        const prediction = classifier.predict(extractFeatures(landmarks));
        const state = stabilizer.update(prediction, now);

        els.letter.textContent = prediction.label;
        els.confidence.textContent = `${(prediction.confidence * 100).toFixed(0)}%`;
        els.progress.value = state.progress;
        if (state.emitted) setText(text + state.emitted);
      } else {
        stabilizer.handLost();
        els.letter.textContent = "–";
        els.confidence.textContent = "";
        els.progress.value = 0;
      }
    }
    requestAnimationFrame(loop);
  }
  loop();
}

main().catch((error) => {
  console.error(error);
  els.status.textContent = `Error: ${error.message}`;
});

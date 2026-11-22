"""
Entrena un clasificador pequeno (MLP) para las 24 letras estaticas de ASL
a partir de data/asl_landmarks.csv y lo exporta a JSON para el navegador.

Uso (desde la raiz del repo, con el venv activo):
    pip install scikit-learn
    python ai_engine/training/train_static_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "asl_landmarks.csv"
OUTPUT_PATH = PROJECT_ROOT / "web" / "models" / "static_mlp.json"

HOLDOUT_FRACTION = 0.2   # ultimo 20% (en orden de grabacion) de cada letra
AUGMENT_COPIES = 10
ROLL_DEG = 15      # giro en el plano de la imagen
PITCH_DEG = 25     # inclinar la mano hacia adelante / atras
YAW_DEG = 25       # girar la mano hacia los lados
SCALE_JITTER = 0.08  # proporciones de mano distintas (dedos mas largos/cortos)
NOISE_STD = 0.015
SEED = 42


def normalize(coords):
    """Misma normalizacion que LandmarkExtractor / features.js:
    muneca al origen y distancia 3D maxima = 1."""
    coords = coords - coords[:, 0:1, :]
    scale = np.linalg.norm(coords, axis=2).max(axis=1)
    scale[scale == 0] = 1
    return coords / scale[:, None, None]


def rotation_matrices(rng, n, roll, pitch, yaw):
    """n matrices 3D aleatorias (angulos en grados, uniformes en [-max, max])."""
    r, p, w = (np.deg2rad(rng.uniform(-m, m, n)) for m in (roll, pitch, yaw))
    cr, sr, cp, sp, cw, sw = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(w), np.sin(w)
    Rz = np.zeros((n, 3, 3)); Rz[:, 0, 0] = cr; Rz[:, 0, 1] = -sr; Rz[:, 1, 0] = sr; Rz[:, 1, 1] = cr; Rz[:, 2, 2] = 1
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cp; Rx[:, 1, 2] = -sp; Rx[:, 2, 1] = sp; Rx[:, 2, 2] = cp
    Ry = np.zeros((n, 3, 3)); Ry[:, 0, 0] = cw; Ry[:, 0, 2] = sw; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -sw; Ry[:, 2, 2] = cw
    return Rz @ Rx @ Ry


def perturb(X, rng, roll, pitch, yaw, scale_jitter, noise_std):
    coords = X.reshape(-1, 21, 3)
    n = len(coords)
    R = rotation_matrices(rng, n, roll, pitch, yaw)
    out = np.einsum("nij,nkj->nki", R, coords)             # rota cada punto en 3D
    out = out * rng.uniform(1 - scale_jitter, 1 + scale_jitter, (n, 1, 3))
    out = out + rng.normal(0, noise_std, out.shape)
    return normalize(out).reshape(n, 63)


def augment(X, y, rng):
    """Rotaciones 3D (inclinar/girar la mano), cambio de proporciones y ruido.
    Tus muestras tienen muy poca variacion (se grabaron casi en la misma pose);
    esto le ensena al modelo que la misma sena se ve distinta segun el angulo."""
    Xs, ys = [X], [y]
    for _ in range(AUGMENT_COPIES):
        Xs.append(perturb(X, rng, ROLL_DEG, PITCH_DEG, YAW_DEG, SCALE_JITTER, NOISE_STD))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def split_by_recording_order(df):
    """Las muestras de una letra se grabaron seguidas: un split aleatorio
    mezclaria frames casi identicos entre train y test e inflaria la precision.
    Por eso el test es el final de cada letra."""
    train_parts, test_parts = [], []
    for _, group in df.groupby("label", sort=False):
        cut = int(len(group) * (1 - HOLDOUT_FRACTION))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def make_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        max_iter=600,
        random_state=SEED,
    )


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATASET_PATH)
    feature_cols = [f"feature_{i}" for i in range(63)]

    encoder = LabelEncoder().fit(df["label"])
    train_df, test_df = split_by_recording_order(df)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = encoder.transform(train_df["label"])
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = encoder.transform(test_df["label"])

    # Test "poses nuevas": el holdout visto desde angulos que no se grabaron.
    X_hard = perturb(X_test, np.random.default_rng(SEED + 1), 10, 20, 20, 0.05, 0.01)

    base = make_model().fit(X_train, y_train)
    X_aug, y_aug = augment(X_train, y_train, rng)
    aug = make_model().fit(X_aug, y_aug)

    print("                 holdout   poses nuevas (+-20 grados)")
    print(f"Sin aumento      {base.score(X_test, y_test):.3f}     {base.score(X_hard, y_test):.3f}")
    print(f"Con aumento 3D   {aug.score(X_test, y_test):.3f}     {aug.score(X_hard, y_test):.3f}")

    hard_pred = aug.predict(X_hard)
    for letter in ("G", "H", "Y"):
        mask = y_test == encoder.transform([letter])[0]
        print(f"  {letter}: acierto en poses nuevas = {(hard_pred[mask] == y_test[mask]).mean():.2f}")

    # Modelo final: todos los datos + aumento
    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    y_all = encoder.transform(df["label"])
    X_all, y_all = augment(X_all, y_all, rng)
    final = make_model().fit(X_all, y_all)

    export = {
        "labels": encoder.classes_.tolist(),
        "feature_count": 63,
        "activation": "relu",
        "layers": [
            {"W": w.T.round(6).tolist(), "b": b.round(6).tolist()}  # W: [salida][entrada]
            for w, b in zip(final.coefs_, final.intercepts_)
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(export), encoding="utf-8")
    print(f"Modelo exportado a {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

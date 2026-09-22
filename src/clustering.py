"""K-means, métricas de agrupamiento y utilidades de comparación contra referencias."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    labels: np.ndarray
    centroids_scaled: np.ndarray
    centroids_original: np.ndarray
    inertia: float
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    dunn_index: float | None
    pca_coords: np.ndarray
    pca_explained_variance: np.ndarray


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Índice de Dunn: min distancia intercluster / max distancia intracluster.

    Igual al usado en el estudio MinTIC 2023 para elegir el número óptimo de
    clusters (valor alto = agrupación más compacta y separada).
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return None

    intra = []
    for c in unique:
        pts = X[labels == c]
        if len(pts) < 2:
            intra.append(0.0)
            continue
        d = cdist(pts, pts)
        intra.append(d.max())
    max_intra = max(intra) if max(intra) > 0 else 1e-9

    inter = []
    for i, ci in enumerate(unique):
        for cj in unique[i + 1 :]:
            d = cdist(X[labels == ci], X[labels == cj])
            inter.append(d.min())
    min_inter = min(inter)

    return float(min_inter / max_intra)


def run_kmeans(
    df: pd.DataFrame,
    features: list[str],
    k: int,
    scale: bool,
    random_state: int,
    n_init: int,
) -> ClusteringResult:
    X_raw = df[features].to_numpy(dtype=float)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X = X_raw

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled) if scaler else centroids_scaled

    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)

    sil = silhouette_score(X, labels) if k > 1 and k < len(X) else None
    ch = calinski_harabasz_score(X, labels) if k > 1 else None
    db = davies_bouldin_score(X, labels) if k > 1 else None
    dunn = dunn_index(X, labels) if k > 1 else None

    return ClusteringResult(
        labels=labels,
        centroids_scaled=centroids_scaled,
        centroids_original=centroids_original,
        inertia=float(model.inertia_),
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        dunn_index=dunn,
        pca_coords=coords,
        pca_explained_variance=pca.explained_variance_ratio_,
    )


def elbow_curve(
    df: pd.DataFrame, features: list[str], scale: bool, k_range: range, random_state: int, n_init: int
) -> pd.DataFrame:
    X_raw = df[features].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X_raw) if scale else X_raw

    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels) if k > 1 else np.nan,
                "dunn_index": dunn_index(X, labels) if k > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class StepFrame:
    """Un fotograma de la animación paso a paso del algoritmo k-medias."""

    step_type: str  # "init" | "assign" | "update" | "converged" | "max_iter"
    iteration: int
    labels: np.ndarray | None
    centroids: np.ndarray
    centroids_2d: np.ndarray
    inertia: float | None
    n_changed: int | None


def kmeans_animation(
    df: pd.DataFrame,
    features: list[str],
    k: int,
    scale: bool,
    random_state: int = 42,
    max_iter: int = 30,
) -> tuple[np.ndarray, list[StepFrame]]:
    """Ejecuta k-medias manualmente, guardando cada paso (inicialización,
    asignación, actualización) para poder reproducirlos uno a uno en la UI.

    Devuelve la proyección PCA 2D de los puntos (fija) y la lista de
    fotogramas; cada fotograma trae los centroides ya proyectados a 2D con
    el mismo PCA, para poder dibujarlos sobre los mismos ejes.
    """
    X_raw = df[features].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X_raw) if scale else X_raw

    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    points_2d = pca.fit_transform(X)

    rng = np.random.RandomState(random_state)
    init_idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[init_idx].copy()

    frames = [StepFrame("init", 0, None, centroids.copy(), pca.transform(centroids), None, None)]

    prev_labels = None
    inertia = None
    n_changed = None
    for it in range(1, max_iter + 1):
        d = cdist(X, centroids)
        labels = d.argmin(axis=1)
        inertia = float((d[np.arange(len(X)), labels] ** 2).sum())
        n_changed = len(X) if prev_labels is None else int((labels != prev_labels).sum())
        frames.append(StepFrame("assign", it, labels.copy(), centroids.copy(), pca.transform(centroids), inertia, n_changed))

        if prev_labels is not None and n_changed == 0:
            frames.append(
                StepFrame("converged", it, labels.copy(), centroids.copy(), pca.transform(centroids), inertia, 0)
            )
            break

        new_centroids = centroids.copy()
        for c in range(k):
            pts = X[labels == c]
            if len(pts) > 0:
                new_centroids[c] = pts.mean(axis=0)
        frames.append(
            StepFrame("update", it, labels.copy(), new_centroids.copy(), pca.transform(new_centroids), inertia, n_changed)
        )
        centroids = new_centroids
        prev_labels = labels
    else:
        frames.append(
            StepFrame("max_iter", max_iter, prev_labels.copy(), centroids.copy(), pca.transform(centroids), inertia, n_changed)
        )

    return points_2d, frames


def align_clusters_to_labels(cluster_labels: np.ndarray, ref_labels: pd.Series) -> dict[int, str]:
    """Empareja cada cluster (entero) con la categoría de referencia más frecuente,
    usando el algoritmo húngaro para maximizar el acuerdo global (evita que dos
    clusters se asignen a la misma categoría cuando hay una alternativa mejor).
    """
    ref_categories = [c for c in pd.unique(ref_labels) if pd.notna(c)]
    clusters = sorted(pd.unique(cluster_labels))

    cost = np.zeros((len(clusters), len(ref_categories)))
    for i, c in enumerate(clusters):
        mask = cluster_labels == c
        counts = ref_labels[mask].value_counts()
        for j, cat in enumerate(ref_categories):
            cost[i, j] = -counts.get(cat, 0)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {clusters[r]: ref_categories[c] for r, c in zip(row_ind, col_ind)}

    # Clusters sin categoría asignada (más clusters que categorías): usar la moda igualmente.
    for c in clusters:
        if c not in mapping:
            mask = cluster_labels == c
            counts = ref_labels[mask].value_counts()
            mapping[c] = counts.idxmax() if len(counts) else "N/A"
    return mapping


@dataclass
class ComparisonResult:
    confusion: pd.DataFrame
    accuracy: float
    ari: float
    nmi: float
    per_class: pd.DataFrame
    mapping: dict[int, str]


def compare_to_reference(cluster_labels: np.ndarray, ref_labels: pd.Series) -> ComparisonResult:
    valid = ref_labels.notna().to_numpy()
    y_true = ref_labels[valid].astype(str).to_numpy()
    y_pred_raw = cluster_labels[valid]

    mapping = align_clusters_to_labels(y_pred_raw, ref_labels[valid])
    y_pred = np.array([mapping[c] for c in y_pred_raw])

    categories = sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    cm_df = pd.DataFrame(cm, index=[f"Real: {c}" for c in categories], columns=[f"Pred: {c}" for c in categories])

    accuracy = float((y_true == y_pred).mean())
    ari = float(adjusted_rand_score(y_true, y_pred_raw))
    nmi = float(normalized_mutual_info_score(y_true, y_pred_raw))

    per_class_rows = []
    for cat in categories:
        tp = int(((y_true == cat) & (y_pred == cat)).sum())
        fp = int(((y_true != cat) & (y_pred == cat)).sum())
        fn = int(((y_true == cat) & (y_pred != cat)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else np.nan
        per_class_rows.append(
            {"categoría": cat, "precisión": precision, "recall": recall, "f1": f1, "soporte": int((y_true == cat).sum())}
        )
    per_class = pd.DataFrame(per_class_rows)

    return ComparisonResult(confusion=cm_df, accuracy=accuracy, ari=ari, nmi=nmi, per_class=per_class, mapping=mapping)

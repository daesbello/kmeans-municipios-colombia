"""Esquemas de etiquetas de referencia para contrastar contra los clusters de k-means.

Ninguno de los dos esquemas es la "verdad absoluta": k-means es un método no
supervisado y el dataset no trae una etiqueta por municipio. Se ofrecen dos
referencias externas, razonables y documentadas, para que el usuario pueda
evaluar qué tan bien el agrupamiento recupera una clasificación conocida.
"""
from __future__ import annotations

import pandas as pd

# Rangos oficiales del DNP para el Índice de Desempeño Fiscal (IDF), escala 0-100.
IDF_BINS = [-float("inf"), 40, 60, 70, 80, float("inf")]
IDF_LABELS = ["Deterioro", "Riesgo", "Vulnerable", "Sostenible", "Solvente"]

# Proporciones del estudio MinTIC (2023) para la agrupación de municipios con k=3:
# cluster avanzado 124/1101, intermedio 416/1101, básico 561/1101.
MINTIC_PROPORTIONS = {
    "Avanzado": 124 / 1101,
    "Intermedio": 416 / 1101,
    "Básico": 561 / 1101,
}


def label_idf_dnp(df: pd.DataFrame) -> pd.Series:
    """Clasifica cada municipio según los rangos oficiales del DNP para el IDF."""
    return pd.cut(df["idf"], bins=IDF_BINS, labels=IDF_LABELS)


def label_mintic_replica(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Reconstruye una referencia tipo 'Avanzado/Intermedio/Básico'.

    Se calcula un score compuesto (promedio de z-scores de las variables de
    caracterización) y se corta por cuantiles para reproducir aproximadamente
    los tamaños de grupo reportados en el estudio MinTIC 2023 (124/416/561 de
    1101 municipios). Es una aproximación de referencia, no la clasificación
    original del estudio (esa no está publicada a nivel de municipio).
    """
    z = (df[features] - df[features].mean()) / df[features].std(ddof=0)
    score = z.mean(axis=1)

    q_basico = MINTIC_PROPORTIONS["Básico"]
    q_intermedio = MINTIC_PROPORTIONS["Intermedio"]
    cut_low = score.quantile(q_basico)
    cut_mid = score.quantile(q_basico + q_intermedio)

    labels = pd.Series("Intermedio", index=df.index)
    labels[score <= cut_low] = "Básico"
    labels[score > cut_mid] = "Avanzado"
    return pd.Categorical(labels, categories=["Básico", "Intermedio", "Avanzado"])


REFERENCE_SCHEMES = {
    "Categorías oficiales IDF (DNP)": label_idf_dnp,
    "Réplica estudio MinTIC (Avanzado/Intermedio/Básico)": label_mintic_replica,
}

"""Carga y limpieza del dataset de municipios de Colombia."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Consolidado_Municipios_Colombia_con_IDF_2022.xlsx"
SHEET_NAME = "Consolidado 2025-2026"

# Nombres cortos usados en toda la app -> nombre real de columna en el Excel
COLUMN_MAP = {
    "poblacion": "Número de habitantes",
    "desempeno_institucional": "Índice de Desempeño Institucional",
    "ingresos": "Ingresos Totales Municipales (millones $, 2025)",
    "idf": "Puntaje IDF (2022)",
    "densidad": "Número de habitantes por km² (densidad poblacional) proyecciòn 2026",
}

# Variables disponibles para caracterizar municipios (subconjunto de las 6 usadas
# en el estudio MinTIC 2023; el dataset no incluye "% hogares con acceso a internet").
FEATURE_LABELS = {
    "poblacion": "Número de habitantes",
    "desempeno_institucional": "Índice de Desempeño Institucional",
    "ingresos": "Ingresos Totales Municipales (millones $)",
    "idf": "Puntaje IDF (Índice de Desempeño Fiscal)",
    "densidad": "Densidad poblacional (hab/km²)",
}

DEFAULT_FEATURES = ["poblacion", "desempeno_institucional", "ingresos", "idf", "densidad"]


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})

    for col in ["desempeno_institucional", "ingresos", "idf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["municipio_id"] = df["Código DIVIPOLA"].astype(str) + " - " + df["Municipio"].astype(str)
    return df


def prepare_dataset(missing_strategy: str, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Devuelve el dataframe listo para clustering + metadatos sobre filas descartadas/imputadas.

    missing_strategy: "excluir" | "imputar_mediana"
    """
    df = load_raw().copy()
    n_total = len(df)

    missing_mask = df[features].isna().any(axis=1)
    n_missing = int(missing_mask.sum())

    if missing_strategy == "excluir":
        df = df.loc[~missing_mask].reset_index(drop=True)
    else:  # imputar_mediana
        for col in features:
            df[col] = df[col].fillna(df[col].median())

    meta = {
        "n_total": n_total,
        "n_missing": n_missing,
        "n_usado": len(df),
        "missing_strategy": missing_strategy,
    }
    return df, meta

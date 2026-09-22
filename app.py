"""Simulador de K-Means sobre municipios de Colombia (IDF 2022).

Réplica interactiva de la metodología de agrupación de municipios descrita en
"Plazos para la implementación del Marco de Referencia de Arquitectura
Empresarial" (MinTIC, 2023): k-medias sobre variables socioeconómicas, con
selección del número de clusters vía Índice de Dunn.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clustering import compare_to_reference, elbow_curve, kmeans_animation, run_kmeans
from src.data_processing import DEFAULT_FEATURES, FEATURE_LABELS, prepare_dataset
from src.reference_labels import REFERENCE_SCHEMES, label_idf_dnp, label_mintic_replica

CLUSTER_COLORS = px.colors.qualitative.Set1


def _animation_figure(points_2d, frame, municipios):
    fig = go.Figure()
    if frame.labels is None:
        fig.add_trace(
            go.Scatter(
                x=points_2d[:, 0],
                y=points_2d[:, 1],
                mode="markers",
                marker=dict(color="lightgray", size=6, opacity=0.6),
                text=municipios,
                hoverinfo="text",
                name="Municipios (sin asignar)",
            )
        )
    else:
        for c in range(frame.centroids.shape[0]):
            mask = frame.labels == c
            fig.add_trace(
                go.Scatter(
                    x=points_2d[mask, 0],
                    y=points_2d[mask, 1],
                    mode="markers",
                    marker=dict(size=6, opacity=0.65, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)]),
                    text=municipios[mask],
                    hoverinfo="text",
                    name=f"Cluster {c}",
                )
            )
    fig.add_trace(
        go.Scatter(
            x=frame.centroids_2d[:, 0],
            y=frame.centroids_2d[:, 1],
            mode="markers",
            marker=dict(symbol="star", size=22, color="black", line=dict(width=2, color="white")),
            name="Centroides",
            hoverinfo="skip",
        )
    )
    step_titles = {
        "init": "Paso 0 · Inicialización",
        "assign": f"Iteración {frame.iteration} · Paso 1: Asignación",
        "update": f"Iteración {frame.iteration} · Paso 2: Actualización",
        "converged": f"Convergencia (iteración {frame.iteration})",
        "max_iter": f"Máximo de iteraciones alcanzado ({frame.iteration})",
    }
    fig.update_layout(
        title=step_titles.get(frame.step_type, frame.step_type),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _animation_explanation(frame, k: int, features: list[str]) -> str:
    names = ", ".join(FEATURE_LABELS[f] for f in features)
    if frame.step_type == "init":
        return (
            f"**Paso 0 · Inicialización.** El algoritmo elige al azar {k} municipios para usarlos como "
            f"centroides iniciales (★). Todavía ningún municipio está asignado a un cluster."
        )
    if frame.step_type == "assign":
        return (
            f"**Iteración {frame.iteration} · Asignación.** Se calcula la distancia euclidiana de cada "
            f"municipio (en el espacio de {len(features)} variables estandarizadas: {names}) a cada uno "
            f"de los {k} centroides, y se asigna al más cercano. **{frame.n_changed} municipios** cambiaron "
            f"de cluster respecto al paso anterior. Inercia actual (suma de distancias² dentro de cada "
            f"cluster): **{frame.inertia:,.0f}**."
        )
    if frame.step_type == "update":
        return (
            "**Actualización.** Cada centroide (★) se recalcula como el promedio de los municipios que "
            "tiene asignados en este momento — por eso se desplaza. En el siguiente paso se vuelve a "
            "evaluar si, con los centroides ya movidos, algún municipio queda más cerca de uno distinto."
        )
    if frame.step_type == "converged":
        return (
            f"**✅ Convergencia en la iteración {frame.iteration}.** Ningún municipio cambió de cluster en "
            f"la última asignación: eso significa que si se recalcularan los centroides de nuevo, no se "
            f"moverían. El algoritmo se detiene aquí — este es el resultado final de k-means."
        )
    return f"Se alcanzó el máximo de iteraciones ({frame.iteration}) sin convergencia completa."

st.set_page_config(page_title="Simulador K-Means · Municipios de Colombia", page_icon="📍", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar: controles del simulador
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controles del simulador")

features = st.sidebar.multiselect(
    "Variables de caracterización",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda k: FEATURE_LABELS[k],
    help="Variables usadas para calcular la distancia entre municipios (basadas en el estudio MinTIC 2023).",
)

missing_strategy_label = st.sidebar.radio(
    "Municipios con datos faltantes (21 Áreas No Municipalizadas sin IDF)",
    options=["Excluir del análisis", "Imputar con la mediana"],
    index=0,
)
missing_strategy = "excluir" if missing_strategy_label.startswith("Excluir") else "imputar_mediana"

scale = st.sidebar.checkbox(
    "Estandarizar variables (StandardScaler)",
    value=True,
    help="Recomendado: las variables tienen escalas muy distintas (habitantes vs. puntajes 0-100).",
)

k = st.sidebar.slider("Número de clusters (k)", min_value=2, max_value=10, value=3)
st.sidebar.caption("El estudio MinTIC 2023 usó k=3, elegido maximizando el Índice de Dunn.")

with st.sidebar.expander("Parámetros avanzados"):
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)
    n_init = st.number_input("n_init (reinicios del algoritmo)", min_value=1, value=10, step=1)

reference_choice = st.sidebar.selectbox(
    "Referencia para matriz de confusión",
    options=["Ninguna"] + list(REFERENCE_SCHEMES.keys()),
    index=1,
)

if not features:
    st.error("Selecciona al menos una variable en la barra lateral para continuar.")
    st.stop()

st.sidebar.divider()
st.sidebar.caption("Desarrollado por David Bello")

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
df, meta = prepare_dataset(missing_strategy, features)

if reference_choice == "Categorías oficiales IDF (DNP)":
    ref_labels = label_idf_dnp(df)
elif reference_choice == "Réplica estudio MinTIC (Avanzado/Intermedio/Básico)":
    ref_labels = label_mintic_replica(df, features)
else:
    ref_labels = None

result = run_kmeans(df, features, k, scale, int(random_state), int(n_init))
df = df.copy()
df["cluster"] = result.labels.astype(str)
if ref_labels is not None:
    df["referencia"] = pd.Series(ref_labels, index=df.index)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("📍 Simulador K-Means — Municipios de Colombia")
st.caption(
    "Dataset: Consolidado de municipios con IDF 2022 (DNP) · "
    f"{meta['n_usado']} de {meta['n_total']} municipios usados "
    f"({meta['n_missing']} con dato faltante, estrategia: {missing_strategy_label.lower()})."
)

tab_sim, tab_class, tab_comp, tab_docs = st.tabs(
    ["🎛️ Simulador", "🎓 Aula interactiva", "📊 Matriz de confusión y métricas", "📚 Documentación"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Simulador
# ---------------------------------------------------------------------------
with tab_sim:
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Municipios", meta["n_usado"])
    kpi_cols[1].metric("k", k)
    kpi_cols[2].metric("Inercia", f"{result.inertia:,.0f}")
    kpi_cols[3].metric("Silhouette", f"{result.silhouette:.3f}" if result.silhouette is not None else "—")
    kpi_cols[4].metric("Davies-Bouldin", f"{result.davies_bouldin:.3f}" if result.davies_bouldin is not None else "—")
    kpi_cols[5].metric("Índice de Dunn", f"{result.dunn_index:.3f}" if result.dunn_index is not None else "—")

    st.markdown("### Número óptimo de clusters")
    st.caption("Curva del codo (inercia) e Índice de Dunn para distintos valores de k, igual que en el estudio MinTIC 2023.")

    @st.cache_data(show_spinner="Calculando curva del codo...")
    def _elbow(missing_strategy, features_tuple, scale, k_min, k_max, random_state, n_init):
        df_, _ = prepare_dataset(missing_strategy, list(features_tuple))
        return elbow_curve(df_, list(features_tuple), scale, range(k_min, k_max + 1), random_state, n_init)

    elbow_df = _elbow(missing_strategy, tuple(features), scale, 2, 10, int(random_state), int(n_init))

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.line(elbow_df, x="k", y="inertia", markers=True, title="Curva del codo")
        fig.add_vline(x=k, line_dash="dash", line_color="crimson")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.line(elbow_df, x="k", y="dunn_index", markers=True, title="Índice de Dunn por k")
        fig.add_vline(x=k, line_dash="dash", line_color="crimson")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Visualización de los clusters (proyección PCA 2D)")
    pca_df = pd.DataFrame(result.pca_coords, columns=["PC1", "PC2"][: result.pca_coords.shape[1]])
    pca_df["cluster"] = df["cluster"].to_numpy()
    pca_df["Municipio"] = df["Municipio"].to_numpy()

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_name="Municipio",
        opacity=0.7,
        title=f"Municipios proyectados en 2D (varianza explicada: {result.pca_explained_variance.sum() * 100:.1f}%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Perfil de los clusters")
    profile = df.groupby("cluster")[features].mean()
    profile["n_municipios"] = df.groupby("cluster").size()
    profile["% del total"] = (profile["n_municipios"] / len(df) * 100).round(1)
    profile = profile.rename(columns=FEATURE_LABELS)
    st.dataframe(profile.style.format(precision=1), use_container_width=True)

    size_fig = px.bar(
        df["cluster"].value_counts().sort_index().reset_index(),
        x="cluster",
        y="count",
        title="Tamaño de cada cluster",
        labels={"count": "N° municipios", "cluster": "Cluster"},
    )
    st.plotly_chart(size_fig, use_container_width=True)

    st.markdown("### Municipios y cluster asignado")
    show_cols = ["Municipio", "Código DIVIPOLA", "cluster"] + (["referencia"] if ref_labels is not None else []) + features
    display_df = df[show_cols].rename(columns=FEATURE_LABELS)
    search = st.text_input("Buscar municipio")
    if search:
        display_df = display_df[display_df["Municipio"].str.contains(search, case=False, na=False)]
    st.dataframe(display_df, use_container_width=True, height=350)
    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"kmeans_municipios_k{k}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# TAB 2 — Aula interactiva (k-means paso a paso)
# ---------------------------------------------------------------------------
with tab_class:
    st.markdown("### 🎓 Cómo piensa k-means, paso a paso")
    st.caption(
        "Aquí el algoritmo se ejecuta de forma manual (sin usar `KMeans.fit` de una sola vez) guardando "
        "cada paso intermedio, para poder recorrerlos uno por uno como en una clase en vivo."
    )
    st.latex(r"V_{C_k} = \sum_{i=1}^{m} (x_i^k - \mu_k)^2 \qquad \text{(inercia: variación total dentro del cluster } C_k\text{)}")
    st.caption(
        "⚠️ Esta animación usa **una sola** inicialización aleatoria de centroides (a diferencia del "
        "Simulador, que corre 10 inicializaciones y se queda con la mejor). Por eso puede converger a un "
        "resultado distinto — y a veces peor — que el de la pestaña Simulador: es precisamente el motivo "
        "por el que k-means en la práctica se corre varias veces. Cambia el `random_state` en Parámetros "
        "avanzados para comprobarlo con otra semilla."
    )

    @st.cache_data(show_spinner="Ejecutando k-means paso a paso...")
    def _kmeans_animation(missing_strategy, features_tuple, scale, k, random_state, max_iter=30):
        df_, _ = prepare_dataset(missing_strategy, list(features_tuple))
        return kmeans_animation(df_, list(features_tuple), k, scale, random_state, max_iter)

    points_2d, frames = _kmeans_animation(missing_strategy, tuple(features), scale, k, int(random_state))
    n_frames = len(frames)
    municipios = df["Municipio"].to_numpy()

    config_fingerprint = (missing_strategy, tuple(features), scale, k, int(random_state))
    if st.session_state.get("anim_config") != config_fingerprint:
        st.session_state.anim_config = config_fingerprint
        st.session_state.frame_idx = 0
    st.session_state.frame_idx = min(st.session_state.get("frame_idx", 0), n_frames - 1)

    n_iterations = frames[-1].iteration
    st.caption(f"Esta ejecución (k={k}) tiene {n_frames} pasos en total, hasta {n_iterations} iteraciones completas.")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    if col1.button("⏮ Reiniciar", use_container_width=True):
        st.session_state.frame_idx = 0
    if col2.button("◀ Anterior", use_container_width=True):
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
    if col3.button("Siguiente ▶", use_container_width=True):
        st.session_state.frame_idx = min(n_frames - 1, st.session_state.frame_idx + 1)
    play = col4.button("▶️ Reproducir animación completa", use_container_width=True)

    new_idx = st.slider("Arrastra para explorar cualquier paso", 0, n_frames - 1, st.session_state.frame_idx)
    if new_idx != st.session_state.frame_idx:
        st.session_state.frame_idx = new_idx

    chart_ph = st.empty()
    text_ph = st.empty()
    inertia_ph = st.empty()

    def _render(i):
        frame = frames[i]
        chart_ph.plotly_chart(_animation_figure(points_2d, frame, municipios), use_container_width=True, key=f"anim_{i}")
        text_ph.info(_animation_explanation(frame, k, features))
        steps_so_far = [(j, frames[j].inertia) for j in range(1, i + 1) if frames[j].inertia is not None]
        if steps_so_far:
            hist_df = pd.DataFrame(steps_so_far, columns=["paso", "inercia"])
            fig_hist = px.line(hist_df, x="paso", y="inercia", markers=True, title="Inercia a lo largo de los pasos")
            fig_hist.update_layout(height=250)
            inertia_ph.plotly_chart(fig_hist, use_container_width=True, key=f"anim_hist_{i}")
        else:
            inertia_ph.empty()

    if play:
        for i in range(st.session_state.frame_idx, n_frames):
            st.session_state.frame_idx = i
            _render(i)
            time.sleep(0.6)
        st.rerun()
    else:
        _render(st.session_state.frame_idx)

# ---------------------------------------------------------------------------
# TAB 3 — Matriz de confusión
# ---------------------------------------------------------------------------
with tab_comp:
    if ref_labels is None:
        st.info("Selecciona un esquema de referencia en la barra lateral para ver la matriz de confusión.")
    else:
        comp = compare_to_reference(result.labels, pd.Series(ref_labels, index=df.index))

        st.markdown(f"#### Comparación: clusters de k-means vs. **{reference_choice}**")
        st.caption(
            "Cada cluster (sin nombre por naturaleza) se empareja con la categoría de referencia más frecuente "
            "dentro de él (algoritmo húngaro, maximiza el acuerdo global)."
        )
        mapping_df = pd.DataFrame(
            [{"cluster": c, "categoría asignada": v} for c, v in sorted(comp.mapping.items())]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy (tras emparejar)", f"{comp.accuracy * 100:.1f}%", help="Coincidencia entre cluster emparejado y categoría real.")
        m2.metric("ARI (Adjusted Rand Index)", f"{comp.ari:.3f}", help="Acuerdo entre agrupaciones corregido por azar. 1 = idéntico, 0 = azar. No depende del emparejamiento de nombres.")
        m3.metric("NMI (Normalized Mutual Info)", f"{comp.nmi:.3f}", help="Información compartida entre las dos particiones, normalizada entre 0 y 1.")

        st.markdown("##### Matriz de confusión")
        cm = comp.confusion
        heat = go.Figure(
            data=go.Heatmap(
                z=cm.values,
                x=list(cm.columns),
                y=list(cm.index),
                text=cm.values,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        heat.update_layout(xaxis_title="Predicho (cluster emparejado)", yaxis_title="Real (referencia)")
        st.plotly_chart(heat, use_container_width=True)
        st.dataframe(cm, use_container_width=True)

        st.markdown("##### Precisión, recall y F1 por categoría")
        st.dataframe(comp.per_class.style.format(precision=3), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 — Documentación
# ---------------------------------------------------------------------------
with tab_docs:
    st.markdown(
        """
## ¿Qué hace esta aplicación?

Este simulador aplica el algoritmo **k-medias (k-means)** sobre un consolidado
de los 1.123 municipios de Colombia, replicando (con las variables disponibles)
la metodología usada por MinTIC en el documento *"Plazos para la implementación
del Marco de Referencia de Arquitectura Empresarial"* (2023) para agrupar
municipios según sus características socioeconómicas.

### Variables de caracterización
| Variable | Fuente original |
|---|---|
| Número de habitantes | DANE |
| Índice de Desempeño Institucional | Función Pública |
| Ingresos Totales Municipales | DNP |
| Puntaje IDF (Índice de Desempeño Fiscal) | DNP |
| Densidad poblacional (hab/km²) | DANE |

El estudio original de MinTIC usa una sexta variable (porcentaje de hogares
con acceso a internet) que no está disponible en este dataset, por lo que el
simulador agrupa con las cinco variables anteriores.

### La pestaña "Aula interactiva"
Mientras que la pestaña Simulador muestra el resultado final del clustering,
la pestaña **🎓 Aula interactiva** ejecuta el algoritmo paso a paso y deja
recorrer cada iteración con controles de reproducir/pausar/avanzar o
arrastrando una barra: inicialización de centroides, asignación de cada
municipio al centroide más cercano, recálculo de centroides, y así hasta la
convergencia — con una explicación de qué está pasando técnicamente en cada
paso, y una curva en vivo de cómo baja la inercia.

### ¿Cómo funciona k-means?
1. Se elige un número de clusters **k**.
2. Se seleccionan k centroides iniciales al azar.
3. Cada municipio se asigna al centroide más cercano (distancia euclidiana).
4. Cada centroide se recalcula como el promedio de los municipios asignados.
5. Se repiten los pasos 3-4 hasta que los centroides dejan de cambiar
   significativamente (convergencia).

Como las variables tienen escalas muy distintas (habitantes vs. puntajes de
0 a 100), se recomienda **estandarizarlas** antes de calcular distancias —
opción disponible en la barra lateral.

### ¿Cómo elegir el número de clusters (k)?
El simulador muestra dos criterios, igual que el estudio original:
- **Curva del codo**: la inercia (suma de distancias al cuadrado dentro de
  cada cluster) siempre baja al aumentar k; se busca el punto donde deja de
  bajar mucho ("codo").
- **Índice de Dunn**: relación entre la distancia mínima *entre* clusters y
  la distancia máxima *dentro* de un cluster. Valores más altos indican
  clusters más compactos y mejor separados. El estudio MinTIC 2023 usó este
  criterio y encontró que k=3 lo maximiza para los municipios de Colombia.

### Métricas internas de calidad del clustering
- **Inercia**: suma de distancias al cuadrado de cada punto a su centroide (menor es más compacto, pero siempre baja al aumentar k).
- **Silhouette**: qué tan similar es cada punto a su propio cluster vs. a los demás (rango -1 a 1, mayor es mejor).
- **Calinski-Harabasz**: razón entre dispersión inter e intra-cluster (mayor es mejor).
- **Davies-Bouldin**: similitud promedio entre cada cluster y el más parecido a él (menor es mejor).
- **Índice de Dunn**: ver arriba.

### Matriz de confusión y métricas de comparación
K-means es un algoritmo **no supervisado**: no conoce ninguna "categoría
verdadera" de antemano, y los números de cluster (0, 1, 2…) no tienen
significado propio. Para poder construir una matriz de confusión hace falta
una referencia externa con la que comparar. Este simulador ofrece dos:

1. **Categorías oficiales IDF (DNP)**: clasificación fiscal oficial del DNP
   a partir del Puntaje IDF (Deterioro < 40, Riesgo 40-60, Vulnerable 60-70,
   Sostenible 70-80, Solvente ≥ 80). Es una referencia real y de una sola
   variable, útil para ver si el agrupamiento *multivariado* recupera algo
   parecido a la clasificación fiscal oficial.
2. **Réplica del estudio MinTIC**: una aproximación construida en esta app
   (no la clasificación original, que no está publicada a nivel de
   municipio) que ordena los municipios por un score compuesto y corta los
   grupos para reproducir aproximadamente los tamaños reportados en el
   estudio (124 avanzados / 416 intermedios / 561 básicos, de 1.101).

Como los clusters no tienen nombre propio, cada uno se empareja con la
categoría de referencia más frecuente en su interior (algoritmo húngaro,
maximiza el acuerdo total). A partir de ese emparejamiento se calculan:

- **Matriz de confusión** y **accuracy**: intuitivas, pero dependen del
  emparejamiento elegido.
- **ARI** y **NMI**: métricas propias de comparación de agrupamientos, que
  **no** dependen de cómo se nombren los clusters — más rigurosas para
  evaluar qué tan parecidas son dos particiones del mismo conjunto.
- **Precisión, recall y F1 por categoría**, calculados tras el emparejamiento.

### Cómo ejecutar la app localmente
```bash
cd kmeans_app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Cómo publicarla gratis
- **Streamlit Community Cloud** (share.streamlit.io): conecta un repositorio
  de GitHub con este código y despliega en un clic, gratis para apps
  públicas.
- **Hugging Face Spaces**: crea un Space tipo "Streamlit", sube estos
  archivos (o conéctalo a GitHub), también gratis en el tier básico.

### Fuentes
- Consolidado de municipios con IDF 2022 (hoja `Consolidado 2025-2026` del
  archivo Excel, cruce DANE/DNP/Función Pública, ver hoja `Validación`).
- MinTIC (2023). *Plazos para la implementación del Marco de Referencia de
  Arquitectura Empresarial*, sección 1.1 "Agrupación de municipios".
        """
    )

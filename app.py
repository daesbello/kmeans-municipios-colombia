"""Simulador de K-Means sobre municipios de Colombia (IDF 2022).

Réplica interactiva de la metodología de agrupación de municipios descrita en
"Plazos para la implementación del Marco de Referencia de Arquitectura
Empresarial" (MinTIC, 2023): k-medias sobre variables socioeconómicas, con
selección del número de clusters vía Índice de Dunn.

El algoritmo se ejecuta paso a paso (inicialización, asignación,
actualización) y toda la vista — gráfica, métricas, matriz de confusión,
perfil de clusters — se recalcula para el paso actual, para poder ver en
vivo cómo evoluciona la simulación.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clustering import compare_to_reference, elbow_curve, frame_metrics, kmeans_animation
from src.data_processing import DEFAULT_FEATURES, FEATURE_LABELS, prepare_dataset
from src.reference_labels import REFERENCE_SCHEMES, label_idf_dnp, label_mintic_replica

CLUSTER_COLORS = px.colors.qualitative.Set1

STEP_TITLES = {
    "init": "Paso 0 · Inicialización",
    "assign": "Iteración {it} · Paso 1: Asignación",
    "update": "Iteración {it} · Paso 2: Actualización",
    "converged": "✅ Convergencia (iteración {it})",
    "max_iter": "⏹ Máximo de iteraciones alcanzado ({it})",
}


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
    fig.update_layout(
        title=STEP_TITLES.get(frame.step_type, frame.step_type).format(it=frame.iteration),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60),
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
    n_init_elbow = st.number_input(
        "n_init para la curva del codo", min_value=1, value=10, step=1,
        help="Solo afecta el análisis agregado de 'elección de k' (no la animación paso a paso, que usa una sola inicialización).",
    )

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
# Datos y referencia
# ---------------------------------------------------------------------------
df, meta = prepare_dataset(missing_strategy, features)
municipios = df["Municipio"].to_numpy()

if reference_choice == "Categorías oficiales IDF (DNP)":
    ref_labels = pd.Series(label_idf_dnp(df), index=df.index)
elif reference_choice == "Réplica estudio MinTIC (Avanzado/Intermedio/Básico)":
    ref_labels = pd.Series(label_mintic_replica(df, features), index=df.index)
else:
    ref_labels = None

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("📍 Simulador K-Means — Municipios de Colombia")
st.caption(
    "Dataset: Consolidado de municipios con IDF 2022 (DNP) · "
    f"{meta['n_usado']} de {meta['n_total']} municipios usados "
    f"({meta['n_missing']} con dato faltante, estrategia: {missing_strategy_label.lower()})."
)

tab_sim, tab_docs = st.tabs(["🎓 Simulador interactivo", "📚 Documentación"])

# ---------------------------------------------------------------------------
# TAB 1 — Simulador interactivo (todo integrado y en vivo)
# ---------------------------------------------------------------------------
with tab_sim:
    st.markdown("### Cómo piensa k-means, paso a paso")
    st.caption(
        "El algoritmo corre de forma manual (no `KMeans.fit` de una sola vez), guardando cada paso "
        "intermedio. Al reproducir o mover la barra, la gráfica, las métricas, el perfil de los clusters "
        "y la matriz de confusión se recalculan para ese paso exacto — todo integrado, en vivo."
    )
    st.latex(r"V_{C_k} = \sum_{i=1}^{m} (x_i^k - \mu_k)^2 \qquad \text{(inercia: variación total dentro del cluster } C_k\text{)}")
    st.caption(
        "⚠️ Esta animación usa **una sola** inicialización aleatoria de centroides (no las mejores de 10 "
        "reinicios), así que puede converger a un resultado distinto —y a veces peor— cada vez que cambias "
        "el `random_state`: es justamente el motivo por el que k-means en la práctica se corre varias veces."
    )

    @st.cache_data(show_spinner="Ejecutando k-means paso a paso...")
    def _kmeans_animation(missing_strategy, features_tuple, scale, k, random_state, max_iter=30):
        df_, _ = prepare_dataset(missing_strategy, list(features_tuple))
        return kmeans_animation(df_, list(features_tuple), k, scale, random_state, max_iter)

    X, points_2d, frames = _kmeans_animation(missing_strategy, tuple(features), scale, k, int(random_state))
    n_frames = len(frames)

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

    st.divider()

    state_ph = st.empty()
    chart_col_ph = st.empty()
    profile_ph = st.empty()
    confusion_ph = st.empty()

    def _render(i):
        frame = frames[i]
        metrics = frame_metrics(X, frame.labels)

        # --- Estado del algoritmo + métricas de calidad, para ESTE paso ---
        with state_ph.container():
            st.markdown("#### Estado del algoritmo en este paso")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Paso", i)
            c2.metric("Iteración", frame.iteration)
            c3.metric("Cambiaron de cluster", frame.n_changed if frame.n_changed is not None else "—")
            c4.metric("Inercia", f"{frame.inertia:,.0f}" if frame.inertia is not None else "—")
            c5.metric("Silhouette", f"{metrics['silhouette']:.3f}" if metrics["silhouette"] is not None else "—")
            c6.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}" if metrics["davies_bouldin"] is not None else "—")
            c7.metric("Índice de Dunn", f"{metrics['dunn_index']:.3f}" if metrics["dunn_index"] is not None else "—")

        # --- Gráfica + explicación + curva de inercia ---
        with chart_col_ph.container():
            gcol, tcol = st.columns([2, 1])
            with gcol:
                st.plotly_chart(_animation_figure(points_2d, frame, municipios), use_container_width=True, key=f"anim_{i}")
            with tcol:
                st.info(_animation_explanation(frame, k, features))
                steps_so_far = [(j, frames[j].inertia) for j in range(1, i + 1) if frames[j].inertia is not None]
                if steps_so_far:
                    hist_df = pd.DataFrame(steps_so_far, columns=["paso", "inercia"])
                    fig_hist = px.line(hist_df, x="paso", y="inercia", markers=True, title="Inercia por paso")
                    fig_hist.update_layout(height=260, margin=dict(t=40))
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"anim_hist_{i}")

        # --- Perfil de clusters para este paso ---
        with profile_ph.container():
            if frame.labels is None:
                st.info("Todavía no hay municipios asignados a un cluster en este paso.")
            else:
                st.markdown("#### Perfil de los clusters en este paso")
                df_step = df.copy()
                df_step["cluster"] = frame.labels.astype(str)
                profile = df_step.groupby("cluster")[features].mean()
                profile["n_municipios"] = df_step.groupby("cluster").size()
                profile["% del total"] = (profile["n_municipios"] / len(df_step) * 100).round(1)
                profile = profile.rename(columns=FEATURE_LABELS)
                pcol, scol = st.columns([2, 1])
                pcol.dataframe(profile.style.format(precision=1), use_container_width=True)
                size_fig = px.bar(
                    df_step["cluster"].value_counts().sort_index().reset_index(),
                    x="cluster", y="count", labels={"count": "N° municipios", "cluster": "Cluster"},
                    title="Tamaño de cada cluster",
                )
                size_fig.update_layout(height=300, margin=dict(t=40))
                scol.plotly_chart(size_fig, use_container_width=True, key=f"anim_sizes_{i}")

        # --- Matriz de confusión para este paso ---
        with confusion_ph.container():
            if frame.labels is None:
                pass
            elif ref_labels is None:
                st.info("Selecciona un esquema de referencia en la barra lateral para ver la matriz de confusión.")
            else:
                comp = compare_to_reference(frame.labels, ref_labels)
                st.markdown(f"#### Matriz de confusión en este paso — vs. **{reference_choice}**")
                st.caption(
                    "Cada cluster se empareja con la categoría de referencia más frecuente dentro de él "
                    "(algoritmo húngaro). Como el clustering aún puede estar cambiando, este emparejamiento "
                    "también se recalcula en cada paso."
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy (tras emparejar)", f"{comp.accuracy * 100:.1f}%")
                m2.metric("ARI", f"{comp.ari:.3f}", help="Adjusted Rand Index: acuerdo corregido por azar, no depende del emparejamiento.")
                m3.metric("NMI", f"{comp.nmi:.3f}", help="Normalized Mutual Information, entre 0 y 1.")

                cm = comp.confusion
                heat = go.Figure(
                    data=go.Heatmap(
                        z=cm.values, x=list(cm.columns), y=list(cm.index),
                        text=cm.values, texttemplate="%{text}", colorscale="Blues",
                    )
                )
                heat.update_layout(
                    xaxis_title="Predicho (cluster emparejado)", yaxis_title="Real (referencia)",
                    height=380, margin=dict(t=30),
                )
                hcol, tcol2 = st.columns([1, 1])
                hcol.plotly_chart(heat, use_container_width=True, key=f"anim_cm_{i}")
                tcol2.dataframe(comp.per_class.style.format(precision=3), use_container_width=True, hide_index=True)

    if play:
        for i in range(st.session_state.frame_idx, n_frames):
            st.session_state.frame_idx = i
            _render(i)
            time.sleep(0.5)
        st.rerun()
    else:
        _render(st.session_state.frame_idx)

    st.divider()

    # --- Tabla completa de municipios (snapshot del paso actual) ---
    current_frame = frames[st.session_state.frame_idx]
    st.markdown("#### Municipios y cluster asignado (paso actual)")
    if current_frame.labels is None:
        st.info("Todavía no hay asignación de clusters en este paso.")
    else:
        display_df = df.copy()
        display_df["cluster"] = current_frame.labels.astype(str)
        if ref_labels is not None:
            display_df["referencia"] = ref_labels
        show_cols = ["Municipio", "Código DIVIPOLA", "cluster"] + (["referencia"] if ref_labels is not None else []) + features
        display_df = display_df[show_cols].rename(columns=FEATURE_LABELS)
        search = st.text_input("Buscar municipio")
        if search:
            display_df = display_df[display_df["Municipio"].str.contains(search, case=False, na=False)]
        st.dataframe(display_df, use_container_width=True, height=350)
        st.download_button(
            "⬇️ Descargar resultados de este paso (CSV)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"kmeans_municipios_k{k}_paso{st.session_state.frame_idx}.csv",
            mime="text/csv",
        )

    st.divider()

    # --- Análisis agregado: elección de k (no depende del paso actual) ---
    with st.expander("📐 ¿Cómo elegir el número de clusters (k)? — curva del codo e Índice de Dunn"):
        st.caption(
            "Este análisis es independiente del paso actual: corre k-means completo (con reinicios "
            "múltiples) para varios valores de k, igual que en el estudio MinTIC 2023."
        )

        @st.cache_data(show_spinner="Calculando curva del codo...")
        def _elbow(missing_strategy, features_tuple, scale, k_min, k_max, random_state, n_init):
            df_, _ = prepare_dataset(missing_strategy, list(features_tuple))
            return elbow_curve(df_, list(features_tuple), scale, range(k_min, k_max + 1), random_state, n_init)

        elbow_df = _elbow(missing_strategy, tuple(features), scale, 2, 10, int(random_state), int(n_init_elbow))

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.line(elbow_df, x="k", y="inertia", markers=True, title="Curva del codo")
            fig.add_vline(x=k, line_dash="dash", line_color="crimson")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.line(elbow_df, x="k", y="dunn_index", markers=True, title="Índice de Dunn por k")
            fig.add_vline(x=k, line_dash="dash", line_color="crimson")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Documentación
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

### Todo integrado en un solo simulador
La pestaña **🎓 Simulador interactivo** ejecuta k-means paso a paso
(inicialización de centroides, asignación de cada municipio al centroide más
cercano, recálculo de centroides, y así hasta la convergencia) y, para el
paso en el que te encuentres —ya sea recorriéndolo con los controles de
reproducir/pausar/avanzar o arrastrando la barra—, recalcula en vivo:

- La gráfica de clusters (proyección PCA 2D) y la explicación técnica de qué está pasando.
- Las métricas de calidad (inercia, silhouette, Davies-Bouldin, Índice de Dunn).
- El perfil y tamaño de cada cluster.
- La matriz de confusión y las métricas de comparación (accuracy, ARI, NMI) contra el esquema de referencia elegido.

Así se puede ver, en una sola vista, cómo van cambiando **a la vez** la
asignación de cada municipio, la calidad del agrupamiento y su parecido con
una clasificación de referencia, a medida que el algoritmo converge.

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
Dentro del simulador, el desplegable **"¿Cómo elegir el número de clusters?"**
muestra dos criterios, igual que el estudio original:
- **Curva del codo**: la inercia (suma de distancias al cuadrado dentro de
  cada cluster) siempre baja al aumentar k; se busca el punto donde deja de
  bajar mucho ("codo").
- **Índice de Dunn**: relación entre la distancia mínima *entre* clusters y
  la distancia máxima *dentro* de un cluster. Valores más altos indican
  clusters más compactos y mejor separados. El estudio MinTIC 2023 usó este
  criterio y encontró que k=3 lo maximiza para los municipios de Colombia.

Este análisis usa k-means con múltiples reinicios (`n_init`) para cada valor
de k, a diferencia de la animación paso a paso, que deliberadamente usa una
sola inicialización para poder mostrar el proceso con claridad.

### Métricas internas de calidad del clustering
- **Inercia**: suma de distancias al cuadrado de cada punto a su centroide (menor es más compacto, pero siempre baja al aumentar k).
- **Silhouette**: qué tan similar es cada punto a su propio cluster vs. a los demás (rango -1 a 1, mayor es mejor).
- **Davies-Bouldin**: similitud promedio entre cada cluster y el más parecido a él (menor es mejor).
- **Índice de Dunn**: ver arriba.

Estas métricas se recalculan para el paso actual de la animación, así que
pueden no estar definidas (mostradas como "—") en pasos muy tempranos donde
todavía no hay al menos dos clusters con municipios asignados.

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

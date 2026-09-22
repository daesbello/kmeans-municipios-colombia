---
title: Simulador K-Means Municipios Colombia
emoji: 📍
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.64.0"
app_file: app.py
pinned: false
---

# Simulador K-Means — Municipios de Colombia (IDF 2022)

Aplicación en Python/Streamlit que aplica k-means a un consolidado de
municipios de Colombia, replicando (con las variables disponibles) la
metodología de agrupación descrita por MinTIC en *"Plazos para la
implementación del Marco de Referencia de Arquitectura Empresarial"* (2023).

Incluye: curva del codo, Índice de Dunn, proyección PCA de los clusters,
perfil de cada cluster, matriz de confusión contra dos esquemas de
referencia (categorías oficiales del IDF y una réplica del estudio MinTIC),
métricas ARI/NMI y precisión/recall/F1, y una pestaña de documentación con
la explicación completa de la metodología y las métricas.

## Autor
David Bello

## Aplicación en producción
🔗 **[kmeans-municipios-colombia.streamlit.app](https://kmeans-municipios-colombia.streamlit.app/)**
(desplegada en Streamlit Community Cloud, se actualiza sola con cada `git push` a `main`)

## Documentación del proyecto
- 📄 [Informe_Simulador_KMeans_David_Bello.pdf](Informe_Simulador_KMeans_David_Bello.pdf):
  documento único con todo organizado — resumen, enlaces, capturas de la app,
  arquitectura, metodología/resultados y el historial de prompts.
- [ARQUITECTURA.md](ARQUITECTURA.md): arquitectura de la app, dependencias,
  despliegue y el historial de prompts usados para construirla con IA.
- [INFORME.md](INFORME.md): metodología, resultados, matriz de confusión y
  conclusiones completas del trabajo.

## Estructura
```
kmeans_app/
├── app.py                  # App Streamlit (UI + orquestación)
├── src/
│   ├── data_processing.py  # Carga y limpieza del Excel
│   ├── clustering.py       # K-means, métricas, comparación con referencia
│   └── reference_labels.py # Esquemas de etiquetas de referencia
├── data/                   # Copia del dataset consolidado
├── requirements.txt
├── ARQUITECTURA.md         # Arquitectura, dependencias, despliegue, prompts
├── INFORME.md              # Metodología y resultados
└── .streamlit/config.toml  # Tema
```

## Uso local
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Abre http://localhost:8501

## Despliegue

**Plataforma usada: Streamlit Community Cloud** (gratuita). El repo de
GitHub está conectado directamente a [share.streamlit.io](https://share.streamlit.io);
cada `git push` a `main` redespliega la app automáticamente. Para conectar
un repo nuevo ahí: "New app" → seleccionar el repo, la rama `main` y
`app.py` como archivo principal.

<details>
<summary>Nota: por qué no Hugging Face Spaces</summary>

El bloque YAML al inicio de este README es la configuración que Hugging Face
Spaces necesita (SDK, versión, archivo principal) por si se quiere usar en
el futuro. No se usó como plataforma final porque, al construir esta app,
Hugging Face Spaces exigía una suscripción paga para correr Spaces con SDK
Streamlit/Gradio/Docker (solo el SDK "Static", sin backend Python, es
gratuito) — ver el detalle en [ARQUITECTURA.md](ARQUITECTURA.md#6-despliegue).
</details>

Ver la pestaña **Documentación** dentro de la app para el detalle completo
de la metodología, las variables y cómo interpretar cada métrica.

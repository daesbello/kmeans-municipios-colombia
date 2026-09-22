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

## Despliegue gratuito en Hugging Face Spaces
El bloque YAML al inicio de este README es la configuración que Hugging Face
Spaces necesita para saber cómo ejecutar la app (SDK, versión, archivo
principal). Pasos:

1. Crea una cuenta gratuita en [huggingface.co](https://huggingface.co) si no
   tienes una.
2. Ve a **New Space** → elige un nombre, SDK **Streamlit**, visibilidad
   pública, y créalo. Esto genera un repositorio git propio de Hugging Face
   (distinto del de GitHub).
3. Conecta ese Space con el repo de GitHub (`kmeans-municipios-colombia`) de
   una de estas formas:
   - **Más simple**: en la página del Space, usa la opción de subir archivos
     y arrastra el contenido de esta carpeta (o usa "Import from GitHub" si
     tu cuenta la tiene habilitada).
   - **Vía git**, desde esta carpeta local:
     ```bash
     git remote add space https://huggingface.co/spaces/<tu-usuario-hf>/<nombre-space>
     git push space main
     ```
     (te pedirá tu usuario y un *access token* de Hugging Face, que generas en
     Settings → Access Tokens de tu cuenta).
4. El Space se construye solo y queda disponible en
   `https://huggingface.co/spaces/<tu-usuario-hf>/<nombre-space>`.

Si en algún momento prefieres Streamlit Community Cloud en su lugar, el
proceso es igual de simple: en [share.streamlit.io](https://share.streamlit.io)
conectas el repo de GitHub y seleccionas `app.py` como archivo principal.

Ver la pestaña **Documentación** dentro de la app para el detalle completo
de la metodología, las variables y cómo interpretar cada métrica.

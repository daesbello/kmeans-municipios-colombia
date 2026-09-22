# Informe — Simulador K-Means de Municipios de Colombia (IDF 2022)

**Autor:** David Bello
**Fecha:** 22 de septiembre de 2026
**Repositorio:** https://github.com/daesbello/kmeans-municipios-colombia
**Aplicación en vivo:** https://kmeans-municipios-colombia.streamlit.app/

---

## 1. Resumen ejecutivo

Este trabajo implementa el algoritmo de agrupamiento **k-medias (k-means)** sobre
un consolidado de los municipios de Colombia, replicando —con las variables
disponibles— la metodología descrita por el Ministerio de Tecnologías de la
Información y las Comunicaciones (MinTIC) en el documento *"Plazos para la
implementación del Marco de Referencia de Arquitectura Empresarial"* (2023),
usado allí para segmentar municipios según su capacidad institucional y
socioeconómica.

El resultado es una aplicación web interactiva (estilo simulador) construida
en Python con Streamlit, que permite ajustar el número de clusters y las
variables de entrada, visualizar la curva del codo y el Índice de Dunn,
proyectar los clusters en 2D, y comparar el agrupamiento resultante contra dos
esquemas de clasificación de referencia mediante matriz de confusión y
métricas de concordancia (ARI, NMI, precisión, recall, F1).

## 2. Objetivo y contexto

El estudio de MinTIC (2023) agrupó los 1.101 municipios de Colombia en 3
clusters usando k-medias sobre variables socioeconómicas, con el fin de
asignar plazos diferenciales para la implementación del Marco de Referencia
de Arquitectura Empresarial (MRAE). El objetivo de este trabajo es:

1. Reproducir esa metodología de forma interactiva y explicable, usando el
   dataset de municipios con Índice de Desempeño Fiscal (IDF) 2022.
2. Construir una herramienta tipo *simulador* que permita explorar el efecto
   de cambiar k, las variables y el preprocesamiento sobre el resultado del
   agrupamiento.
3. Evaluar cuantitativamente el agrupamiento obtenido, tanto con métricas
   internas (inercia, silhouette, Calinski-Harabasz, Davies-Bouldin, Índice
   de Dunn) como comparándolo contra clasificaciones externas de referencia
   mediante matriz de confusión.

## 3. Fuentes de datos

| Fuente | Contenido | Uso |
|---|---|---|
| `data/Consolidado_Municipios_Colombia_con_IDF_2022.xlsx` (hoja `Consolidado 2025-2026`) | 1.123 municipios: población, ingresos, desempeño institucional, IDF 2022, densidad poblacional | Dataset principal para el clustering |
| Hoja `Validación` del mismo Excel | Metadatos del cruce de datos (DANE/DNP/Función Pública) | Verificación: confirma que 21 municipios quedan sin IDF por ser Áreas No Municipalizadas (ANM) |
| *Plazos para la implementación del Marco de Referencia de Arquitectura Empresarial* (MinTIC, 2023) | Metodología original de agrupación de municipios (sección 1.1) | Referencia metodológica: variables, algoritmo, criterio de selección de k, resultados (124/416/561 municipios) |

De las 6 variables usadas en el estudio original (población, ingresos
municipales, % hogares con acceso a internet, desempeño institucional,
densidad poblacional, IDF), el dataset disponible contiene 5; **no incluye
el porcentaje de hogares con acceso a internet**, por lo que el agrupamiento
de este trabajo se hace con las 5 variables restantes.

De los 1.123 municipios del consolidado, **21 corresponden a Áreas No
Municipalizadas (ANM)** sin Puntaje IDF reportado por el DNP, y se excluyen
del análisis por defecto (quedan 1.101 municipios, el mismo número que usa el
estudio de MinTIC). La aplicación permite alternativamente imputar estos
valores con la mediana en vez de excluirlos.

## 4. Metodología implementada

**Preprocesamiento:** las variables se estandarizan (`StandardScaler`, media
0 y desviación 1) antes de calcular distancias, ya que tienen escalas muy
distintas (habitantes vs. puntajes de 0 a 100).

**Algoritmo:** k-medias (`sklearn.cluster.KMeans`), que asigna cada municipio
al centroide más cercano y recalcula los centroides iterativamente hasta
converger, minimizando la suma de distancias euclidianas al cuadrado dentro
de cada cluster.

**Selección de k:** se replican los dos criterios usados en el estudio
original:
- **Curva del codo**: inercia (suma de distancias al cuadrado intra-cluster) en función de k.
- **Índice de Dunn**: razón entre la distancia mínima *entre* clusters y la
  distancia máxima *dentro* de un cluster; valores más altos indican una
  mejor separación. El estudio de MinTIC (2023) reportó que k=3 maximiza
  este índice para los municipios de Colombia.

| k | Inercia | Silhouette | Índice de Dunn |
|---|---|---|---|
| 2 | 4,529.95 | 0.934 | 0.5249 |
| **3** | **3,538.96** | **0.294** | **0.0010** |
| 4 | 2,836.03 | 0.287 | 0.0023 |
| 5 | 2,110.83 | 0.293 | 0.0051 |
| 6 | 1,646.36 | 0.296 | 0.0026 |
| 7 | 1,298.08 | 0.311 | 0.0066 |
| 8 | 1,088.50 | 0.293 | 0.0045 |

En esta ejecución, el Índice de Dunn es máximo en k=2 y cae abruptamente a
partir de k=3 (ver sección 9, Limitaciones): esto se debe a la presencia de
municipios extremadamente grandes (p. ej. Bogotá) que distorsionan la
distancia máxima intra-cluster, una conocida sensibilidad del Índice de Dunn
a valores atípicos. Aun así, se fija **k=3** como configuración por defecto
del simulador, replicando el valor reportado en el estudio de MinTIC 2023.

## 5. Herramienta desarrollada

La aplicación (`app.py`, Streamlit) tiene tres secciones:

1. **Simulador**: controles para variables, manejo de datos faltantes,
   estandarización, k y parámetros del algoritmo; muestra métricas de
   calidad, curva del codo, Índice de Dunn, proyección PCA 2D de los
   clusters, perfil de cada cluster y tabla descargable en CSV.
2. **Matriz de confusión y métricas**: compara los clusters contra dos
   esquemas de referencia seleccionables (ver sección 7).
3. **Documentación**: explicación completa de la metodología, las variables
   y cómo interpretar cada métrica, integrada dentro de la propia app.

Código fuente organizado en `src/data_processing.py` (carga y limpieza),
`src/clustering.py` (k-means y métricas) y `src/reference_labels.py`
(esquemas de referencia).

## 6. Resultados obtenidos (configuración por defecto: k=3, 5 variables, estandarizado, 1.101 municipios)

| Métrica | Valor |
|---|---|
| Municipios usados | 1,101 de 1,123 |
| Inercia | 3,538.96 |
| Silhouette | 0.294 |
| Calinski-Harabasz | 305.0 |
| Davies-Bouldin | 1.244 |
| Índice de Dunn | 0.001 |

**Tamaño y perfil de los clusters:**

| Cluster | N° municipios | Población media | Desempeño institucional | Ingresos medios (millones $) | IDF medio | Densidad media (hab/km²) |
|---|---|---|---|---|---|---|
| 0 | 550 | 23,688 | 55.0 | 57,518 | 51.7 | 271.7 |
| 1 | 544 | 46,053 | 73.8 | 115,137 | 59.7 | 544.4 |
| 2 | 7 | 2,168,542 | 72.8 | 17,537,625 | 69.1 | 24,871.8 |

**Interpretación:** el cluster 2 agrupa solo 7 municipios (las grandes
ciudades/capitales, con población e ingresos varios órdenes de magnitud por
encima del resto), mientras que los clusters 0 y 1 dividen al resto del país
en un grupo de condiciones más básicas (0) y uno intermedio-avanzado (1). Esto
es coherente con el hallazgo del estudio MinTIC (2023) de un cluster "más
avanzado" pequeño, uno intermedio y uno "más básico" (aunque los tamaños
relativos difieren, ver limitaciones).

## 7. Comparación con referencias externas (matriz de confusión)

K-means es un algoritmo no supervisado: no conoce ninguna categoría real de
antemano. Para construir una matriz de confusión se usan dos referencias
externas, y cada cluster se empareja con la categoría de referencia más
frecuente en su interior (algoritmo húngaro, maximiza el acuerdo global).

### 7.1 Categorías oficiales IDF (DNP)

Clasificación fiscal oficial a partir del Puntaje IDF (Deterioro < 40, Riesgo
40-60, Vulnerable 60-70, Sostenible 70-80, Solvente ≥ 80):

| | Pred: Deterioro | Pred: Riesgo | Pred: Solvente | Pred: Sostenible | Pred: Vulnerable |
|---|---|---|---|---|---|
| **Real: Deterioro** | 0 | 41 | 0 | 0 | 2 |
| **Real: Riesgo** | 0 | 432 | 0 | 1 | 272 |
| **Real: Solvente** | 0 | 0 | 0 | 0 | 2 |
| **Real: Sostenible** | 0 | 6 | 0 | 4 | 42 |
| **Real: Vulnerable** | 0 | 71 | 0 | 2 | 226 |

- **Accuracy (tras emparejar):** 60.1% — **ARI:** 0.093 — **NMI:** 0.119

| Categoría | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Deterioro | — | 0.000 | — | 43 |
| Riesgo | 0.785 | 0.613 | 0.688 | 705 |
| Solvente | — | 0.000 | — | 2 |
| Sostenible | 0.571 | 0.077 | 0.136 | 52 |
| Vulnerable | 0.415 | 0.756 | 0.536 | 299 |

**Interpretación:** el acuerdo es bajo (ARI 0.093, NMI 0.119). Esto es
esperable: el IDF es una sola variable fiscal, mientras que el clustering es
multivariado (población, ingresos, desempeño institucional, densidad, IDF).
El agrupamiento multivariado no busca reconstruir la clasificación fiscal,
sino capturar similitud socioeconómica general — un municipio puede ser
fiscalmente "Vulnerable" pero agruparse por tamaño/densidad con otros
municipios en "Riesgo".

### 7.2 Réplica del estudio MinTIC (Avanzado/Intermedio/Básico)

Referencia construida en esta app (no es la clasificación original del
estudio, que no está publicada a nivel de municipio) ordenando los
municipios por un score compuesto y cortando los grupos para reproducir
aproximadamente los tamaños reportados en 2023 (124/416/561 de 1.101):

| | Pred: Avanzado | Pred: Básico | Pred: Intermedio |
|---|---|---|---|
| **Real: Avanzado** | 7 | 1 | 116 |
| **Real: Básico** | 0 | 518 | 43 |
| **Real: Intermedio** | 0 | 31 | 385 |

- **Accuracy (tras emparejar):** 82.7% — **ARI:** 0.610 — **NMI:** 0.546

| Categoría | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Avanzado | 1.000 | 0.056 | 0.107 | 124 |
| Básico | 0.942 | 0.923 | 0.932 | 561 |
| Intermedio | 0.708 | 0.925 | 0.802 | 416 |

**Interpretación:** el acuerdo es mucho mayor (ARI 0.610) porque esta
referencia se construye a partir de las mismas variables que el clustering.
Es esperable un acuerdo alto en "Básico" e "Intermedio" (que concentran la
mayoría de los municipios), pero el cluster "Avanzado" del k-means (solo 7
municipios, las grandes ciudades) es mucho más pequeño que el grupo
"Avanzado" de la referencia (124 municipios) — de ahí el recall bajo (0.056)
en esa categoría: el k-means aísla únicamente a los outliers extremos como
cluster propio, mientras que la referencia agrupa a 124 municipios
"grandes/avanzados" de forma más amplia.

## 8. Limitaciones

- **Variable faltante:** no se dispone del porcentaje de hogares con acceso a
  internet (usado en el estudio original), por lo que el agrupamiento se
  basa en 5 de las 6 variables originales.
- **Sensibilidad a outliers:** la población y los ingresos de las grandes
  ciudades (en particular Bogotá) son órdenes de magnitud mayores que los del
  resto de municipios. Aun estandarizando las variables, esto genera un
  cluster minúsculo (7 municipios) y distorsiona fuertemente el Índice de
  Dunn, que depende de la distancia máxima intra-cluster.
- **Referencia "réplica MinTIC" es aproximada:** no es la clasificación
  original publicada por MinTIC a nivel de municipio (esa no está
  disponible públicamente), sino una construcción propia que solo reproduce
  los tamaños de grupo reportados.
- **21 municipios excluidos** por no tener Puntaje IDF (Áreas No
  Municipalizadas), configurable en la app para imputarlos en su lugar.

## 9. Conclusiones

1. Es posible reproducir, con datos públicos y las variables disponibles, un
   agrupamiento de municipios de Colombia consistente en su estructura
   general con el reportado por MinTIC (2023): un grupo pequeño de grandes
   ciudades, y una división del resto del país en dos niveles de desarrollo
   socioeconómico relativo.
2. El agrupamiento multivariado (k-means) y la clasificación fiscal
   unidimensional oficial (IDF-DNP) miden cosas distintas y tienen un
   acuerdo bajo (ARI 0.093): el desempeño fiscal por sí solo no determina el
   perfil socioeconómico general de un municipio.
3. El Índice de Dunn y la inercia son sensibles a outliers extremos
   (grandes ciudades); en un análisis más detallado valdría la pena evaluar
   una transformación logarítmica de población e ingresos, o excluir
   explícitamente las grandes capitales antes de determinar k.
4. La herramienta desarrollada permite explorar estas decisiones (variables,
   preprocesamiento, k, manejo de datos faltantes) de forma interactiva, lo
   que facilita entender el efecto de cada supuesto metodológico sobre el
   resultado final del agrupamiento.

## 10. Cómo reproducir

```bash
git clone https://github.com/daesbello/kmeans-municipios-colombia
cd kmeans-municipios-colombia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
O accede directamente a la versión desplegada:
https://kmeans-municipios-colombia.streamlit.app/

## 11. Referencias

- Ministerio de Tecnologías de la Información y las Comunicaciones (MinTIC).
  (2023). *Plazos para la implementación del Marco de Referencia de
  Arquitectura Empresarial*. Versión 3.0.
- Departamento Nacional de Planeación (DNP). Índice de Desempeño Fiscal
  (IDF), vigencia 2022.
- Consolidado de municipios de Colombia con IDF 2022 (cruce DANE / DNP /
  Función Pública), hoja `Validación` del archivo de datos de este
  repositorio.

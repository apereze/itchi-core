# Avance de ITCHI v0.2.1: máximo de lluvia en P95, viento Saffir-Simpson y deslizamientos

Documento del avance para la tutora. Fija qué se compara, qué se espera antes de
ver los resultados y cómo se reproduce. Los resultados se leen en los notebooks
11–14 de `notebooks/third test/`.

---

## 1. Qué se muestra

| Caso | Pregunta | Experimentos | Notebook |
|---|---|---|---|
| Esquemas | Cómo determina ITCHI v0.2.1 el peligro | — (sin datos) | 11 |
| 1. Lluvia | Cómo se ve ITCHI si el máximo de lluvia pasa del P99 al P95 de 6 h, con la rampa con suelo saturado de v0.2.0 | `v020`, `p95` | 12 |
| 2. Viento | Cómo se ve ITCHI con la cola en 12 m/s y la escala Saffir-Simpson, frente a v0.2.0 | `v020`, `w12`, `ss` | 13 |
| 3. Deslizamientos | Cómo define ITCHI el peligro en el municipio de un deslizamiento y con cuánta anticipación fue alto | `v020`, `p95`, `ss`, `p95ss` | 14 |

Casos 1 y 2: Emily 2005, Alex 2010, Odile 2014, Patricia 2015 y Cristóbal 2020,
en malla (0.1°) y por municipio. Odile es de 2014 (la solicitud decía 2012;
no hubo un Odile que afectara a México ese año).

Caso 3: Manuel 2013 (alud de La Pintada, Atoyac de Álvarez, Guerrero) y Earl 2016
(Huauchinango, Puebla).

## 2. Experimentos

Overlays sobre `configs/default.yaml` en `configs/experiments/v021/`:

| Etiqueta | Lluvia | Cola de viento | Escala de viento |
|---|---|---|---|
| `v020` | Matriz v0.2: P75 → 0, P90 → 0.25, P95 → 0.5, **P99 → 1**; escalada por R24 (P90 → P95 de 24 h) y A72 (0 → P75 de 72 h), P95 → 0.8 | Hasta R_W06 (2 m/s) | Relativa a Vmax |
| `p95` | P75 → 0, P90 → 0.25, **P95 → 1**; la misma escalada sube el **P90 a 0.8** | Hasta R_W06 | Relativa a Vmax |
| `w12` | v0.2 | Hasta 12 m/s | Relativa a Vmax |
| `ss` | v0.2 | Hasta 12 m/s | **Saffir-Simpson** |
| `p95ss` | Como `p95` | Hasta 12 m/s | Saffir-Simpson |

Código nuevo en la librería (las opciones por defecto no cambian v0.2.0):

- `precipitation.PRECIP_MATRIX_V02_P95`;
- `wind.normalize_wind_hazard_absolute`; `constants.SAFFIR_SIMPSON_KT`,
  `SAFFIR_SIMPSON_HAZARD_NODES`;
- `compute_w06_wind_hazard_with_tail(hazard_scale=...)` y
  `compute_itchi_snapshot_v02(wind_hazard_scale=..., wind_absolute_nodes=...)`;
- configuración `wind.hazard_scale` y `wind.saffir_simpson_nodes`.

## 3. Caso 1: máximo de lluvia en P95

Mover solo el nodo del máximo al P95 dejaría sin efecto la escalada (en v0.2 la
escalada sube a 0.8 el nodo inmediatamente debajo del máximo, el P95, que ya
valdría 1). Por eso la variante recorre la lógica de v0.2 una clase: sin
escalada P75 → 0, P90 → 0.25, P95 → 1; con suelo saturado y persistencia,
P90 → 0.8. Es la idea de M1 del Bloque 1 (allí P90 → 1 y la rampa base tenía
P90 → 0.5).

P6 (mm en 6 h, mediana de México) a partir del cual empieza cada clase:

| | Clase 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| P99, sin escalada | 9.8 | 18.0 | 25.6 | 33.9 | 43.2 |
| P99, suelo saturado | 9.8 | 18.0 | 22.6 | 26.0 | 29.3 |
| P95, sin escalada | 9.8 | 18.0 | 21.9 | 24.4 | 26.8 |
| P95, suelo saturado | 9.8 | 12.4 | 15.0 | 17.5 | 20.1 |

**Expectativas (antes de ver los casos; notebook 11, A2 y A3):**

1. P95 nunca baja el peligro (Δclase ≥ 0).
2. Sin escalada, lo que supera el P95 pasa a clase 5 y lo que está entre P90 y
   P95 sube hasta dos clases.
3. Con suelo saturado la clase 5 empieza en el P90, así que los núcleos de
   lluvia persistente crecen más.
4. La clase 4 se estrecha de 9.3 a 2.4 mm de P6 (sin escalada).

**Antecedente.** El Bloque 1 ya evaluó el P95 como máximo y no lo adoptó: no
mejoró la discriminación (PSS de clase 5: 0.553 → 0.546) y subió un 61 % las
alertas de clase 5 (`docs/experimento_b1.md`, sección 9). Aquí solo se muestra
cómo se ve el índice; adoptarlo sería una decisión operativa.

## 4. Caso 2: viento absoluto homologado con Saffir-Simpson

| Viento local (kt, sostenido 1 min) | Categoría | H_W | Clase |
|---|---|---|---|
| 23.3 (12 m/s) – 34 | cola | 0 → 0.2 | 1 mínimo |
| 34 – 63 | tormenta tropical | 0.2 → 0.4 | 2 bajo |
| 64 – 95 | huracán 1–2 | 0.4 → 0.6 | 3 moderado |
| 96 – 112 | huracán 3 | 0.6 → 0.8 | 4 alto |
| ≥ 113 | huracán 4–5 | 0.8 → 1 (1 desde 137 kt) | 5 máximo |

- Lineal entre nodos; la cola por debajo de 34 kt es la de v0.2 con h34 = 0.2.
- Dentro del RMW (ojo) el peligro es el de Vmax: la pared del ojo cruza el píxel
  dentro de la ventana de 6 h. v0.2.0 asignaba 0.2.
- Las clases son (inferior, superior]: un viento exactamente en un límite de
  categoría cae en la clase de abajo (solo sobre una isolínea).
- La agrupación es una propuesta. La alternativa de 4 niveles del plan v0.3
  (< 34, TT, H1–H2, ≥ 96 kt) se prueba solo con `wind.saffir_simpson_nodes`.

**Expectativas:** el corte en 12 m/s solo reduce (Δclase ≤ 0) y actúa lejos del
centro; Saffir-Simpson baja sobre todo las tormentas débiles; donde la lluvia
domina, ITCHI no cambia.

**Limitaciones.** Viento del perfil W06 (PA21) calibrado a R34, sin reducción
sobre tierra (plan v0.3, decisión 6). El corte en 12 m/s no tiene todavía una
justificación de daño (sería con ráfaga).

## 5. Caso 3: deslizamientos y anticipación

| Ciclón | Municipio | Hora local | UTC | Fuente |
|---|---|---|---|---|
| Manuel 2013 | Atoyac de Álvarez (Gro.) | 16 sep 2013, ~15:30 | 20:30 (± 1 h) | La Jornada, 3 nov 2013; Proceso, 21 sep 2013 |
| Earl 2016 | Huauchinango (Pue.) | 7 ago 2016, madrugada | 08:00 (± 3 h) | Boletín ERN, 8 ago 2016 |

Horario de verano (UTC−5) en ambos años. Algunas fuentes secundarias fechan La
Pintada el 14 de septiembre; los reportes de campo dan el lunes 16.

**Hechos que condicionan la lectura.** Manuel se disipó sobre México hacia las
06 UTC del 16 y se regeneró en el golfo de California el 17 (NHC, TCR EP132013):
no hay posición del ciclón al ocurrir el deslizamiento. Ingrid estaba activa en
el golfo de México. Por eso se muestran ITCHI en las posiciones y el H_P continuo
(lluvia sin atribución) en todos los tiempos.

**Anticipación** (`casos_v021.lead_times`, fijada antes de ver las series): con
los tiempos ≤ t_evento en las 120 h previas, para cada clase k, inicio del
último episodio continuo con clase ≥ k; anticipación = t_evento − inicio; horas
desde el fin = t_evento − último tiempo del episodio. Es anticipación observada
(MSWEP): una cota superior de la que daría un pronóstico (fase F5).

**Limitaciones.** ITCHI no incluye pendiente ni suelo; el máximo municipal de un
municipio grande (Atoyac va de la costa a la sierra) puede venir de un píxel
lejos de la localidad; dos eventos ilustran, no validan.

## 6. Reproducción (Anaconda Prompt, entorno `itchi`)

```bash
cd scripts
python casos_v021.py                  # caché en outputs/diagnostics/avance_v021/
python -m pytest ../tests/test_v021_avance.py ../tests/test_casos_v021.py -v
```

Después, los notebooks 11 a 14. Figuras en `outputs/figures/avance_v021/`.

## 7. Decisiones para discutir

1. ¿Máximo de lluvia en P95 (por razones operativas)? Si sí, ¿escalada P90 → 0.8 o P90 → 1 (M1)?
2. Agrupación de las categorías Saffir-Simpson en clases (5 o 4 niveles).
3. Justificación física del corte en 12 m/s.
4. Estadístico municipal: máximo, P90 o fracción de área.
5. Atribución después de la disipación (remanentes) para los deslizamientos.

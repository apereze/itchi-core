# Contexto para continuar ITCHI v0.3 en una conversación nueva

Pega o adjunta este documento al empezar. Resume dónde quedó el proyecto al
cierre de ITCHI v0.2.1 (30 de octubre de 2026) y qué sigue.

## Estado del repositorio (github.com/apereze/itchi-core)

- `main` en a23239c (v0.2.0) + parches de v0.2.1 aplicados localmente por Adolfo:
  consistencia, Bloque 1 y cierre. Al cerrar: commit en `release/v0.2.1`, PR a
  `main` y tag `v0.2.1`.
- Rama `dev/v0.3`, creada desde `main` ya con v0.2.1. Su primer commit es F1
  (periodos de retorno): parche `itchi-v0.3-F1-periodos-retorno.patch`, escrito y
  probado con datos sintéticos, **sin correr todavía con MSWEP**.
- Adolfo hace los commits y el push; Claude prepara los parches (`git apply`) y
  da los comandos.
- Los scripts se corren en el **Anaconda Prompt de Windows** (entorno `itchi`),
  no en WSL. Los acumulados de 6 h están en `D:/MSWEP_V316_6H`.

## Forma de trabajo

- **Tutora:** muy meticulosa. Los avances deben ir explicados y esquematizados,
  sin huecos. Las decisiones de calibración se discuten con ella antes de
  implementarse.
- **Figuras:** cada experimento lleva un notebook con figuras y una guía de
  lectura (como los notebooks 09 y 10 de `notebooks/third test/`), no solo
  tablas.
- **Criterios:** se fijan por escrito antes de ver los resultados.
- **Pruebas:** las funciones científicas son NumPy puro y tienen pruebas en
  `tests/`.

## Primera tarea en la conversación nueva

1. Correr F1: `python scripts/periodos_retorno.py` (~30 min). Después, el notebook
   `notebooks/third test/10_periodos_retorno.ipynb`.
2. Revisar juntos la tabla resumen y las figuras R3 (rareza real de P75–P99) y R4
   (GEV frente a POT).
3. Diseñar F2 (catálogo de estados): formato de almacenamiento, agregación
   municipal y viento local por cuadrante.

---

# Plan de ITCHI v0.3: peligrosidad climatológica de los ciclones tropicales en México

**Objetivo.** Convertir ITCHI en una medida de **peligro** de los ciclones
tropicales que afectaron México entre 2000 y 2025, en malla y por municipio. El
peligro se define por la rareza climatológica de la lluvia y por umbrales
físicos absolutos de viento, no por la calibración contra declaratorias.
Después, medir con cuánta anticipación lo detectan los pronósticos (AIFS).

**Versión.** v0.3.0. Cambia la definición del peligro (periodos de retorno,
viento absoluto y número de niveles distinto de SIAT-CT), así que no es un
parche de v0.2. Se desarrolla en la rama `dev/v0.3` desde `main` con v0.2.1.

---

## 1. Decisiones de partida

| # | Decisión | Por qué |
|---|---|---|
| P1 | El peligro por lluvia es la **rareza local**: periodo de retorno T (años) de R6, R24 y R72 por píxel, con todas las causas de lluvia (MSWEP V3, mayo–noviembre, 2000–2025) | Es comparable entre regiones y no depende de los impactos. El P95 de 6 h se alcanza unas 3 veces por temporada en cada píxel; no es un «máximo» |
| P2 | Las declaratorias son solo una **prueba ordinal**: el nivel k+1 debe tener más declaratorias que el nivel k | Son un registro de impactos ruidoso y muy general; el subregistro sesga los umbrales absolutos, pero menos el orden |
| P3 | El sesgo de los pronósticos se corrige con la **climatología del propio modelo** (percentiles o periodos de retorno de AIFS por anticipación), no bajando los umbrales del índice | La definición del peligro observado no cambia. Bajar el umbral de *alerta* para avisar antes es una decisión de la capa de pronóstico |
| P4 | El viento usa **umbrales absolutos** de Saffir-Simpson aplicados al viento local de cada píxel (perfil W06 por cuadrante, calibrado a R34), en niveles escalonados | v0.2 normaliza con la Vmax de cada tormenta (D9): una tormenta de 40 kt llega a H_W = 1 |
| P5 | El área de influencia es ROCLOUD por cuadrante con piso R_W06 (si falta, vale 0 o es menor, se usa R_W06); R_W06 también marca el fin de la cola de viento | Igual que en v0.2.1 |
| P6 | El número de niveles no está fijo (de 2 a 5); se elige con criterios explícitos (sección 3, F3) | Los niveles SIAT-CT son de comunicación operativa, no de peligro |
| P7 | Se separan **datos** y **esquema**: primero se guarda un catálogo de variables físicas bajo cada ciclón y después se evalúa cualquier esquema sobre él | Así se pueden probar umbrales y niveles en segundos, sin releer MSWEP |

## 2. Lo que ya existe (v0.2.1)

- **Código:** `compute_itchi_snapshot_v02` (malla, por tiempo sinóptico); tabla de
  clases P6 × R24 × A72; perfiles W06 y radios R_W06 (PA21); regla ROCLOUD con
  piso R_W06; configuración por overlays (`configs/experiments/`).
- **Datos procesados:**
  - acumulados de 6 h de MSWEP V3 en `D:/MSWEP_V316_6H/mswep_v3_6h_AAAA.nc`;
  - percentiles de 6, 24 y 72 h (`p_wet_raw`) en
    `data/processed/mswep_v3_{d}h_percentiles_may_nov.nc`;
  - pares y pasos municipio–ciclón dentro de R_W06 en
    `outputs/diagnostics/declaratorias_6h/` (381 929 pares; 2 977 768 pasos);
  - declaratorias asociadas a ciclones por fecha.
- **Resultados que orientan v0.3:**
  - verificación v0.2 (`docs/verification_v02.md`);
  - Bloque 1 (`docs/experimento_b1.md`, sección 9): P95 como máximo no mejora la
    discriminación; R24 y A72 importan por encima de P95; cortar la cola de
    viento recupera la clase 1.
- **Escrito para v0.3, todavía sin correr con MSWEP:**
  - `itchi.return_periods`, `scripts/periodos_retorno.py`,
    `scripts/figuras_retorno.py` y `notebooks/third test/10_periodos_retorno.ipynb`;
  - documentación en `docs/periodos_retorno.md`.

## 3. Fases

El orden de trabajo de origen contaba dos veces los umbrales de periodo de
retorno (1a y 2). Aquí son una sola fase, F1.

### F1. Umbrales climatológicos de lluvia (orden de trabajo 1a y 2)

- **Qué hace.** Por píxel y duración (6, 24, 72 h):
  - máximos estacionales + GEV → periodo de retorno T;
  - picos independientes + GPD → intervalo de recurrencia R, que sirve también
    por debajo de un año.
- **Entregable.**
  - `data/processed/mswep_v3_return_periods_may_nov.nc`;
  - figuras R1–R5 y tabla resumen (notebook 10), incluida la rareza real de P75,
    P90, P95 y P99.
- **Terminado cuando:**
  - GEV y POT coinciden en T = 10 años (R4);
  - la forma de la cola (R5) es razonable;
  - se acordaron con la tutora la separación entre eventos (48 h y 96 h) y el
    número de picos por temporada (3).
- **Decisiones abiertas:**
  - ajuste regional si ξ por píxel es ruido;
  - hasta qué T se confía (con 26 temporadas, unos 10 años).

### F2. Catálogo de estados bajo ciclones (orden de trabajo 1b)

- **Qué hace.** Recorre las 13 599 posiciones de 2000–2025. Para cada posición
  y cada píxel o municipio dentro del área de influencia guarda:
  - P6, R24 y A72 (mm), con su T y R;
  - el viento local V (kt): W06 calibrado a R34, por cuadrante;
  - la distancia, el cuadrante, R34_q, ROCLOUD_q, R_W06_q y R_attr_q;
  - la identificación del ciclón, el tiempo, la Vmax y la RMW.

  No se calcula ningún peligro.
- **Requiere:**
  - agregación píxel → municipio ponderada por fracción de área (Bloque 2a),
    guardando máximo, media y percentiles del municipio;
  - viento local por cuadrante en todos los pasos.
- **Tamaño.**
  - Por municipio: ~3 millones de filas.
  - En malla: hay que restringir a los píxeles terrestres de México
    (~16 000 a 0.1°), unas 4 × 10⁷ filas. Se guarda en Parquet o Zarr
    particionado por año, no en CSV.
- **Terminado cuando** el catálogo reproduce, para Cristóbal 2020 y Odile 2014,
  los campos de `compute_itchi_snapshot_v02` (prueba de consistencia).

### F3. Esquemas sobre el catálogo (orden de trabajo 1c, 3 y 4)

- **Lluvia.** Nivel = máximo sobre las duraciones del nivel de T, con fronteras
  en años. Las fronteras exactas se deciden con F1 y la prueba ordinal.
- **Viento** (niveles escalonados, Saffir-Simpson local). Ejemplo con 4 niveles:

  | Nivel | Viento local |
  |---|---|
  | 0 | < 34 kt |
  | 1 | Tormenta tropical (34–63 kt) |
  | 2 | Categorías 1–2 (64–95 kt) |
  | 3 | Huracán mayor (≥ 96 kt, corte operativo de NHC) |

  La cola por debajo de 34 kt se corta en 17 kt o en 12 m/s; falta justificarlo
  físicamente. El Bloque 1 dio PSS de clase 1 de 0.480 con 17 kt y 0.472 con
  12 m/s.
- **Combinación.** Nivel ITCHI = máximo del nivel de lluvia y el de viento. Es
  más fácil de comunicar que la unión probabilística de v0.2.
- **Número de niveles** (de 2 a 5), con tres criterios:
  1. distinguibilidad: se fusionan niveles contiguos cuyos intervalos se traslapan
     (P(D) con bootstrap por ciclón, o T);
  2. ganancia de información: RPS o información mutua al agregar un nivel;
  3. predictibilidad con AIFS (se evalúa en F5).
- **Validación externa** (solo ordinal):
  - declaratorias asociadas por nombre de tormenta (D5);
  - deslizamientos, con el caso de Manuel 2013 y La Pintada;
  - casos Cristóbal 2020, Odile 2014, Emily 2005, Alex 2010 y Patricia 2015.

### F4. Climatología de ITCHI (orden de trabajo 1d, producto final)

- Por municipio y en malla, para el esquema elegido:
  - número de ciclones con nivel ≥ k (2000–2025);
  - periodo de retorno de cada nivel;
  - mapas de peligrosidad;
  - horas en alerta por evento (D7).
- Produce el artículo de peligrosidad de los ciclones tropicales en México.

### F5. Pronóstico (orden de trabajo 5)

- Percentiles y periodos de retorno de los hindcasts de AIFS 1.1 (2000–2025) por
  anticipación (climatología del propio modelo, P3).
- ITCHI pronosticado por anticipación contra ITCHI observado (F4):
  - con cuánta anticipación se detecta cada nivel;
  - en qué niveles el pronóstico tiene habilidad.

## 4. Decisiones para discutir con la tutora

1. Separación entre eventos y picos por temporada en F1.
2. Fronteras de los niveles de lluvia en años.
3. Corte de la cola de viento (17 kt o 12 m/s) y agrupación de las categorías
   Saffir-Simpson.
4. Agregación municipal: máximo, fracción de área en nivel ≥ k o cuantil alto.
5. Combinación por máximo de niveles en lugar de la unión probabilística.
6. Viento sobre tierra: el «viento equivalente en terreno abierto» (como
   Saffir-Simpson) frente a una reducción por rugosidad o el decaimiento tierra
   adentro.

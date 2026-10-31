# Bloque 1 de ITCHI v0.2.1: P95 como peligro máximo, umbrales de R24 y A72, y corte del viento en 12 m/s

Diseño del experimento, antes de correrlo. Fija las hipótesis, los esquemas, las
métricas y los criterios de decisión, para que la elección no dependa de mirar
los resultados. Implementación: `precipitation.compute_precipitation_class_table_hazard`,
`configs/experiments/b1/`, `scripts/experimento_b1.py`.

---

## 1. Preguntas

1. **Lluvia (factor P).** ¿Qué gana o pierde ITCHI si el P95 de 6 h es el peligro
   máximo (H_P = 1) sin esperar al P99?
2. **Umbrales de R24 y A72.** Con P95 como máximo, ¿cuánto aporta cada umbral de
   persistencia (R24) y de antecedente (A72) a las clases inferiores?
3. **Viento (factor W).** ¿Qué pasa si la cola de viento termina en 12 m/s
   (23.3 kt) en lugar de R_W06 (2 m/s)?
4. **Interacción.** ¿Los efectos de P y W se suman o se compensan?

## 2. Qué dice ya la evidencia de v0.2 (`verification_v02.md`)

- El antecedente A72 cambia P(D) sobre todo con P6 ≥ P95 (tabla 3.2). Con P95 como
  máximo, esa región ya queda saturada, así que **lo que R24 y A72 pueden aportar
  está en P75–P95**. Ahí v0.2 encontró un efecto pequeño: escalar P75–P95 cambió el
  PSS ≥ 3 en +0.006 (D3). La expectativa honesta es que M1 − M0 sea pequeño; el
  experimento lo mide con su intervalo.
- Un P95 por píxel se supera unas 3.4 veces por temporada, contra unas 0.7 para el
  P99 (mediana nacional, todas las causas, no solo ciclones). El sesgo de frecuencia
  del nivel máximo, que ya es 4.7 en v0.2, va a subir.
- La cola cortada en 24 kt (≈ 12 m/s) dio AUC 0.833 y PSS ≥ 1 de 0.460; el corte en
  17 kt dio 0.848 y 0.490 (sección 4, evaluación aproximada). Por eso 17 kt se
  incluye como referencia.

## 3. Formulación de la lluvia: tabla de clases P6 × R24 × A72

Cada variable se compara con **sus propios percentiles locales** (6, 24 y 72 h;
A72 usa la climatología de 72 h porque A72(T) = R72(T − 6 h)) mediante una
coordenada de clase:

```text
u = 0 en 0 mm,  u = 1 en P75,  u = 2 en P90,  u = 3 en P95 (y por encima)
```

lineal entre nodos. El peligro es una tabla de valores en los nodos enteros,
interpolada de forma multilineal en (u_P6, u_R24, u_A72). Si la tabla no decrece
en ningún eje, H_P no decrece con P6, con R24 ni con A72, y es continuo salvo en la
compuerta de P75 (M2, M3). La tabla se escribe con una rampa base de P6 y reglas
«si R24 ≥ Pxx y A72 ≥ Pyy, la rampa de P6 pasa a ser ...».

La matriz v0.2 es un caso particular de esta forma: `PRECIP_TABLE_V02` la
reproduce a 1e-12 (`tests/test_class_table.py`). Así, v0.2 y los esquemas nuevos se
comparan dentro de la misma familia.

### Esquemas candidatos (en todos, P95 = máximo y P75 = inicio de alerta)

| Esquema | Regla | Qué mide |
|---|---|---|
| **M0** | P75 → 0, P90 → 0.5, P95 → 1 (solo P6) | Referencia sin R24 ni A72 |
| **M1** | M0; con R24 ≥ P95 y A72 ≥ P75, P90 ya es el máximo (P75–P90 sube de 0 a 1) | La regla de v0.2 recorrida una clase: «considerar su uso desde antes» |
| M1[...] | Rejilla: R24 ≥ P90 o P95 × A72 ≥ P75, P90 o P95; y solo R24 o solo A72 | **Impacto de cada umbral** (sensibilidad, no selección) |
| **M2** | Máximo de las tres duraciones, cada una con su escalera (P90 → 0.5, P95 → 1), solo con P6 ≥ P75 | Las tres duraciones como fuentes independientes de peligro |
| **M3** | Tabla 4 × 4 × 4 calibrada: cada celda recibe el peligro de la clase de solo 6 h con la misma P(D); nunca por debajo de M0; validación cruzada dejando un año fuera | Lo que los datos asignan a cada combinación de umbrales |

Las celdas de M3 se suavizan hacia P(D | clase de P6) con 50 pseudoconteos, y la
tabla se fuerza a ser monótona con una envolvente (máximo sobre las celdas
dominadas). La envolvente sesga hacia arriba las celdas con pocos pares; por eso
M3 solo se adopta si supera a M1 en validación cruzada.

**Compuerta.** Ningún esquema alerta por R24 o A72 si P6 < P75. En M2 y M3 eso
produce un salto en P75: debajo, H_P = 0; arriba, el valor de la celda.

## 4. Viento

W06 (PA21) en la distancia del municipio. Dentro de 34 kt,
H_W = h34 + (1 − h34)(V − 34)/(Vmax − 34); fuera, la cola es
h34 (V − V_out)/(34 − V_out), con h34 = 0.2.

| Variante | V_out |
|---|---|
| W0 (v0.2) | 2 m/s = V(R_W06) |
| **W1** | 12 m/s = 23.3 kt |
| referencia | 17 kt = 8.7 m/s |
| sin viento | H_W = 0 |

En el pipeline: `wind.tail_end_wind_ms: 12.0` (`configs/experiments/b1/p0_w1.yaml`).

**Pendiente: justificación física de 12 m/s.** 12 m/s está en el intervalo de
Beaufort 6 (10.8–13.8 m/s). Es viento sostenido de 1 min del perfil; si el argumento
es de daño, hace falta convertir a ráfaga, que es lo que suele relacionarse con
daño.

## 5. Diseño

**Unidad.** Par municipio–ciclón (`negativos_w06.py`): 381 929 pares, 2 % con
declaratoria. ITCHI = 1 − (1 − H_W)(1 − H_P), en clases SIAT de 0.2.

**Modo de evaluación.** Con `pasos_municipio_ct_w06.csv.gz`
(`negativos_w06.py --guardar-pasos`), el índice se calcula en cada paso y el par
toma su máximo. Sin ese archivo, se evalúa en el pico de P6. El modo pico subestima
los esquemas cuyo máximo cae en otro paso, como la escalada por R24. **Se reporta el
modo pasos.**

**Diseño 2 × 2.** Lluvia {v0.2, M1} × viento {W0, W1}. Las celdas son
`configs/experiments/b1/p{0,1}_w{0,1}.yaml`. El script calcula además todos los
esquemas por todas las variantes de viento.

**Métricas.**
- AUC promedio por año. La tasa base cambia de 0.2 a 9 % entre años.
- PSS por clase k = 1…5.
- En las clases 3 y 5: POD, POFD, FAR, CSI, SEDI (Ferro y Stephenson, 2011) y sesgo.
- ΔPSS contra v0.2 + W0, con intervalo de 90 % por bootstrap de ciclones (los pares
  de un mismo ciclón no son independientes).
- **ΔPOD a igual POFD** contra la curva ROC de la referencia. Si es ≈ 0, el esquema
  solo desplaza el umbral sobre la misma curva: alerta más, pero no distingue mejor.

**Evidencia descriptiva.** P(D) y n por celda de clases P6 × R24 × A72
(`celdas_P6_R24_A72.csv`). Es la respuesta directa a «qué impacto tienen los
umbrales de R24 y A72».

## 6. Criterios de decisión (fijados antes de ver los resultados)

1. **P95 como máximo** (M0 contra v0.2). Se adopta si el intervalo de ΔPSS ≥ 5
   excluye 0 **y** ΔPOD@POFD ≥ 0. Si ΔPOD@POFD ≈ 0, el cambio no se justifica por
   discriminación: se presenta como decisión operativa (alertar más, por el
   subregistro de las declaratorias), con su costo en sesgo y FAR.
2. **Umbrales de R24 y A72.** Se adopta la regla más simple (M1, luego M2, luego M3)
   cuyo ΔPSS frente a M0 tenga un intervalo que excluya 0. La rejilla de M1 se
   reporta como sensibilidad; no se elige el mejor punto de la rejilla con los mismos
   pares.
3. **Corte en 12 m/s** (W1 contra W0). Se adopta si recupera la clase 1 (intervalo de
   ΔPSS ≥ 1 > 0) sin perder la información del viento (AUC con viento > AUC sin
   viento). Se compara con 17 kt: si 17 kt domina, la elección de 12 m/s necesita un
   argumento físico.
4. **Interacción.** Se compara (P1W1 − P0W1) con (P1W0 − P0W0). Si difieren más que el
   intervalo, P y W no se pueden decidir por separado.

## 7. Limitaciones

- El viento por par es aproximado: perfil sin calibrar a R34 y sin cuadrantes. La
  confirmación con el pipeline completo va en los casos de estudio, con
  `compute_itchi_snapshot_v02` y los overlays del 2 × 2.
- La lluvia se mide en el píxel del máximo de 6 h del municipio. R24 y A72 son los
  de ese píxel, no su propio máximo municipal.
- La asociación declaratoria–ciclón es por fechas (D5 pendiente); 29.8 % de los pares
  declarados se excluyen.
- Las declaratorias subregistran impactos. Parte de las falsas alarmas son impactos
  no declarados, así que la FAR es una cota superior.

## 8. Reproducción y figuras

```bash
cd scripts
python negativos_w06.py --guardar-pasos     # una vez: pares y pasos
python experimento_b1.py                    # tablas en outputs/diagnostics/experimento_b1/
python -m pytest ../tests/test_class_table.py ../tests/test_experimento_b1.py -v
```

Figuras e interpretación: `notebooks/third test/09_bloque1_experimentos.ipynb`
(funciones en `scripts/figuras_b1.py`; PNG en `outputs/figures/experimento_b1/`).

| Figura | Muestra | Experimento |
|---|---|---|
| A1 | H_P contra P6 en cuatro situaciones de R24 y A72 | E1, E2 |
| A2 | Clase de H_P en el plano P6 × R24 para A72 = 0, P75, P95 | E2 |
| A3 | Dónde termina la cola de viento (R_W06, 12 m/s, 17 kt) | E3 |
| B1 | P(D) y n por celda P6 × R24 × A72 | E2 (evidencia) |
| B2 | ΔPSS contra v0.2 y contra M0; sesgo y POFD | E1, E2 |
| B3 | Curvas ROC con los puntos de cada clase | E1, E2 |
| B4 | Rejilla de umbrales de M1 y ablación | E2 |
| B5 | Diseño 2 × 2 (PSS por clase y AUC) | E3, E4 |
| B6 | Porcentaje de pares por clase, sin y con declaratoria | E1–E4 |
| B7 | Tabla de decisiones con los criterios de la sección 6 | todos |

En la aproximación por pares, la región directa del viento es V ≥ 34 kt o
distancia ≤ Rmax, para que el ojo no quede como cola. Por eso los números de W0
pueden diferir un poco de la sección 4 de `verification_v02.md`.

En malla (casos de estudio):

```python
from itchi.config import itchi_v02_settings_from_config, load_experiment_config
settings = itchi_v02_settings_from_config(load_experiment_config("configs/experiments/b1/p1_w1.yaml"))
result = compute_itchi_snapshot_v02(p6, r24, a72, ..., **settings)
```

## 9. Resultados (corrida del 30 de octubre de 2026)

Modo pasos, 381 929 pares, 7 462 con declaratoria (2.0 %), 1000 réplicas de
bootstrap por ciclón. Tablas completas en
`outputs/diagnostics/experimento_b1/resumen_b1.md`.

**E1. P95 como máximo: no se adopta.**

| Solo lluvia | v0.2 | M0 (P95 máximo) |
|---|--:|--:|
| AUC por año | 0.869 | 0.842 |
| PSS clase ≥ 5 | 0.553 | 0.546 (Δ −0.007 [−0.023, +0.008]) |
| ΔPOD a igual POFD | — | −0.004 |
| Sesgo clase 5 | 6.4 | 10.3 (+61 %) |
| PSS clase ≥ 3 | 0.549 | 0.521 |

M0 se mueve sobre la misma curva ROC: alerta más sin distinguir mejor, y
pierde AUC porque iguala todo lo que está por encima del P95.

**E2. Por debajo de P95, R24 y A72 casi no aportan.** Contra M0, el mejor es
M1 con +0.006 [+0.001, +0.011]. Todo el aporte viene de R24 ≥ P95, porque la
variante solo con R24 da lo mismo. A72 ≥ P75 solo empeora el resultado (−0.008),
y M2 y M3 no aportan. En cambio, por encima de P95 sí importan: con P6 ≥ P95 y
R24 ≥ P95, P(D) pasa de 5.7 % (A72 < P75) a 10.0, 13.9 y 21.8 % (A72 ≥ P95).
Un P6 ≥ P95 sin R24 ≥ P95 tiene P(D) ≈ 1–3 %, igual que P90–P95.

**E3. El corte de la cola funciona; 17 kt rinde algo más que 12 m/s.**

| Viento (lluvia v0.2) | Pares con H_W > 0 | PSS clase ≥ 1 | AUC |
|---|--:|--:|--:|
| Sin viento | 0 % | 0.445 | 0.869 |
| Cola hasta R_W06 | 100 % | 0.000 | 0.897 |
| 12 m/s | 9.1 % | 0.472 [+0.408, +0.526] | 0.887 |
| 17 kt | 18.5 % | 0.480 [+0.456, +0.505] | 0.895 |

**E4. Los efectos son aditivos.** v0.2 → M1 cuesta −0.009 en PSS clase 5 con
cualquier viento.

**Qué queda para v0.3.** Las declaratorias son un registro ruidoso y muy general
de los impactos. Por eso estos resultados se toman como evidencia ordinal, no
como calibración final. En v0.3 la lluvia se define por periodo de retorno y el
viento por umbrales absolutos (Saffir-Simpson); ver `docs/plan_v0.3.md`.

## Referencias

- Ferro, C. A. T. y Stephenson, D. B. (2011). Extremal Dependence Indices: improved
  verification measures for deterministic forecasts of rare binary events.
  *Weather and Forecasting*, 26(5), 699–713. https://doi.org/10.1175/WAF-D-10-05030.1.
  SEDI y el comportamiento de los puntajes con eventos raros.

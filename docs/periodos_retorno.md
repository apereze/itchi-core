# Climatología de rareza de la lluvia: periodos de retorno (paso 1)

ITCHI mide el peligro de un ciclón por lo **raro** que es lo que produce en
cada lugar. Este paso calcula, por píxel de MSWEP V3 (0.1°, mayo–noviembre,
2000–2025, todas las causas de lluvia) y por duración (6, 24, 72 h), cada
cuánto se alcanza una cantidad de lluvia.

## Definiciones

- R_d(T) es la lluvia en (T − d, T], suma de acumulados de 6 h dentro de cada
  temporada.
- **Periodo de retorno T (años)**, con máximos estacionales + GEV:
  P(máximo de la temporada > x) = 1/T. Es confiable para T ≥ 2 años. Con 26
  temporadas, la estimación empírica alcanza unos 10 años; más allá es
  extrapolación.
- **Intervalo de recurrencia R (años)**, con picos independientes + GPD (POT):
  R(x) = 1/λ(x), donde λ es el número medio de eventos por temporada con pico
  ≥ x.
  - Eventos: picos separados al menos 48 h (6 y 24 h) o 96 h (72 h), y ≥ 1 mm.
  - Umbral: el pico número K + 1 del registro, con K = 3 × 26.
  - Sirve también por debajo de un año.
- **Relación:** T = 1 / (1 − exp(−1/R)), si los eventos son de Poisson.
- **Estimadores:** L-momentos (Hosking, 1990; Hosking y Wallis, 1987), con la
  forma ξ acotada en [−0.5, 0.5].

## Criterios fijados antes de correr con MSWEP

Se fijan aquí, antes de ver los resultados. Las estadísticas se calculan sobre
los píxeles con centro dentro de los polígonos municipales (`mun23gw`), no sobre
la caja de México, que incluye mar y países vecinos. Las de la GPD excluyen los
píxeles sin umbral en la cola (`gpd_full_threshold = 0`).

| Control | Criterio | Si no se cumple |
|---|---|---|
| Umbral GPD en la cola | `gpd_full_threshold = 1` en ≥ 95 % de los píxeles de México, por duración | El umbral es el piso de 1 mm en zonas áridas: la GPD se ajusta a todos los eventos húmedos. Bajar K o usar solo GEV ahí |
| R4, GEV frente a POT en T = 10 años | Mediana de \|POT − GEV\| / GEV ≤ 5 % y percentil 90 ≤ 12 %, por duración | Revisar la separación entre eventos (picos dependientes inflan la tasa y aplanan la cola) y la forma (R5) en las zonas que se separan |
| R5, forma de la cola | Si la desviación estándar espacial de ξ es ≤ √2 veces la de muestreo (GEV 0.17, GPD 0.13), el mapa de ξ es ruido | ξ regional (L-momentos regionales) o común por región climática |
| Confianza en T | Se reportan T ≤ 10 años como estimados; 25, 50 y 100 años como extrapolación | — |

**De dónde salen los números.** Monte Carlo con el modelo exacto (eventos de
Poisson con excesos GPD, ξ = 0 y 0.15, 6 y 15 eventos por temporada, y también
sobredispersión entre temporadas), 26 temporadas y K = 3 por temporada: la
diferencia relativa entre GEV y POT en T = 10 años tiene mediana ≈ 2.5 % y
percentil 90 ≈ 6 %, sin sesgo; el criterio deja el doble. Los dos estimadores no
son independientes: comparten los eventos mayores. Que coincidan es una prueba
de consistencia (declustering, conversión T–R), no una validación independiente.
La desviación estándar de ξ por píxel debida solo al muestreo es ≈ 0.17 (GEV) y
≈ 0.13 (GPD). Con varianza espacial real menor que la de muestreo, un ξ común
reduce el error cuadrático medio; de ahí el factor √2 (σ_obs² = σ_real² +
σ_muestreo²). Advertencia: los píxeles vecinos comparten tormentas, así que el
error de muestreo está correlacionado en el espacio y un mapa suave no prueba que
ξ tenga señal.

## Rareza real de los percentiles de v0.2 (R3)

En la misma lectura se cuenta, por píxel y duración, el número de eventos
independientes por temporada con pico ≥ Pq (`p_wet_raw`, q = 0.75, 0.90, 0.95,
0.99). Es un conteo directo, no un ajuste, así que resuelve también los
percentiles que caen por debajo del umbral de la GPD (unos 3 eventos por
temporada), que con la GPD solo darían «< 0.33 años».

## Archivos

| Archivo | Contenido |
|---|---|
| `src/itchi/return_periods.py` | Funciones NumPy: sumas móviles, picos independientes, GEV, GPD, conversiones |
| `scripts/periodos_retorno.py` | Lee una vez los 26 años y escribe `data/processed/mswep_v3_return_periods_may_nov.nc` |
| `scripts/figuras_retorno.py` | Figuras R1–R5 y tabla resumen |
| `notebooks/third test/10_periodos_retorno.ipynb` | Figuras con la guía de lectura |

```bat
conda activate itchi
cd scripts
python periodos_retorno.py                        :: ~30 min, ~2-3 GB de RAM
python periodos_retorno.py --filas-por-banda 10   :: si falta memoria
```

## Uso posterior

Cualquier lluvia x de un píxel se convierte en su rareza con:

```python
from itchi.return_periods import gev_return_period, pot_recurrence_interval
T = gev_return_period(x, gev_xi, gev_sigma, gev_mu)            # años, T >= 1
R = pot_recurrence_interval(x, gpd_u, gpd_sigma, gpd_xi, gpd_rate)   # NaN si x < gpd_u
```

El paso 2 (catálogo de estados bajo los ciclones) guarda P6, R24, A72 y su
rareza para definir los niveles de peligro en periodo de retorno.

## Limitaciones

- Los picos en el primer o último paso de la temporada pueden ser eventos truncados
  (la lluvia siguió fuera de mayo–noviembre). Afecta a pocos eventos y casi nunca a
  los mayores.
- La conversión T = 1 / (1 − exp(−1/R)) supone eventos de Poisson. Con
  sobredispersión entre temporadas (años con más ciclones, ENSO) la conversión se
  desvía para R cercano a 1 año; en T = 10 años el efecto es despreciable (Monte
  Carlo).
- 26 temporadas: los niveles de T ≥ 25 años son extrapolaciones. Si R5 muestra
  ruido, conviene un ajuste regional (L-momentos regionales) o acotar la forma.
- Los picos se separan con una ventana móvil, así que un pico menor que queda a
  menos de la separación de otro mayor no cuenta como evento. Es conservador:
  subestima ligeramente el número de eventos.
- MSWEP subestima los extremos más localizados por su resolución de 0.1°; los
  niveles son de área (~11 km), no de pluviómetro.

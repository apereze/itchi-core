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

- 26 temporadas: los niveles de T ≥ 25 años son extrapolaciones. Si R5 muestra
  ruido, conviene un ajuste regional (L-momentos regionales) o acotar la forma.
- Los picos se separan con una ventana móvil, así que un pico menor que queda a
  menos de la separación de otro mayor no cuenta como evento. Es conservador:
  subestima ligeramente el número de eventos.
- MSWEP subestima los extremos más localizados por su resolución de 0.1°; los
  niveles son de área (~11 km), no de pluviómetro.

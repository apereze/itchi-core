# Declaratorias asociadas a CT vs MSWEP V3 de 6 h

Script: `scripts/declaratorias_6h.py`

## Datos de entrada

`data/external/` no se versiona (`.gitignore`). Estructura esperada:

```text
data/external/
├── declaratorias/
│   ├── declaratorias27052025.csv     # CENAPRED, latin-1, corte 27-05-2025
│   └── inafed_bd_1748479609.csv      # catálogo INAFED (solo control de claves)
├── mun23gw/
│   └── mun23gw.{shp,shx,dbf,prj,xml} # INEGI/CONABIO 2023, EPSG:4326
└── Rocloud/
    ├── ROCLOUD_NA.csv                # mismas tablas que dfna/dfep del artículo
    └── ROCLOUD_EP.csv
data/processed/mswep_v3_6h_percentiles_may_nov.nc   # scripts/percentiles_6h.py
D:/MSWEP_V316_6H/mswep_v3_6h_YYYY.nc                # scripts/mswep_v3_6h_accum.py
```

Las rutas son atributos de `Config` y pueden cambiarse.

## Método

1. Declaratorias `Hidrometeorológico` cuyas `Observaciones` mencionan ciclón tropical,
   huracán, tormenta tropical o depresión tropical (criterio del artículo).
2. `[Fecha Inicio, Fecha Fin]` se expande a 00/06/12/18 UTC.
3. Se conservan los pasos con al menos un CT activo en ROCLOUD a la misma hora; si hay
   varios se guarda el más cercano al municipio. Opcional: `dias_antes`, `dias_despues`
   (tolerancia de fechas), `max_dist_km`, `require_inside_rocloud`, `solo_lluvia`.
4. Para cada paso T se toma P(T) = acumulado en (T-6h, T] sobre los píxeles cuyo centro
   está dentro del municipio (si no hay ninguno, el píxel más cercano; `n_pix = 0`).
5. `P_evento` = lluvia de 6 h con la que se emitió la declaratoria (máximo sobre píxeles
   y pasos). `P_min` (municipio) = menor `P_evento` entre sus declaratorias.
   `P_min_decl` (píxel) = mínimo, entre declaratorias, del máximo de 6 h en ese píxel.
6. Distancia en mm a los percentiles locales del mismo píxel: `dif_Pxx = P_evento − Pxx`
   (eventos), `dif_min_Pxx` (municipios), `P_min_menos_Pxx` (píxel). Negativo = la
   declaratoria se emitió con lluvia menor que ese percentil. Se usa `p_wet_raw`
   (`p_wet` deja P99.9 en NaN en todo México por el filtro de suficiencia). El archivo
   actual tiene P75, P90, P95, P99 y P99.9; si se recalcula con P70, el script lo toma solo.

## Uso

```bash
conda activate itchi
cd scripts
python declaratorias_6h.py
```

Exploración y figuras: `notebooks/declaratorias_6h_exploracion.ipynb` (ejecuta el script si
no existen salidas, resume, mapea y compara variantes de criterios).

Desde un notebook propio:

```python
import sys; sys.path.insert(0, "../scripts")
from declaratorias_6h import Config, run

out = run(Config(solo_lluvia=True))          # o Config() para replicar el artículo
mun = out["municipios"]; ev = out["eventos"]; pix = out["pixel"]
```

## Salidas (`outputs/diagnostics/declaratorias_6h/`)

| Archivo | Unidad | Contenido |
|---|---|---|
| `declaratorias_ct_pasos.csv` | declaratoria × T | CT más cercano, distancia, P máx/media, fracción de área ≥ cada percentil |
| `declaratorias_ct_eventos.csv` | declaratoria | `P_evento`, paso y CT del pico, `Pxx_pixel_max`, `dif_Pxx` |
| `declaratorias_ct_municipios.csv/.gpkg` | municipio | `P_min`, P10, mediana, máx., `Pxx_pixel_min`, `dif_min_Pxx` |
| `declaratorias_ct_pixel_min.nc` | píxel | `P_min_decl`, `n_decl`, `P_min_menos_Pxx`, `P_min_sobre_Pxx` |
| `qc_claves_sin_geometria.csv` | clave | claves de declaratorias sin polígono en `mun23gw` |

## Limitaciones

- Solo coincidencia temporal (como en el artículo): en temporada activa casi todo paso
  tiene algún CT en NA o EP, aunque esté a > 1000 km. Revisar `dist_km` / `dentro_rocloud`.
- Declaratorias por viento o marea dan mínimos cercanos a 0 (`menciona_lluvia = False`).
- Las fechas de la declaratoria pueden no cubrir el impacto: Delta 2020 en Cozumel da 0.8 mm
  con 5–6 oct y 101 mm (>= P99.9) con `dias_despues=1` (entrada a tierra el 7 oct).
- Emergencia y desastre del mismo evento cuentan dos veces en `n_decl`; el mínimo no cambia.
- El pico de 6 h no representa inundaciones fluviales ni deslizamientos por lluvia acumulada.

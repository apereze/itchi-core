# Development log

## Checkpoint

### Estado actual

Se configuró la estructura base del repositorio `itchi-core` y se implementaron los módulos iniciales del núcleo ITCHI:

- `constants.py`
- `config.py`
- `precipitation.py`
- `masks.py`
- `components.py`
- `index.py`
- `geometry.py`
- `units.py`
- `pipeline.py`

También se configuraron:

- `README.md`
- `.gitignore`
- `environment.yml`
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `docs/architecture.md`

### Funcionalidad implementada

El repositorio ya puede:

1. calcular peligro normalizado por precipitación `H_P`;
2. construir máscaras directa, indirecta y exterior;
3. calcular componentes `H_Pdir`, `H_Pind`, `H_W`, `H_dir`, `H_ind`;
4. calcular el índice final `ITCHI`;
5. calcular distancia radial y cuadrantes relativos;
6. convertir radios de millas náuticas a kilómetros;
7. ejecutar una integración por snapshot con geometría.

### Convenciones fijadas

- Cuadrantes internos: `RNE`, `RSE`, `RSW`, `RNW`.
- Distancias y radios en kilómetros.
- `R34` de IBTrACS puede entrar en millas náuticas y se convierte internamente.
- `ROCLOUD` se asume en kilómetros salvo que se indique otra unidad.
- La precipitación se maneja como snapshot, no como acumulado temporal.
- ITCHI debe permanecer en `[0, 1]`.

### Pendiente inmediato

Continuar con la validación de `pipeline.py`, especialmente la función:

```python
compute_itchi_snapshot_from_grid(...)
````

Después, el siguiente módulo sugerido es:

```text
aggregation.py
```

para calcular:

* `ITCHI_max`
* `ITCHI_acc`

### Comandos útiles para retomar

```bash
conda activate itchi
git pull
python -m pytest tests/
pre-commit run --all-files
```

---

## 2026-10-26 — Cierre de ITCHI v0.2

- Lluvia: matriz P6 × (R24, A72). Regla «P6 en P95–P99 con R24 ≥ P95 y A72 ≥ P75 → máximo»,
  forma continua en `compute_precipitation_matrix_hazard`. PSS del nivel máximo 0.413 → 0.514.
- Viento: W06 calibrado a R34 con cola hasta R_W06 (`wind_model="w06_tail"`, h34 = 0.2).
  La cola satura la clase 1 (96 % de los pares con viento); se propone cortarla en 17 kt (D1).
- Verificación: 381 929 pares municipio–ciclón (`scripts/negativos_w06.py`); resultados en
  `docs/verification_v02.md`.
- Casos Cristóbal 2020 y Odile 2014: notebooks 05–08 (`notebooks/second test/`).
- Convención actualizada: la precipitación entra como acumulado de 6 h que termina en el
  tiempo sinóptico; R24 y A72 solo escalan H_P.
- Pendiente para v0.3: decisiones D1–D9 de `docs/verification_v02.md`.


## 2026-10-28 — Inicio de 0.2.1: consistencia y punto de entrada v0.2

- `compute_itchi_snapshot_v02` (pipeline): P6, R24, A72 y climatologías de 6/24/72 h → ITCHI v0.2
  en una sola llamada. Sus valores por defecto son los de `configs/default.yaml`; un test lo verifica.
- Radio de atribución: ROCLOUD con piso R_W06 por cuadrante (faltante, 0 o menor → R_W06).
  R_W06 queda solo como fin de la cola de viento.
- Configuración: convención temporal v0.2 (acumulado de 6 h que termina en T), `p_wet_raw`,
  modos de atribución documentados, bloque `r_w06` (PA21, 2 m/s).
- Gancho para experimentos: `wind_tail_end_kt` (corte de la cola, D1).
- Pendiente: correr `pytest` local y en CI; luego los experimentos de 0.2.1.

## 2026-10-30 — Cierre de ITCHI v0.2.1

- Versión 0.2.1. Incluye el punto de entrada `compute_itchi_snapshot_v02`, la regla
  ROCLOUD con piso R_W06, la configuración consistente y la tabla de clases P6 × R24 × A72.
- Bloque 1: P95 como máximo no se adopta; R24/A72 solo importan por encima de P95; cortar
  la cola de viento recupera la clase 1 (17 kt rinde algo más que 12 m/s). Resultados en
  `docs/experimento_b1.md`, sección 9.
- Las declaratorias quedan como prueba ordinal. Siguiente línea de trabajo: v0.3
  (climatología de estados, periodos de retorno, viento con umbrales absolutos);
  ver `docs/plan_v0.3.md`.

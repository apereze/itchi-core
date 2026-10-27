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


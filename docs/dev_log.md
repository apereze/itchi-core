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

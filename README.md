# ITCHI Core

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-itchi--core-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](tests/)
[![License: TBD](https://img.shields.io/badge/License-TBD-yellow.svg)](#licencia)

**Integrated Tropical Cyclone Hazard Index**

Repositorio para el desarrollo del núcleo computacional de **ITCHI v0.1**, un índice físico de peligro asociado a ciclones tropicales.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Descripción general](#descripción-general)
- [Alcance del repositorio](#alcance-del-repositorio)
- [Principio metodológico](#principio-metodológico)
- [Flujo computacional](#flujo-computacional)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Entradas esperadas](#entradas-esperadas)
- [Salidas esperadas](#salidas-esperadas)
- [Estado actual del proyecto](#estado-actual-del-proyecto)
- [Instalación](#instalación)
- [Ejemplos reproducibles](#ejemplos-reproducibles)
- [Pruebas y control de calidad](#pruebas-y-control-de-calidad)
- [Data Requirements](#data-requirements)
- [Contributing](#contributing)
- [Autoría](#autoría)
- [Licencia](#licencia)
- [References](#references)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

# Create and activate environment
conda env create -f environment.yml
conda activate itchi

# Install package in development mode
pip install -e .

# Verify installation
python -c "import itchi; print('ITCHI installed successfully')"

# Run tests
python -m pytest tests/

# Run synthetic smoke test
python examples/smoke_test_synthetic.py
```

Optional diagnostic figure:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png
```

The `outputs/` directory is ignored by Git.

---

## Descripción general

**ITCHI** significa **Integrated Tropical Cyclone Hazard Index**.

La versión inicial, **ITCHI v0.1**, se plantea como un índice físico continuo entre `0` y `1`, diseñado para representar el peligro asociado a ciclones tropicales en una malla espacial.

El índice integra tres fuentes principales de peligro:

1. **Precipitación directa**, asociada a lluvia dentro de la región directa del ciclón.
2. **Viento**, estimado a partir de un campo normalizado de peligro por viento.
3. **Precipitación indirecta**, asociada a bandas externas dentro de la región atribuible al ciclón.

El objetivo de este repositorio es construir de forma organizada, trazable, modular y reproducible el cálculo del índice ITCHI.

---

## Alcance del repositorio

Este repositorio contiene únicamente los elementos necesarios para la **creación del índice ITCHI**.

### Incluye

- Lectura y estandarización de datos de trayectoria ciclónica.
- Lectura y limpieza de radios de viento, como `R34`.
- Resolución del radio directo efectivo `R_direct`.
- Manejo de depresiones tropicales sin `R34` mediante `RMW`.
- Lectura y limpieza de radios estructurales externos, como `ROCLOUD`.
- Lectura/escritura de productos científicos con `xarray`.
- Cálculo de peligro normalizado por precipitación.
- Cálculo o uso de peligro normalizado por viento.
- Construcción de máscaras espaciales:
  - región directa;
  - región indirecta;
  - región exterior.
- Cálculo de componentes físicos del índice.
- Cálculo final de `ITCHI`.
- Control de calidad físico/computacional.
- Compilación de múltiples snapshots por evento.
- Productos derivados por evento: `ITCHI_max` e `ITCHI_acc`.
- Notebooks y scripts sintéticos de validación.

### No incluye por ahora

- Validación con declaratorias de emergencia o desastre.
- Modelos de aprendizaje automático.
- Calibración estadística avanzada.
- Visualizaciones finales para publicación.
- Dashboard o aplicación web.
- Manuscrito científico completo.

Estos elementos podrán desarrollarse posteriormente en otros módulos o repositorios.

---

## Principio metodológico

ITCHI v0.1 distingue tres regiones alrededor del ciclón tropical:

| Región | Condición conceptual | Interpretación |
|---|---|---|
| Región directa | `r <= R_direct_q` | Zona de peligro directo por viento y precipitación |
| Región indirecta | `R_direct_q < r <= ROCLOUD_q` | Zona de precipitación externa atribuida al ciclón |
| Región exterior | `r > ROCLOUD_q` | Zona sin contribución al índice |

La separación conceptual es:

```text
Peligro directo = viento + precipitación dentro de R_direct
```

```text
Peligro indirecto = precipitación entre R_direct y ROCLOUD
```

La región exterior no contribuye al índice.

### `R34` frente a `R_direct`

El repositorio distingue explícitamente entre:

| Variable | Significado |
|---|---|
| `R34` | Radio observado de vientos de 34 kt |
| `R_direct` | Radio efectivo usado para definir la región directa |
| `RMW` | Radio de máximo viento |
| `ROCLOUD` | Radio externo de atribución nubosa/precipitante |

Reglas principales:

```text
Si R34 existe:
    R_direct_q = R34_q

Si R34 es parcial:
    faltantes de R34 se rellenan con el promedio de cuadrantes disponibles

Si no existe R34 y Vmax < 34 kt:
    R_direct_q = RMW

Si no existe R34 ni RMW:
    se usa fallback_direct_radius_km, si está configurado
```

Esto evita asignar artificialmente `R34` a depresiones tropicales que físicamente no alcanzan vientos de 34 kt.

---

## Tratamiento de la precipitación

En esta versión, la precipitación se maneja como **snapshot**, no como acumulado temporal.

Esto significa que el campo de precipitación representa el estado o intensidad de la precipitación en un tiempo determinado. Por tanto, ITCHI no debe asumir que la precipitación se acumula de 3 h a 6 h.

Esta decisión permite compatibilidad con:

- MSWEP histórico;
- productos de pronóstico;
- hindcasts;
- productos downscalados;
- productos corregidos por sesgo.

---

## Resolución temporal del índice

El índice se calcula en tiempos sinópticos asociados al ciclón:

```text
00, 06, 12 y 18 UTC
```

Esto mantiene coherencia temporal con:

- trayectoria del ciclón;
- intensidad;
- radios de viento;
- `R34`;
- `RMW`;
- `ROCLOUD`;
- perfil radial de viento;
- precipitación snapshot.

---

## Normalización de precipitación

El peligro por precipitación se calcula con percentiles climatológicos locales:

| Percentil | Interpretación |
|---|---|
| `P90` | Inicio de peligro bajo u ocasional |
| `P95` | Peligro alto |
| `P99` | Peligro máximo |

La normalización debe calcularse por:

- celda espacial;
- mes;
- hora sinóptica o tiempo válido comparable.

La función por tramos es:

```text
P < Q90        → H_P = 0
Q90 ≤ P < Q95 → H_P aumenta de 0 a 0.5
Q95 ≤ P < Q99 → H_P aumenta de 0.5 a 1
P ≥ Q99       → H_P = 1
```

---

## Componentes del índice

| Componente | Descripción |
|---|---|
| `H_P` | Peligro normalizado por precipitación |
| `H_W` | Peligro normalizado por viento |
| `H_Pdir` | Precipitación directa dentro de `R_direct` |
| `H_Pind` | Precipitación indirecta entre `R_direct` y `ROCLOUD` |
| `H_dir` | Componente directo |
| `H_ind` | Componente indirecto |
| `ITCHI` | Índice integrado final |

---

## Fórmula conceptual

Componente directo:

```text
H_dir = 1 - (1 - H_W)^alpha * (1 - H_Pdir)^beta
```

Componente indirecto:

```text
H_ind = H_Pind
```

Índice final:

```text
ITCHI = 1 - (1 - H_dir)^lambda_direct * (1 - H_ind)^mu_indirect
```

En la versión base:

```text
alpha = beta = lambda_direct = mu_indirect = 1.0
```

---

## Flujo computacional

```text
Track del ciclón
    center_lon, center_lat, vmax_kt, rmw_km, R34
        ↓
ROCLOUD por cuadrante
        ↓
radii.py
    resolución de R_direct_q y ROCLOUD_q
        ↓
geometry.py
    radius_km, quadrant
        ↓
precipitation.py
    H_P desde precipitación snapshot y Q90/Q95/Q99
        ↓
wind.py
    V* proporcionado o calculado con Vmax/RMW
        ↓
masks.py
    M_direct, M_indirect, M_exterior
        ↓
components.py
    H_Pdir, H_Pind, H_W, H_dir, H_ind
        ↓
index.py
    ITCHI_g,h,t
        ↓
quality_control.py
    validaciones físicas y estructurales
        ↓
compiler.py + aggregation.py
    snapshots, metadata, ITCHI_max, ITCHI_acc
```

---

## Estructura del repositorio

```text
itchi-core/
│
├── README.md
├── environment.yml
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── docs/
│   └── architecture.md
│
├── examples/
│   ├── README.md
│   └── smoke_test_synthetic.py
│
├── notebooks/
│   └── 00_smoke_test_synthetic.ipynb
│
├── src/
│   └── itchi/
│       ├── __init__.py
│       ├── constants.py
│       ├── config.py
│       ├── units.py
│       ├── radii.py
│       ├── geometry.py
│       ├── precipitation.py
│       ├── masks.py
│       ├── components.py
│       ├── index.py
│       ├── aggregation.py
│       ├── quality_control.py
│       ├── io.py
│       ├── tracks.py
│       ├── rocloud.py
│       ├── wind.py
│       ├── pipeline.py
│       └── compiler.py
│
└── tests/
    ├── test_aggregation.py
    ├── test_components.py
    ├── test_compiler.py
    ├── test_geometry.py
    ├── test_index.py
    ├── test_io.py
    ├── test_masks.py
    ├── test_pipeline.py
    ├── test_precipitation.py
    ├── test_quality_control.py
    ├── test_radii.py
    ├── test_rocloud.py
    ├── test_tracks.py
    ├── test_units.py
    └── test_wind.py
```

---

## Entradas esperadas

Las entradas mínimas para calcular ITCHI son:

1. Datos de trayectoria del ciclón.
2. Centro del ciclón por tiempo: `center_lon`, `center_lat`.
3. Intensidad: `vmax_kt`.
4. Radio de máximo viento: `rmw_km`, cuando esté disponible.
5. Radios de viento, especialmente `R34`.
6. Radios externos `ROCLOUD`.
7. Campo de precipitación tipo snapshot.
8. Percentiles climatológicos locales: `Q90`, `Q95`, `Q99`.
9. Campo de viento normalizado `V*` o parámetros para construirlo.

---

## Salidas esperadas

| Salida | Descripción |
|---|---|
| `ITCHI_g,h,t` | Índice por celda y tiempo |
| `H_P` | Peligro por precipitación |
| `H_W` | Peligro por viento |
| `H_Pdir` | Precipitación directa |
| `H_Pind` | Precipitación indirecta |
| `H_dir` | Componente directo |
| `H_ind` | Componente indirecto |
| `M_direct` | Máscara de región directa |
| `M_indirect` | Máscara de región indirecta |
| `M_exterior` | Máscara exterior |
| `ITCHI_max` | Máximo por evento |
| `ITCHI_acc` | Acumulado acotado o persistencia por evento |

---

## Estado actual del proyecto

| Componente | Estado | Descripción |
|---|---|---|
| Estructura base | ✅ Completado | Repositorio, ambiente, configuración y paquete instalable |
| Módulos core | ✅ Completado | Precipitación, geometría, máscaras, componentes e índice |
| Radios y unidades | ✅ Completado | Conversión de unidades, cuadrantes, `R_direct`, `ROCLOUD` |
| Viento | ✅ Completado | Perfil radial simple y normalización `V*` |
| Pipeline integrado | ✅ Completado | Cálculo por snapshot con control de calidad opcional |
| Agregación | ✅ Completado | `ITCHI_max` e `ITCHI_acc` |
| Compiler | ✅ Completado | Orquestación de múltiples snapshots por evento |
| I/O | ✅ Completado | Conversión a `xarray.Dataset`, lectura y escritura |
| Tracks y ROCLOUD | ✅ Completado | Estandarización tabular y extracción de radios |
| Pruebas unitarias | ✅ Completado | Cobertura por módulo con `pytest` |
| Notebook sintético | ✅ Completado | Smoke test visual/reproducible |
| Ejemplo ejecutable | ✅ Completado | Script sintético desde terminal |
| Caso real | ⏳ Pendiente | Integración con datos reales de un ciclón |
| Climatología formal | ⏳ Pendiente | Cálculo/lectura robusta de Q90/Q95/Q99 |
| Alineación temporal avanzada | ⏳ Pendiente | Módulo específico para precipitación snapshot |

**Leyenda**: ✅ Completado | 🔄 En progreso | ⏳ Pendiente

---

## Instalación

### Requisitos previos

- Python 3.11 o superior
- Conda o pip
- Git

### Opción 1: Conda

```bash
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

conda env create -f environment.yml
conda activate itchi

pip install -e .
```

### Opción 2: pip/venv

```bash
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Verificar instalación

```bash
python -c "import itchi; print('✓ ITCHI instalado correctamente')"
python -m pytest tests/
pre-commit run --all-files
```

---

## Ejemplos reproducibles

### Synthetic smoke test

```bash
python examples/smoke_test_synthetic.py
```

Con figura diagnóstica:

```bash
python examples/smoke_test_synthetic.py \
  --plot \
  --figure-path outputs/figures/smoke_test_synthetic.png
```

### Notebook sintético

```bash
jupyter lab notebooks/00_smoke_test_synthetic.ipynb
```

---

## Pruebas y control de calidad

Ejecutar todas las pruebas:

```bash
python -m pytest tests/
```

Ejecutar pre-commit:

```bash
pre-commit run --all-files
```

Ejecutar pruebas de un módulo específico:

```bash
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_compiler.py
```

Reglas de consistencia verificadas:

- `ITCHI` permanece en `[0, 1]`.
- `H_P`, `H_W`, `H_dir`, `H_ind`, `H_Pdir` y `H_Pind` permanecen en `[0, 1]`.
- `R_direct_q <= ROCLOUD_q`.
- La región exterior no contribuye al índice.
- Las máscaras directa, indirecta y exterior no se solapan.
- Los campos preservan dimensiones y coordenadas cuando se usa `xarray`.
- Las unidades geométricas se normalizan a kilómetros.

---

## Data Requirements

El repositorio requiere datos en los siguientes formatos.

### 1. Trayectorias de ciclones tropicales

**Formato:** IBTrACS, best-track local, CSV o Parquet.

Variables mínimas:

- `storm_id`
- `time`
- `lat`
- `lon`
- `vmax_kt`
- `pmin_hpa` opcional
- `rmw_km` opcional

### 2. Radios de viento

Variables esperadas:

- `R34_NE`, `R34_SE`, `R34_SW`, `R34_NW`, o equivalentes.
- Unidad típica: millas náuticas (`nm`).

### 3. Radios ROCLOUD

Variables esperadas:

- `ROCLOUD_RNE`, `ROCLOUD_RSE`, `ROCLOUD_RSW`, `ROCLOUD_RNW`, o equivalentes.
- Unidad esperada: kilómetros (`km`).

### 4. Campos de precipitación

**Formato:** NetCDF o Zarr.

Especificaciones:

- Resolución objetivo: 0.1° × 0.1°.
- Tipo: snapshot, no acumulado temporal.
- Unidades: deben coincidir con las climatologías usadas para Q90/Q95/Q99.
- Cobertura: dominio suficiente alrededor del centro del ciclón.

Fuentes compatibles:

- MSWEP histórico.
- Pronósticos numéricos.
- Hindcasts.
- Productos downscalados.
- Productos corregidos por sesgo.

### 5. Percentiles climatológicos

Variables requeridas:

- `Q90`
- `Q95`
- `Q99`

Deben estar calculados por celda espacial y por ventana temporal comparable.

---

## Contributing

Las contribuciones son bienvenidas.

### Configuración de desarrollo

```bash
git clone https://github.com/YOUR_USERNAME/itchi-core.git
cd itchi-core

conda env create -f environment.yml
conda activate itchi
pip install -e .
pre-commit install
```

### Convenciones de código

- Estilo: `black` con línea máxima de 88 caracteres.
- Linting: `ruff`.
- Type hints recomendados en funciones públicas.
- Docstrings estilo NumPy.
- Toda nueva funcionalidad debe incluir pruebas.
- No subir datos pesados ni productos derivados.

### Commits

Formato sugerido:

```text
type: brief description
```

Tipos recomendados:

- `feat`
- `fix`
- `docs`
- `test`
- `refactor`
- `style`
- `chore`

Ejemplo:

```text
feat: add event-level ITCHI compiler
```

---

## Autoría

**Adolfo Perez-Estrada**

Universidad Nacional Autónoma de México (UNAM)
Instituto de Ciencias de la Atmósfera y Cambio Climático (ICACC)

---

## Licencia

Licencia por definir.

Para más detalles sobre licencias de software científico, ver [choosealicense.com](https://choosealicense.com/).

---

## References

### Key Resources

- **IBTrACS**: International Best Track Archive for Climate Stewardship.
- **MSWEP**: Multi-Source Weighted-Ensemble Precipitation.
- **ROCLOUD**: Database for the outer sizes of tropical cyclones over the Middle Americas.

### Related Documentation

- [Project Architecture](docs/architecture.md)
- [Examples](examples/)
- [Synthetic smoke test notebook](notebooks/00_smoke_test_synthetic.ipynb)

### Scientific References

- Pérez-Estrada & Dominguez (2025): A database for the outer sizes of tropical cyclones over the Middle Americas.
- Pérez-Alarcón et al. (2021): Comparative climatology of outer tropical cyclone size using radial wind profiles.
- Knapp et al. (2010): The International Best Track Archive for Climate Stewardship (IBTrACS).
- Beck et al. (2019): MSWEP V2 global 3-hourly 0.1° precipitation.

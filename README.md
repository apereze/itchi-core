# ITCHI Core

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: TBD](https://img.shields.io/badge/License-TBD-yellow.svg)](#16-licencia)
[![Tests](https://img.shields.io/badge/Tests-Planned-orange.svg)]()

**Integrated Tropical Cyclone Hazard Index**

Repositorio para el desarrollo del núcleo computacional de **ITCHI v0.1**, un índice físico de peligro asociado a ciclones tropicales.

---

## Table of Contents

- [Quick Start](#quick-start)
- [General Description](#1-descripción-general)
- [Repository Scope](#2-alcance-del-repositorio)
- [Methodology](#3-principio-metodológico)
- [Installation](#14-instalación)
- [Repository Status](#repository-status)
- [Data Requirements](#data-requirements)
- [Contributing](#contributing)
- [License](#16-licencia)
- [References](#references)

---

## Quick Start

Get up and running with ITCHI Core in minutes:

```bash
# Clone the repository
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

# Create and activate environment (using conda)
conda env create -f environment.yml
conda activate itchi-core

# Install package in development mode
pip install -e .

# Verify installation
python -c "import itchi; print('ITCHI installed successfully')"

# Run tests
pytest tests/
```

For detailed setup instructions, see [Installation](#14-instalación).

---

## 1. Descripción general

**ITCHI** significa **Integrated Tropical Cyclone Hazard Index**.

La versión inicial, **ITCHI v0.1**, se plantea como un índice físico continuo entre `0` y `1`, diseñado para representar el peligro asociado a ciclones tropicales en una malla espacial.

El índice integra tres componentes principales:

1. **Peligro por precipitación directa**, asociado a lluvia dentro de la región de viento significativo del ciclón.
2. **Peligro por viento**, estimado a partir de la estructura radial del ciclón.
3. **Peligro por precipitación indirecta**, asociado a bandas externas del ciclón.

El objetivo de este repositorio es construir de forma organizada, trazable y reproducible el cálculo del índice ITCHI.

---

## 2. Alcance del repositorio

Este repositorio contiene únicamente los elementos necesarios para la **creación del índice ITCHI**.

### Incluye

- Lectura y estandarización de datos de trayectoria ciclónica.
- Lectura de radios de viento, como `R34`.
- Lectura de radios estructurales externos, como `ROCLOUD`.
- Lectura de campos de precipitación tipo snapshot.
- Alineación temporal entre precipitación y tiempos sinópticos del ciclón.
- Cálculo de percentiles climatológicos locales de precipitación.
- Cálculo del peligro normalizado por precipitación.
- Cálculo del peligro normalizado por viento.
- Construcción de máscaras espaciales:
  - región directa,
  - región indirecta,
  - región exterior.
- Cálculo de los componentes físicos del índice.
- Cálculo final de `ITCHI`.
- Generación de productos derivados por evento.

### No incluye por ahora

- Validación con declaratorias de emergencia o desastre.
- Modelos de aprendizaje automático.
- Calibración estadística avanzada.
- Visualizaciones finales para publicación.
- Manuscrito científico completo.
- Dashboard o aplicación web.

Estos elementos podrán desarrollarse posteriormente en otros módulos o repositorios.

---

## 3. Principio metodológico

ITCHI v0.1 distingue tres regiones alrededor del ciclón tropical:

| Región | Condición conceptual | Interpretación |
|---|---|---|
| Región directa | Dentro de `R34` | Zona con viento significativo y precipitación directa |
| Región indirecta | Fuera de `R34`, pero dentro de `ROCLOUD` | Zona de bandas externas asociadas al ciclón |
| Región exterior | Fuera de `ROCLOUD` | Zona no atribuida al ciclón en ITCHI v0.1 |

La separación conceptual es:

```text
Peligro directo = viento + precipitación dentro de R34
```

```text
Peligro indirecto = precipitación entre R34 y ROCLOUD
```

La región exterior no contribuye al índice.

---

## 4. Tratamiento de la precipitación

En esta versión del repositorio, la precipitación se manejará como **snapshot**, no como acumulado temporal.

Esto significa que el campo de precipitación representa el estado o intensidad de la precipitación en un tiempo determinado.

Por tanto, ITCHI no debe asumir que la precipitación se acumula de 3 h a 6 h.

La lógica general será:

```text
campo de precipitación en snapshot
        ↓
alineación con tiempo sinóptico del ciclón
        ↓
normalización con percentiles climatológicos locales
        ↓
peligro por precipitación
        ↓
ITCHI
```

Esta decisión permite que el índice sea compatible con:

* MSWEP histórico;
* productos de pronóstico;
* hindcasts;
* productos downscalados;
* productos corregidos por sesgo.

---

## 5. Resolución temporal del índice

Aunque la precipitación se maneje como snapshot, el índice se calculará en tiempos sinópticos asociados al ciclón:

```text
00, 06, 12 y 18 UTC
```

Esto permite mantener coherencia temporal con:

* trayectoria del ciclón;
* intensidad del ciclón;
* radios de viento;
* `R34`;
* `ROCLOUD`;
* perfil radial de viento.

---

## 6. Normalización de precipitación

El peligro por precipitación se calculará usando percentiles climatológicos locales.

Los percentiles principales serán:

| Percentil | Interpretación                     |
| --------- | ---------------------------------- |
| `P90`     | Inicio de peligro bajo u ocasional |
| `P95`     | Peligro alto                       |
| `P99`     | Peligro máximo                     |

La normalización debe calcularse por:

* celda espacial;
* mes;
* hora sinóptica o tiempo válido comparable.

Esto evita comparar directamente regiones con climatologías de lluvia distintas.

---

## 7. Componentes del índice

El índice se construye a partir de los siguientes componentes:

| Componente | Descripción                                     |
| ---------- | ----------------------------------------------- |
| `H_P`      | Peligro normalizado por precipitación           |
| `H_W`      | Peligro normalizado por viento                  |
| `H_Pdir`   | Precipitación directa dentro de `R34`           |
| `H_Pind`   | Precipitación indirecta entre `R34` y `ROCLOUD` |
| `H_dir`    | Componente directo del peligro                  |
| `H_ind`    | Componente indirecto del peligro                |
| `ITCHI`    | Índice integrado final                          |

La combinación de componentes se realizará con una unión acotada para mantener el índice dentro del intervalo `[0, 1]`.

---

## 8. Fórmula conceptual

El componente directo combina viento y precipitación directa:

```text
H_dir = 1 - (1 - H_W) * (1 - H_Pdir)
```

El componente indirecto se define como:

```text
H_ind = H_Pind
```

El índice final se calcula como:

```text
ITCHI = 1 - (1 - H_dir) * (1 - H_ind)
```

Esta formulación permite que el índice se active por:

* viento intenso;
* precipitación directa intensa;
* precipitación indirecta intensa;
* combinación de estos procesos.

---

## 9. Estructura inicial esperada del repositorio

La estructura del repositorio se irá construyendo progresivamente.

Una estructura inicial recomendada es:

```text
itchi-core/
│
├── README.md
├── .gitignore
├── environment.yml
├── pyproject.toml
│
├── configs/
│   └── default.yaml
│
├── docs/
│   └── architecture.md
│
├── notebooks/
│   └── 00_check_inputs.ipynb
│
├── scripts/
│   └── run_single_storm.py
│
├── src/
│   └── itchi/
│       ├── __init__.py
│       ├── config.py
│       ├── precipitation.py
│       ├── geometry.py
│       ├── masks.py
│       ├── wind.py
│       ├── components.py
│       ├── index.py
│       └── pipeline.py
│
└── tests/
    └── test_index.py
```

Esta estructura puede crecer conforme avance el desarrollo.

---

## 10. Entradas esperadas

Las entradas mínimas para calcular ITCHI son:

1. Datos de trayectoria del ciclón.
2. Intensidad del ciclón.
3. Radios de viento, especialmente `R34`.
4. Radios `ROCLOUD`.
5. Campo de precipitación tipo snapshot.
6. Percentiles climatológicos locales de precipitación.
7. Parámetros o perfiles de viento radial.

---

## 11. Salidas esperadas

Las salidas mínimas del repositorio serán:

| Salida        | Descripción                         |
| ------------- | ----------------------------------- |
| `ITCHI_g_h_t` | Índice por celda, ciclón y tiempo   |
| `H_P`         | Peligro por precipitación           |
| `H_W`         | Peligro por viento                  |
| `H_Pdir`      | Precipitación directa               |
| `H_Pind`      | Precipitación indirecta             |
| `H_dir`       | Componente directo                  |
| `H_ind`       | Componente indirecto                |
| `ITCHI_max`   | Máximo por evento                   |
| `ITCHI_acc`   | Acumulado o persistencia por evento |

---

## 12. Estado actual del proyecto

Este repositorio se encuentra en etapa inicial.

La prioridad actual es construir de forma ordenada:

1. la estructura base del repositorio;
2. la documentación metodológica mínima;
3. las funciones centrales del índice;
4. una primera corrida para un ciclón individual;
5. pruebas básicas de consistencia.

---

## 13. Reglas de consistencia

La implementación debe verificar que:

* `ITCHI` siempre esté dentro del intervalo `[0, 1]`;
* la región exterior no contribuya al índice;
* `R34` no sea mayor que `ROCLOUD` cuando ambas variables estén disponibles;
* las máscaras directa, indirecta y exterior sean coherentes;
* la precipitación usada sea comparable con la climatología usada para normalizarla;
* los campos de pronóstico o downscaling estén alineados por `valid_time`.

---

## 14. Instalación

### Requisitos previos

- Python 3.9 o superior
- Conda o pip (recomendado: conda)
- Git

### Configuración

#### Opción 1: Usando Conda (recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

# Crear el ambiente
conda env create -f environment.yml

# Activar el ambiente
conda activate itchi-core

# Instalar el paquete en modo desarrollo
pip install -e .
```

#### Opción 2: Usando pip

```bash
# Clonar el repositorio
git clone https://github.com/apereze/itchi-core.git
cd itchi-core

# Crear un ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install -e .
```

### Verificar la instalación

```bash
# Verificar que el paquete está instalado
python -c "import itchi; print('✓ ITCHI instalado correctamente')"

# Ejecutar las pruebas
pytest tests/ -v
```

### Dependencias principales

Se usará un ambiente de Python con paquetes científicos y geoespaciales:

* `numpy` — computación numérica
* `pandas` — manejo de datos tabulares
* `xarray` — datos multidimensionales etiquetados
* `geopandas` — datos geoespaciales
* `shapely` — geometría espacial
* `pyproj` — transformaciones de coordenadas
* `scipy` — algoritmos científicos
* `matplotlib` — visualización
* `cartopy` — mapas geográficos
* `tqdm` — barras de progreso
* `pytest` — pruebas unitarias

---

## Repository Status

| Componente | Estado | Descripción |
|---|---|---|
| Estructura base | ✅ Completado | Repositorio y documentación inicial |
| Módulos core | ⏳ En desarrollo | precipitation, geometry, masks, components, index |
| Pruebas unitarias | ⏳ Planificado | Cobertura para todos los módulos |
| Pipeline integrado | ⏳ Planificado | Función principal de cálculo |
| Documentación | 🔄 En progreso | Arquitectura y metodología |
| Primer ejemplo funcional | ⏳ Planificado | Caso de prueba con ciclón real |
| Módulos adicionales |  🔄 En progreso | tracks.py, rocloud.py, wind.py, aggregation.py, io.py |

**Leyenda**: ✅ Completado | 🔄 En progreso | ⏳ Planificado

---

## Data Requirements

El repositorio requiere datos en los siguientes formatos:

### 1. Trayectorias de ciclones tropicales

**Formato**: IBTrACS o equivalente
**Variables mínimas**:
- `time` — timestamp (UTC)
- `lat`, `lon` — posición del centro
- `vmax` — velocidad máxima sostenida (kt)
- `mslp` — presión mínima a nivel del mar (mb)

**Fuente recomendada**: [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)

### 2. Radios de viento

**Formato**: NetCDF o CSV
**Variables**:
- `R34_NE`, `R34_SE`, `R34_SW`, `R34_NW` — radio de vientos 34 nudos (nm)
- `R50_*`, `R64_*` — opcional, radios adicionales
- `ROCLOUD_*` — radio de cobertura de nube (km)

### 3. Campos de precipitación

**Formato**: NetCDF o Zarr
**Especificaciones**:
- Resolución: 0.1° × 0.1° (compatible con MSWEP)
- Tipo: snapshot (no acumulado)
- Unidades: mm/h o mm/día
- Cobertura: al menos +/- 5° del centro del ciclón

**Fuentes compatibles**:
- [MSWEP](http://www.gloh2o.org/) — histórico
- Pronósticos numéricos (GFS, HWRF, etc.)
- Satélite (IMERG, PERSIANN, etc.)

### 4. Percentiles climatológicos

**Variables requeridas**:
- `Q90`, `Q95`, `Q99` — percentiles por celda espacial
- Dimensiones: `(lat, lon, month, hour)`
- Unidades: mismas que el campo de precipitación

**Generación**: Ver `notebooks/compute_climatology.ipynb` (por crear)

### 5. Estructura de directorios de datos

```
data/
├── examples/
│   ├── tc_track_sample.nc
│   ├── precipitation_sample.nc
│   └── climatology_sample.nc
├── raw/
│   ├── ibtracs/
│   ├── precipitation/
│   └── climatology/
└── processed/
    └── itchi_products/
```

---

## Contributing

Las contribuciones son bienvenidas. Para contribuir, por favor:

### 1. Configurar el ambiente de desarrollo

```bash
# Clonar tu fork
git clone https://github.com/YOUR_USERNAME/itchi-core.git
cd itchi-core

# Crear rama de desarrollo
git checkout -b feature/nombre-de-tu-feature
```

### 2. Convenciones de código

- **Estilo**: PEP 8 (usa `black` o `flake8`)
- **Type hints**: Recomendados para funciones públicas
- **Docstrings**: NumPy style para documentación
- **Tests**: Toda nueva funcionalidad debe incluir pruebas

Ejemplo:

```python
def compute_precipitation_hazard(
    precipitation: np.ndarray,
    q90: float,
    q95: float,
    q99: float,
) -> np.ndarray:
    """
    Compute normalized precipitation hazard.

    Parameters
    ----------
    precipitation : np.ndarray
        Precipitation field in mm/h
    q90, q95, q99 : float
        Local climatological percentiles

    Returns
    -------
    np.ndarray
        Normalized hazard H_P in [0, 1]
    """
    pass
```

### 3. Commits

Usa mensajes descriptivos siguiendo el formato:

```
type: brief description

Detailed explanation if needed.

- Bullet point 1
- Bullet point 2
```

**Tipos**: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

Ejemplo:
```
feat: add precipitation normalization module

Implements H_P calculation with piecewise linear normalization
following architecture specification. Includes unit tests and
docstring documentation.

- Add precipitation.py module
- Implement normalize_precipitation() function
- Add tests/test_precipitation.py
- Update CHANGELOG.md
```

### 4. Testing

```bash
# Ejecutar todas las pruebas
pytest tests/ -v

# Ejecutar con cobertura
pytest tests/ --cov=itchi

# Pruebas de un módulo específico
pytest tests/test_precipitation.py -v
```

### 5. Enviar un Pull Request

```bash
# Asegúrate de que tu código está actualizado
git fetch origin
git rebase origin/main

# Push tu rama
git push origin feature/nombre-de-tu-feature
```

Después, abre un PR en GitHub con:
- Título descriptivo
- Referencia a cualquier issue relacionado
- Descripción clara de los cambios
- Checklist de verificación completado

### Directrices de revisión

- Al menos 1 revisión requerida
- Todas las pruebas deben pasar
- Cobertura de código no debe disminuir
- Documentación debe estar actualizada

### Reportar issues

Usa la plantilla de issue de GitHub e incluye:
- Descripción clara del problema
- Pasos para reproducirlo
- Comportamiento esperado vs. actual
- Versión de Python y dependencias

---

## 15. Autoría

[Adolfo Perez-Estrada](https://github.com/apereze)

Universidad Nacional Autónoma de México (UNAM)
Instituto de Ciencias de la Atmosfera y Cambio Climático (ICACC)

---

## 16. Licencia

Licencia por definir.

Para más detalles sobre licencias de software científico, ver [choosealicense.com](https://choosealicense.com/).

---

## References

### Key Resources

- **IBTrACS**: [International Best Track Archive for Climate Stewardship](https://www.ncei.noaa.gov/products/international-best-track-archive)
- **MSWEP**: [Multi-Source Weighted-Ensemble Precipitation](http://www.gloh2o.org/)
- **ROCLOUD**: [A database for the outer sizes of tropical cyclones over the Middle Americas](https://data.mendeley.com/drafts/5bpzbwhynd)

### Related Documentation

- [Project Architecture](docs/architecture.md)
- [Methodology Details](docs/methodology.md) *(pending)*
- [API Reference](docs/api.md) *(pending)*

### Scientific References
- Perez-Estrada & Dominguez (2025): A database for the outer sizes of tropical cyclones over the Middle Americas
- Pérez-Alarcon et al (2021): Comparative climatology of outer tropical cyclone size using radial wind profiles
- Knapp et al. (2010): The International Best Track Archive for Climate Stewardship (IBTrACS)
- Beck et al. (2019): MSWEP V2 Global 3-hourly 0.1° Precipitation

# ITCHI Core

**Integrated Tropical Cyclone Hazard Index**

Repositorio para el desarrollo del núcleo computacional de **ITCHI v0.1**, un índice físico de peligro asociado a ciclones tropicales.

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
````

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
│   └── methodology.md
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

La instalación se definirá conforme avance la estructura del repositorio.

De forma preliminar, se usará un ambiente de Python con paquetes científicos y geoespaciales como:

* `numpy`
* `pandas`
* `xarray`
* `geopandas`
* `shapely`
* `pyproj`
* `scipy`
* `matplotlib`
* `cartopy`
* `tqdm`

---

## 15. Autoría

Adolfo Perez-Estrada 
Universidad Nacional Autónoma de México (UNAM)
Instituto de Ciencias de la Atmosfera y Cambio Climático

---

## 16. Licencia

Licencia por definir.

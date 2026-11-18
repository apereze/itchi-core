# Historial de ITCHI (v0.1 a v0.2.2)

Documentos de las versiones anteriores al esquema vigente. Se conservan porque
registran las decisiones y los resultados que llevaron a él; el código, los
notebooks y las configuraciones que describen ya no están en la rama de
trabajo y se recuperan del tag `v0.2.2` (por ejemplo,
`git show v0.2.2:notebooks/"second test"/05_itchi_v02_mejoras.ipynb > 05.ipynb`
o `git worktree add ../itchi-v022 v0.2.2`).

| Documento | Contenido |
|---|---|
| `methodology.md` | Metodología de v0.1 (rampa Q90/Q95/Q99, viento dentro de R34) y v0.2 (matriz con máximo en P99, viento relativo a Vmax) |
| `verification_v02.md` | Calibración y verificación de v0.2 contra declaratorias |
| `experimento_b1.md` | Bloque 1 de v0.2.1: P95 como máximo, umbrales de R24 y A72, corte del viento en 12 m/s |
| `avance_v021.md` | Avance v0.2.1 (casos, Saffir-Simpson, deslizamientos) y, en la sección 8, los acuerdos que dieron v0.2.2 |
| `api.md`, `architecture.md`, `workflows.md`, `data_contracts.md` | API, arquitectura y flujos de v0.1 (compilador de eventos, `compute_itchi_snapshot_from_grid`) |
| `real_case_validation.md`, `dev_log.md` | Lista de validación con casos reales y bitácora de desarrollo de v0.1 |

El esquema vigente está en `docs/methodology.md`.

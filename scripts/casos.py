"""
ITCHI-CORE: ITCHI de los casos de estudio en malla y por municipio.

Calcula ITCHI con el esquema de configs/default.yaml en cada posición de la
trayectoria que puede afectar a México (itchi.cases), lo guarda en caché
(outputs/diagnostics/casos/<caso>.nc) e imprime los municipios y píxeles
terrestres por clase del máximo del evento.

Experimentos adicionales: --experimento nombre=configs/experiments/<overlay>.yaml
(cada overlay cambia solo sus claves sobre configs/default.yaml).

Ejecutar desde scripts/ (Anaconda Prompt, entorno itchi):
    python casos.py                                  # todos los casos
    python casos.py --casos ODILE2014 MANUEL2013 --forzar
"""

from __future__ import annotations

import argparse
import time

from itchi.cases import CASES, CaseConfig, class_counts, load_or_compute


def main() -> None:
    ap = argparse.ArgumentParser(description="ITCHI por caso en malla y por municipio")
    ap.add_argument("--casos", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--forzar", action="store_true", help="recalcula aunque exista el caché")
    ap.add_argument("--experimento", nargs="*", default=[], metavar="NOMBRE=OVERLAY",
                    help="experimentos además del esquema por defecto")
    args = ap.parse_args()
    experiments = {"itchi": None}
    for item in args.experimento:
        name, overlay = item.split("=", 1)
        experiments[name] = overlay
    cfg = CaseConfig(experiments=experiments)
    for case in args.casos:
        t0 = time.time()
        C = load_or_compute(case, cfg, force=args.forzar)
        print(class_counts(C).to_string())
        print(f"{case}: {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()

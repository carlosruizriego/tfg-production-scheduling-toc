# -*- coding: utf-8 -*-
"""Ampliación del estudio de parametrización (criterio estricto de la Revisión 10):
   D) probabilidad de CRUCE pc en {0.7, 0.9, 1.0} (pm=0.3, pop=50) — comentario #76.
   E) tiempos de convergencia de CP-SAT también en C3 y C4 — comentario #103.
Añade los resultados al mismo resultados_params_ga.json."""
import os, sys, json, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE, "results", "resultados_params_ga.json")

import estudio_c2 as E2
SEEDS_AJUSTE = list(range(1, 11))

def ga_pc(ordenes, n_gen, pop, pm, pc, seed=0):
    rng = np.random.default_rng(seed); n = len(ordenes)
    def fit(seq):
        return E2.evaluar(ordenes, seq)["throughput_neto"]
    pobl = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(x) for x in pobl]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2)
            p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            h = E2.ox(p1, p2, rng) if rng.random() < pc else p1[:]
            if rng.random() < pm:
                i, k = rng.integers(0, n, 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]
        pobl = nueva; fits = [fit(x) for x in pobl]
    return E2.evaluar(ordenes, pobl[int(np.argmax(fits))])

def main():
    salida = json.load(open(PATH))
    if "sensibilidad_pc" not in salida:
        print("[D] Barrido de probabilidad de cruce pc (pm=0.3, pop=50, 45 gen)", flush=True)
        res = {}
        for pc in (0.7, 0.9, 1.0):
            vals = [ga_pc(E2.generar_ordenes_c2(seed=s), 45, 50, 0.3, pc)["throughput"]
                    for s in SEEDS_AJUSTE]
            res["pc%s" % pc] = {"media": float(np.mean(vals)), "std": float(np.std(vals))}
            print("   pc=%s: %.1f +- %.1f" % (pc, np.mean(vals), np.std(vals)), flush=True)
        salida["sensibilidad_pc"] = res
        json.dump(salida, open(PATH, "w"), indent=2)
    if "c3" not in salida.get("tiempos_cpsat", {}):
        print("[E] Tiempos CP-SAT en C3 y C4 (5 instancias)", flush=True)
        import estudio_c3 as E3
        import estudio_c4 as E4
        import planta_caso3, planta_caso4
        gens = {
            "c3": (E3, lambda s: planta_caso3.generar_ordenes_c3(
                mix=getattr(E3, "POOL", getattr(planta_caso3, "POOL_DEFAULT", None))[:], seed=s)),
            "c4": (E4, lambda s: planta_caso4.generar_ordenes_c4(seed=s)),
        }
        for k, (mod, gen) in gens.items():
            ts = []
            for s in SEEDS_AJUSTE[:5]:
                o = gen(s)
                t0 = time.time(); mod.metodo_cpsat(o); ts.append(time.time() - t0)
            salida["tiempos_cpsat"][k] = {"medio_s": round(float(np.mean(ts)), 2),
                                          "max_s": round(float(np.max(ts)), 2)}
            print("   %s: medio %.2fs max %.2fs" % (k, np.mean(ts), np.max(ts)), flush=True)
        json.dump(salida, open(PATH, "w"), indent=2)
    print("[OK] estudio ampliado", flush=True)

if __name__ == "__main__":
    main()

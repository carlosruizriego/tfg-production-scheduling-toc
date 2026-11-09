# -*- coding: utf-8 -*-
"""Parametrización del genético POR CASO (criterio estricto, comentarios #76-#78):
para cada caso C1..C5 (nivel de trabajo), sobre 10 instancias de ajuste propias
(semillas 1..10, disjuntas de las de test):
  - rejilla pm x poblacion (0.1/0.3/0.5 x 30/50/80) con 45 generaciones;
  - barrido de probabilidad de cruce pc (0.7/0.9/1.0) con pm=0.3, pop=50;
  - configuracion adoptada del caso (Tabla 6.2) evaluada en las mismas instancias;
  - curva de convergencia (120 generaciones, pop=50, pm=0.3).
Reanudable por caso. Salida: clave 'porcaso' en resultados_params_ga.json.
"""
import os, sys, json, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE, "results", "resultados_params_ga.json")
SEEDS = list(range(1, 11))

import estudio_c1 as E1
import estudio_c2 as E2
import estudio_c3 as E3
import estudio_c4 as E4
import estudio_c5 as E5
import planta_caso3, planta_caso4, planta_caso5

CASOS = {
    "c1": dict(mod=E1, gen=lambda s: E1.generar_instancia(seed=s), fit="throughput",
               adopt=(40, 40)),
    "c2": dict(mod=E2, gen=lambda s: E2.generar_ordenes_c2(seed=s), fit="throughput_neto",
               adopt=(45, 50)),
    "c3": dict(mod=E3, gen=lambda s: planta_caso3.generar_ordenes_c3(mix=E3.POOL[:], seed=s),
               fit="throughput", adopt=(45, 50)),
    "c4": dict(mod=E4, gen=lambda s: planta_caso4.generar_ordenes_c4(seed=s),
               fit="throughput", adopt=(45, 50)),
    "c5": dict(mod=E5, gen=lambda s: planta_caso5.generar_ordenes_c5(pool=E5.POOL_DEFAULT[:], seed=s),
               fit="throughput", adopt=(50, 50)),
}

def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.integers(0, n, 2)); h = [None] * n
    h[a:b] = p1[a:b]; resto = [x for x in p2 if x not in h]; pos = 0
    for i in range(n):
        if h[i] is None: h[i] = resto[pos]; pos += 1
    return h

def ga(cfg, ordenes, n_gen, pop, pm, pc=1.0, seed=0, curva=False):
    mod, fitkey = cfg["mod"], cfg["fit"]
    rng = np.random.default_rng(seed); n = len(ordenes)
    def fit(seq):
        return mod.evaluar(ordenes, seq)[fitkey]
    pobl = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(x) for x in pobl]
    hist = [max(fits)]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2)
            p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            h = ox(p1, p2, rng) if rng.random() < pc else p1[:]
            if rng.random() < pm:
                i, k = rng.integers(0, n, 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]
        pobl = nueva; fits = [fit(x) for x in pobl]
        hist.append(max(fits))
    mejor = pobl[int(np.argmax(fits))]
    th = cfg["mod"].evaluar(ordenes, mejor)["throughput"]
    return (th, hist) if curva else (th, None)

def procesar(caso):
    cfg = CASOS[caso]
    t0 = time.time()
    res = {}
    # rejilla
    grid = {}
    for pm in (0.1, 0.3, 0.5):
        for pop in (30, 50, 80):
            vals = [ga(cfg, cfg["gen"](s), 45, pop, pm)[0] for s in SEEDS]
            grid["pm%s_pop%d" % (pm, pop)] = {"media": float(np.mean(vals)),
                                              "std": float(np.std(vals))}
    res["grid"] = grid
    # pc
    pcres = {}
    for pc in (0.7, 0.9, 1.0):
        vals = [ga(cfg, cfg["gen"](s), 45, 50, 0.3, pc=pc)[0] for s in SEEDS]
        pcres["pc%s" % pc] = {"media": float(np.mean(vals)), "std": float(np.std(vals))}
    res["pc"] = pcres
    # adoptada (Tabla 6.2)
    ngen_a, pop_a = cfg["adopt"]
    vals = [ga(cfg, cfg["gen"](s), ngen_a, pop_a, 0.3)[0] for s in SEEDS]
    res["adoptada"] = {"n_gen": ngen_a, "pop": pop_a,
                       "media": float(np.mean(vals)), "std": float(np.std(vals))}
    # curva
    curvas = [ga(cfg, cfg["gen"](s), 120, 50, 0.3, curva=True)[1] for s in SEEDS]
    media = np.mean(np.array(curvas, dtype=float), axis=0)
    mejora = media - media[0]
    total = mejora[-1] if mejora[-1] > 0 else 1.0
    meseta = int(np.argmax(mejora >= 0.99 * total))
    res["curva"] = [round(float(x), 1) for x in media]
    res["meseta_gen"] = meseta
    res["t_s"] = round(time.time() - t0, 1)
    return res

def main():
    salida = json.load(open(PATH))
    porcaso = salida.get("porcaso", {})
    for caso in ("c1", "c2", "c3", "c4", "c5"):
        if caso in porcaso:
            print("[%s] ya calculado, salto" % caso, flush=True)
            continue
        print("[%s] procesando..." % caso, flush=True)
        porcaso[caso] = procesar(caso)
        salida["porcaso"] = porcaso
        json.dump(salida, open(PATH, "w"), indent=2)
        g = porcaso[caso]
        mejor_k = max(g["grid"], key=lambda k: g["grid"][k]["media"])
        print("   mejor rejilla %s=%.1f | adoptada=%.1f | meseta gen %d | %.0fs"
              % (mejor_k, g["grid"][mejor_k]["media"], g["adoptada"]["media"],
                 g["meseta_gen"], g["t_s"]), flush=True)
    print("[OK] porcaso completo", flush=True)

if __name__ == "__main__":
    main()

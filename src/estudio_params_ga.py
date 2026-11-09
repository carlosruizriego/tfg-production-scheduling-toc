# -*- coding: utf-8 -*-
"""
Estudio de parametrizacion del algoritmo genetico (Caso 2 como caso de referencia).

  A) Sensibilidad: rejilla pm x pop (probabilidad de mutacion x tamano de poblacion)
     con n_gen = 45 fijo, sobre 10 instancias de AJUSTE (semillas 1..10, distintas
     de las 30 de test que usan semillas 100+; sin fuga de informacion).
  B) Convergencia: curva del mejor fitness por generacion (pop=50, pm=0.3, 120 gen)
     sobre las mismas 10 instancias -> justifica el numero de generaciones.
  C) Tiempo de resolucion del metodo exacto (CP-SAT) por instancia (C1 y C2).

Salida: results/resultados_params_ga.json y figura rig_params_ga.png.
"""
import os, sys, json, time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_params_ga.json")

import estudio_c2 as E2                      # evaluar (con penalizacion), ox, generador

SEEDS_AJUSTE = list(range(1, 11))            # disjuntas de las de test (100+)

def ga(ordenes, n_gen, pop, pm, seed=0, curva=False):
    rng = np.random.default_rng(seed); n = len(ordenes)
    def fit(seq):
        return E2.evaluar(ordenes, seq)["throughput_neto"]
    pobl = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(s) for s in pobl]
    mejor_hist = [max(fits)]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2)
            p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            h = E2.ox(p1, p2, rng)
            if rng.random() < pm:
                i, k = rng.integers(0, n, 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]      # elitismo
        pobl = nueva; fits = [fit(s) for s in pobl]
        mejor_hist.append(max(fits))
    mejor = pobl[int(np.argmax(fits))]
    res = E2.evaluar(ordenes, mejor)
    return (res, mejor_hist) if curva else (res, None)

def main():
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}

    # ---------- A) sensibilidad pm x pop ----------
    if "sensibilidad" not in salida:
        print("[A] Sensibilidad pm x pop (n_gen=45, 10 instancias de ajuste)", flush=True)
        grid = {}
        t0 = time.time()
        for pm in (0.1, 0.3, 0.5):
            for pop in (30, 50, 80):
                vals = []
                for s in SEEDS_AJUSTE:
                    ordenes = E2.generar_ordenes_c2(seed=s)
                    res, _ = ga(ordenes, n_gen=45, pop=pop, pm=pm, seed=0)
                    vals.append(res["throughput"])
                grid[f"pm{pm}_pop{pop}"] = {
                    "pm": pm, "pop": pop, "media": float(np.mean(vals)),
                    "std": float(np.std(vals)), "evals": pop * 46,
                }
                print(f"   pm={pm} pop={pop}: {np.mean(vals):.1f} +- {np.std(vals):.1f}",
                      flush=True)
        salida["sensibilidad"] = grid
        salida["t_sensibilidad_s"] = round(time.time() - t0, 1)
        json.dump(salida, open(PATH, "w"), indent=2)

    # ---------- B) convergencia ----------
    if "convergencia" not in salida:
        print("[B] Curva de convergencia (pop=50, pm=0.3, 120 generaciones)", flush=True)
        curvas = []
        t0 = time.time()
        for s in SEEDS_AJUSTE:
            ordenes = E2.generar_ordenes_c2(seed=s)
            _, hist = ga(ordenes, n_gen=120, pop=50, pm=0.3, seed=0, curva=True)
            curvas.append(hist)
            print(f"   instancia {s}: final {hist[-1]:.0f}", flush=True)
        media = np.mean(np.array(curvas), axis=0)
        salida["convergencia"] = {"media_por_gen": [round(float(x), 1) for x in media],
                                  "n_gen": 120, "pop": 50, "pm": 0.3}
        salida["t_convergencia_s"] = round(time.time() - t0, 1)
        json.dump(salida, open(PATH, "w"), indent=2)

    # ---------- C) tiempo del exacto ----------
    if "tiempos_cpsat" not in salida:
        print("[C] Tiempos de resolucion CP-SAT (5 instancias, C1 y C2)", flush=True)
        tiempos = {}
        import estudio_c1 as E1
        for nombre, mod, gen in (("c1", E1, lambda s: E1.generar_instancia(seed=s)),
                                 ("c2", E2, lambda s: E2.generar_ordenes_c2(seed=s))):
            ts = []
            for s in SEEDS_AJUSTE[:5]:
                orden = gen(s)
                t0 = time.time()
                mod.metodo_cpsat(orden)
                ts.append(time.time() - t0)
            tiempos[nombre] = {"medio_s": round(float(np.mean(ts)), 2),
                               "max_s": round(float(np.max(ts)), 2)}
            print(f"   {nombre}: medio {np.mean(ts):.2f}s max {np.max(ts):.2f}s", flush=True)
        salida["tiempos_cpsat"] = tiempos
        json.dump(salida, open(PATH, "w"), indent=2)

    # ---------- figura ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = salida["sensibilidad"]; conv = salida["convergencia"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    pops = [30, 50, 80]; pms = [0.1, 0.3, 0.5]
    cols = {0.1: "#9aa0a6", 0.3: "#2E8B57", 0.5: "#C55A11"}
    x = np.arange(len(pops)); w = 0.25
    for i, pm in enumerate(pms):
        medias = [g[f"pm{pm}_pop{p}"]["media"] for p in pops]
        stds = [g[f"pm{pm}_pop{p}"]["std"] for p in pops]
        a1.bar(x + (i - 1) * w, medias, w, yerr=stds, capsize=4,
               label=f"pm = {pm}", color=cols[pm], edgecolor="black", zorder=3,
               error_kw=dict(ecolor="#333", lw=1.1))
    a1.set_xticks(x); a1.set_xticklabels([f"población {p}" for p in pops])
    a1.set_ylabel("Throughput medio (EUR), 10 instancias de ajuste")
    a1.set_title("Sensibilidad del genético a pm y al tamaño de población\n(Caso 2, 45 generaciones)",
                 fontsize=10.5)
    ymin = min(g[k]["media"] for k in g) - 60
    a1.set_ylim(ymin, None)
    a1.legend(fontsize=8.5); a1.grid(True, axis="y", alpha=0.3, zorder=0)
    curva = conv["media_por_gen"]
    a2.plot(curva, color="#2E8B57", lw=1.8, zorder=3)
    a2.axvline(45, color="#C0392B", ls="--", lw=1.2)
    a2.text(47, curva[5], "45 generaciones\n(valor adoptado)", fontsize=8.5, color="#C0392B")
    a2.set_xlabel("Generación"); a2.set_ylabel("Mejor aptitud media (EUR)")
    a2.set_title("Convergencia del genético (pop = 50, pm = 0,3)", fontsize=10.5)
    a2.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "rig_params_ga.png"), dpi=150)
    plt.close()
    print("[OK] resultados_params_ga.json + rig_params_ga.png", flush=True)

if __name__ == "__main__":
    main()

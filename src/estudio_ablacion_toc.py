# -*- coding: utf-8 -*-
"""
ABLACION DE LA LOGICA TOC
=========================
Cuantifica cuanto aporta la logica de la Teoria de las Restricciones, aislandola
del resto del aparato algoritmico. Compara, sobre el MISMO simulador y las mismas
30 instancias por caso, tres niveles de "inteligencia TOC":

  1. FIFO          -> TOC-ciego (orden de llegada).
  2. TOC-greedy    -> heuristica TOC pura: identifica el cuello (recurso de mayor
                      carga) y despacha por T/CU decreciente (margen por unidad de
                      tiempo de cuello). Sin optimizacion ni busqueda.
  3. NSGA-II       -> optimizacion completa (se lee de los resultados ya calculados).

Metrica principal: fraccion de la mejora alcanzable (FIFO -> NSGA-II) que captura la
heuristica TOC por si sola, a coste de calculo despreciable. Test de Wilcoxon
(TOC-greedy vs FIFO) para la significacion.

Salida: results/resultados_ablacion_toc.json y figura rig_ablacion_toc.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import estudio_c1, estudio_c2, estudio_c3, estudio_c4

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
FIGDIR = os.path.join(_BASE, "figures")


def tcu_greedy(ordenes, n_maq, tiempo_en, margen_de):
    """Heuristica TOC: cuello = recurso de mayor carga; despacho por T/CU decreciente."""
    carga = [sum(tiempo_en(o, m) for o in ordenes) for m in range(n_maq)]
    cuello = int(np.argmax(carga))

    def tcu(o):
        t = tiempo_en(o, cuello)
        return float("inf") if t <= 0 else margen_de(o) / t

    return sorted(range(len(ordenes)), key=lambda i: tcu(ordenes[i]), reverse=True)


# ─── Adaptadores por caso (estructura de orden distinta en cada uno) ───
def _t_c1(o, m):
    return sum(op.duracion for op in o.operaciones if op.maquina == m)


def _t_c2(o, m):
    return sum(t for mm, t in o.ruta if mm == m)


CASES = [
    dict(key="c1", gen=lambda sd: estudio_c1.generar_instancia(seed=sd),
         evaluar=estudio_c1.evaluar, n_maq=estudio_c1.N_MACHINES,
         tiempo=_t_c1, margen=lambda o: o.valor_venta - o.coste_material,
         seeds=estudio_c1.SEEDS_TEST),
    dict(key="c2", gen=lambda sd: estudio_c2.generar_ordenes_c2(seed=sd),
         evaluar=estudio_c2.evaluar, n_maq=estudio_c2.N_MACHINES_C2,
         tiempo=_t_c2, margen=lambda o: o.margen, seeds=estudio_c2.SEEDS_TEST),
    dict(key="c3", gen=estudio_c3.gen, evaluar=estudio_c3.evaluar,
         n_maq=estudio_c3.N_MACHINES_C3, tiempo=lambda o, m: o.tiempos[m],
         margen=lambda o: o.margen, seeds=estudio_c3.SEEDS_TEST),
    dict(key="c4", gen=estudio_c4.gen, evaluar=estudio_c4.evaluar,
         n_maq=estudio_c4.N_MACHINES_C4, tiempo=lambda o, m: o.tiempos[m],
         margen=lambda o: o.margen, seeds=estudio_c4.SEEDS_TEST),
]


def raw_from_json(key, metodo):
    d = json.load(open(os.path.join(RESDIR, f"resultados_{key}.json")))
    return np.array(d["raw"][metodo], dtype=float)


def main():
    print("=" * 64)
    print("  ABLACION DE LA LOGICA TOC (FIFO -> TOC-greedy -> NSGA-II)")
    print("=" * 64)
    salida = {}
    for case in CASES:
        k = case["key"]
        toc = []
        for sd in case["seeds"]:
            ordenes = case["gen"](sd)
            seq = tcu_greedy(ordenes, case["n_maq"], case["tiempo"], case["margen"])
            toc.append(case["evaluar"](ordenes, seq)["throughput"])
        toc = np.array(toc, dtype=float)
        fifo = raw_from_json(k, "FIFO")
        nsga = raw_from_json(k, "NSGA-II")
        mejora_max = nsga.mean() - fifo.mean()
        captura = 100 * (toc.mean() - fifo.mean()) / mejora_max if mejora_max > 1e-9 else 0.0
        try:
            _, p = stats.wilcoxon(toc, fifo)
        except ValueError:
            p = 1.0
        salida[k] = {
            "fifo": round(float(fifo.mean()), 1), "toc_greedy": round(float(toc.mean()), 1),
            "nsga": round(float(nsga.mean()), 1),
            "toc_std": round(float(toc.std()), 1),
            "mejora_toc_vs_fifo_pct": round(100 * (toc.mean() - fifo.mean()) / fifo.mean(), 1),
            "fraccion_brecha_capturada_pct": round(float(captura), 1),
            "wilcoxon_toc_vs_fifo_p": float(p),
        }
        print(f"\n[{k.upper()}] FIFO={fifo.mean():.1f}  TOC-greedy={toc.mean():.1f}  "
              f"NSGA-II={nsga.mean():.1f}")
        print(f"   TOC vs FIFO: {salida[k]['mejora_toc_vs_fifo_pct']:+.1f}%  "
              f"(p={p:.4f})  | captura {salida[k]['fraccion_brecha_capturada_pct']:.0f}% "
              f"de la brecha FIFO->NSGA-II")
    json.dump(salida, open(os.path.join(RESDIR, "resultados_ablacion_toc.json"), "w"), indent=2)
    print("\n[OK] resultados_ablacion_toc.json")
    figura(salida)


def figura(salida):
    keys = ["c1", "c2", "c3", "c4"]
    etiq = {"c1": "Caso 1", "c2": "Caso 2", "c3": "Caso 3", "c4": "Caso 4"}
    x = np.arange(len(keys)); w = 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    # Panel 1: escalera FIFO -> TOC-greedy -> NSGA-II
    for i, (met, col, lab) in enumerate([("fifo", "#9aa0a6", "FIFO (TOC-ciego)"),
                                         ("toc_greedy", "#C28F2C", "Heurística T/CU (TOC pura)"),
                                         ("nsga", "#2E8B57", "Genético (optimización)")]):
        a1.bar(x + (i - 1) * w, [salida[k][met] for k in keys], w, label=lab, color=col,
               edgecolor="black", zorder=3)
    a1.set_xticks(x); a1.set_xticklabels([etiq[k] for k in keys])
    a1.set_ylabel("Throughput medio T (EUR)")
    a1.set_title("La lógica TOC captura parte de la mejora sin optimizar")
    a1.legend(fontsize=8.5); a1.grid(True, axis="y", alpha=0.3, zorder=0)
    # Panel 2: fraccion de brecha capturada
    frac = [salida[k]["fraccion_brecha_capturada_pct"] for k in keys]
    bars = a2.bar(x, frac, 0.55, color="#C28F2C", edgecolor="black", zorder=3)
    for xi, f in zip(x, frac):
        a2.text(xi, f + 2, f"{f:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    a2.axhline(100, color="#2E8B57", ls="--", lw=1, label="óptimo (genético)")
    a2.set_xticks(x); a2.set_xticklabels([etiq[k] for k in keys])
    a2.set_ylabel("% de la brecha FIFO→genético capturada por la TOC")
    a2.set_ylim(0, 115)
    a2.set_title("Cuánto del potencial logra la heurística TOC")
    a2.legend(fontsize=8.5); a2.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "rig_ablacion_toc.png"), dpi=150)
    plt.close()
    print("[OK] rig_ablacion_toc.png")


if __name__ == "__main__":
    main()

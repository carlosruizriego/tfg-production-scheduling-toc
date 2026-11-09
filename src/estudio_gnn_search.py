# -*- coding: utf-8 -*-
"""
Agente GNN MEJORADO: red de grafos mas profunda (E) + busqueda en inferencia (A).
=================================================================================
Quinto intento de superar al genetico en los casos medios (C2, C4), donde el agente
GNN ya lo igualaba. Dos palancas sobre el agente del apartado 7.10:
  (E) GNN de 3 capas de paso de mensajes (en vez de 2): proposal mas expresivo.
  (A) BUSQUEDA EN INFERENCIA: en lugar de decidir greedy (argmax), se MUESTREAN K
      secuencias de la politica estocastica y se queda la MEJOR evaluada en el simulador.
      La politica propone; el simulador selecciona. Usa K=100 evaluaciones << las ~1600
      del genetico, de modo que ganar seria un resultado fuerte (10x menos evaluaciones).
Solo se entrena el PPO; baselines leidos de los JSON. Reutiliza demos cacheadas (la
observacion no cambia respecto al agente base). Reanudable, wake-lock.
Salida: resultados_gnn_search.json y rig_gnn_search.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch as th
from scipy import stats
from stable_baselines3 import PPO

import estudio_ppo_gnn as PG
import gnn_grafo as G
from estudio_ppo_toc import _wake_lock

RESDIR = PG.RESDIR; FIGDIR = PG.FIGDIR
PATH = os.path.join(RESDIR, "resultados_gnn_search.json")
SEEDS = [0, 1, 2]
LAYERS = 3
K_SEARCH = 100
CASES = [c for c in PG.CASES if c["key"] in ("c2", "c4")]   # casos medios (donde GNN ya iguala)


def build(case):
    cfg = G.cfg_para(case["key"])
    GraphEnv, OBS = G.make_gnn_env(case["base"], cfg)
    Pol = G.make_masked_policy(OBS - cfg["NJOBS"], cfg["NJOBS"])
    pk = dict(features_extractor_class=G.GNNExtractor,
              features_extractor_kwargs=dict(N=cfg["N_PAD"], NJ=cfg["NJOBS"], H=64, L=LAYERS),
              net_arch=dict(pi=[], vf=[64]))
    return cfg, GraphEnv, Pol, pk


def eval_search(model, case, GraphEnv, K):
    """Muestrea K secuencias de la politica y devuelve, por instancia, la mejor."""
    out = []
    for sd in case["seeds_test"]:
        ordenes = case["gen"](sd); best = -1.0
        for _ in range(K):
            e = GraphEnv(train=False); e.fijar_instancia(ordenes); o, _ = e.reset(); done = False
            while not done:
                ot, _ = model.policy.obs_to_tensor(o)
                a = int(model.policy.get_distribution(ot).sample().item())
                o, _, t, tr, _ = e.step(a); done = t or tr
            thr = case["evaluar"](ordenes, getattr(e, case["sched"]))[case["evalkey"]]
            if thr > best:
                best = thr
        out.append(best)
    return np.array(out, dtype=float)


def train_one(case, seed, demos):
    O, A, R = demos
    cfg, GraphEnv, Pol, pk = build(case)
    env = GraphEnv(train=True); env.rng = np.random.default_rng(seed)
    model = PPO(Pol, env, verbose=0, seed=seed, policy_kwargs=pk, **PG.HP)
    PG.bc_pretrain(model, O, A, PG.BC_EPOCHS, seed)
    PG.warm_critic(model, O, R)
    model.learn(total_timesteps=int(case["steps"] * 0.4))
    greedy = PG.eval_masked(model, case, GraphEnv)
    search = eval_search(model, case, GraphEnv, K_SEARCH)
    return greedy.tolist(), search.tolist()


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print(f"  GNN MEJORADO: {LAYERS} capas (E) + busqueda K={K_SEARCH} (A) | casos medios", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    prog = salida.get("_prog", {})
    for case in CASES:
        k = case["key"]; prog.setdefault(k, {})
        if any(str(s) not in prog[k] for s in SEEDS):
            _, GraphEnv, _, _ = build(case)
            demos = PG.construir_demos(case, GraphEnv)   # reutiliza cache
        for s in SEEDS:
            if str(s) in prog[k]:
                continue
            g, se = train_one(case, s, demos)
            prog[k][str(s)] = {"greedy": g, "search": se}
            salida["_prog"] = prog; json.dump(salida, open(PATH, "w"))
            print(f"   [{k}] semilla {s}: greedy={np.mean(g):.1f}  search={np.mean(se):.1f}", flush=True)
        gre = np.mean([prog[k][str(s)]["greedy"] for s in SEEDS], axis=0)
        sea = np.mean([prog[k][str(s)]["search"] for s in SEEDS], axis=0)
        base = PG.baselines(k); nsga_raw = np.array(base["raw_nsga"], dtype=float)
        try:
            _, p = stats.wilcoxon(sea, nsga_raw)
        except ValueError:
            p = 1.0
        salida[k] = {"gnn_greedy": round(float(gre.mean()), 1),
                     "gnn_search": round(float(sea.mean()), 1),
                     "nsga": round(base["NSGA-II"], 1), "fifo": round(base["FIFO"], 1),
                     "mejora_search_vs_nsga_pct": round(100 * (sea.mean() - base["NSGA-II"]) / base["NSGA-II"], 1),
                     "wilcoxon_search_vs_nsga_p": float(p)}
        json.dump(salida, open(PATH, "w"), indent=2)
        gana = "SUPERA" if sea.mean() > base["NSGA-II"] else "no supera"
        print(f"  -> [{k}] greedy {gre.mean():.1f}  SEARCH {sea.mean():.1f}  | NSGA-II "
              f"{base['NSGA-II']:.0f}  ({gana}, {salida[k]['mejora_search_vs_nsga_pct']:+.1f}%, "
              f"p={p:.3f})", flush=True)
    print("\n[OK] resultados_gnn_search.json", flush=True)
    figura(salida)
    _wake_lock(False)


def figura(salida):
    keys = [c["key"] for c in CASES if c["key"] in salida]
    etiq = {f"c{i}": f"Caso {i}" for i in range(1, 6)}
    series = [("fifo", "#9aa0a6", "FIFO"), ("gnn_greedy", "#7B2FBF", "PPO-GNN (greedy)"),
              ("gnn_search", "#C0392B", f"PPO-GNN + búsqueda (K={K_SEARCH})"),
              ("nsga", "#2E8B57", "Genético")]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (m, c, lab) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [salida[k][m] for k in keys], w, label=lab, color=c,
               edgecolor="black", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([etiq[k] for k in keys])
    ax.set_ylabel("Throughput medio T (EUR)")
    ax.set_title("Agente GNN con búsqueda en inferencia frente al genético (casos medios)")
    ax.legend(ncol=2, fontsize=9, loc="upper left"); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "rig_gnn_search.png"), dpi=150); plt.close()
    print("[OK] rig_gnn_search.png", flush=True)


if __name__ == "__main__":
    main()

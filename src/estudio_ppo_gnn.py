# -*- coding: utf-8 -*-
"""
PPO con politica GNN sobre grafo disyuntivo, warm-start por imitacion del GA.
=============================================================================
Cuarto y definitivo intento de hacer competitivo al agente. Los tres anteriores
(reward shaping, imitacion+masking con MLP) fracasaron por la observacion pobre del
MLP. Aqui la politica es una RED NEURONAL DE GRAFOS sobre el grafo disyuntivo
(nucleo en gnn_grafo.py), que da al agente la representacion POR-OPERACION que faltaba.

Pipeline por caso y semilla:
  1. WARM-START: imitar (behavioral cloning) una buena secuencia del GA -> GNN-BC.
  2. WARM-CRITIC: regresion de valor sobre el retorno terminal del GA (evita que PPO
     degrade el warm-start por un critico sin entrenar).
  3. FINE-TUNE PPO sobre el simulador (recompensa = throughput, sin shaping para no
     desalinear la metrica) -> GNN-PPO.
Se reportan AMBOS (GNN-BC y GNN-PPO) como ablacion honesta (sin seleccionar sobre test).
Solo se entrena el PPO; FIFO/CP-SAT/NSGA-II se LEEN de los JSON. Reanudable, wake-lock.
Salida: results/resultados_ppo_gnn.json y rig_ppo_gnn.png.
"""
import numpy as np, json, os, warnings, pickle
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch as th
from scipy import stats
from stable_baselines3 import PPO

import estudio_c1, estudio_c2, estudio_c3, estudio_c4, estudio_c5
from estudio_ppo_pro import ga_secuencia
from estudio_ppo_toc import _wake_lock
import gnn_grafo as G

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results"); FIGDIR = os.path.join(_BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_ppo_gnn.json")

SEEDS = [0, 1, 2]
N_DEMOS = 60
BC_EPOCHS = 30
HP = dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
          gamma=1.0, ent_coef=0.005)
NORM = {"c1": 700.0, "c2": 800.0, "c3": 700.0, "c4": 900.0, "c5": 1200.0}

CASES = [
    dict(key="c1", base=estudio_c1.SeqConstructEnv, gen=lambda s: estudio_c1.generar_instancia(seed=s),
         evaluar=estudio_c1.evaluar, sched="scheduled", seeds_test=estudio_c1.SEEDS_TEST,
         steps=120_000, fitkey="throughput", evalkey="throughput"),
    dict(key="c2", base=estudio_c2.SeqEnv, gen=lambda s: estudio_c2.generar_ordenes_c2(seed=s),
         evaluar=estudio_c2.evaluar, sched="sched", seeds_test=estudio_c2.SEEDS_TEST,
         steps=150_000, fitkey="throughput", evalkey="throughput"),   # homogeneizado a throughput
    dict(key="c3", base=estudio_c3.SeqEnv, gen=estudio_c3.gen, evaluar=estudio_c3.evaluar,
         sched="sched", seeds_test=estudio_c3.SEEDS_TEST, steps=150_000,
         fitkey="throughput", evalkey="throughput"),
    dict(key="c4", base=estudio_c4.SeqEnv, gen=estudio_c4.gen, evaluar=estudio_c4.evaluar,
         sched="sched", seeds_test=estudio_c4.SEEDS_TEST, steps=160_000,
         fitkey="throughput", evalkey="throughput"),
    dict(key="c5", base=estudio_c5.SeqEnvC5, gen=estudio_c5.gen, evaluar=estudio_c5.evaluar,
         sched="sched", seeds_test=estudio_c5.SEEDS_TEST, steps=200_000,
         fitkey="throughput", evalkey="throughput"),
]


def build(case):
    cfg = G.cfg_para(case["key"])
    GraphEnv, OBS = G.make_gnn_env(case["base"], cfg)
    Pol = G.make_masked_policy(OBS - cfg["NJOBS"], cfg["NJOBS"])
    pk = dict(features_extractor_class=G.GNNExtractor,
              features_extractor_kwargs=dict(N=cfg["N_PAD"], NJ=cfg["NJOBS"], H=64),
              net_arch=dict(pi=[], vf=[64]))
    return cfg, GraphEnv, Pol, pk


def construir_demos(case, GraphEnv):
    """Devuelve (obs, accion, retorno_norm) imitando secuencias del GA."""
    pk = os.path.join(RESDIR, f"demos_gnn_{case['key']}.pkl")
    if os.path.exists(pk):
        return pickle.load(open(pk, "rb"))
    O, A, R = [], [], []
    for sd in range(300, 300 + N_DEMOS):
        ordenes = case["gen"](sd)
        seq = ga_secuencia(ordenes, case["evaluar"], case["fitkey"], seed=sd)
        thr = case["evaluar"](ordenes, seq)[case["evalkey"]] / NORM[case["key"]]
        e = GraphEnv(train=False); e.fijar_instancia(ordenes); o, _ = e.reset()
        for a in seq:
            O.append(o.astype(np.float32)); A.append(int(a)); R.append(float(thr))
            o, _, t, tr, _ = e.step(int(a))
            if t or tr:
                break
    out = (np.array(O, np.float32), np.array(A, np.int64), np.array(R, np.float32))
    pickle.dump(out, open(pk, "wb"))
    return out


def bc_pretrain(model, O, A, epochs, seed):
    opt = th.optim.Adam([p for n, p in model.policy.named_parameters() if "value_net" not in n],
                        lr=5e-4)
    rng = np.random.default_rng(seed); n = len(O)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, 256):
            b = idx[i:i + 256]
            ot, _ = model.policy.obs_to_tensor(O[b])
            loss = -model.policy.get_distribution(ot).log_prob(th.as_tensor(A[b])).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def warm_critic(model, O, R, epochs=10):
    """Regresion de valor sobre el retorno del GA (politica congelada)."""
    head = [p for n, p in model.policy.named_parameters() if "value_net" in n]
    opt = th.optim.Adam(head, lr=1e-3)
    Rt = th.as_tensor(R).unsqueeze(-1)
    rng = np.random.default_rng(0); n = len(O)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, 512):
            b = idx[i:i + 512]
            ot, _ = model.policy.obs_to_tensor(O[b])
            v = model.policy.predict_values(ot)
            loss = ((v - Rt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def eval_masked(model, case, GraphEnv):
    out = []
    for sd in case["seeds_test"]:
        ordenes = case["gen"](sd)
        e = GraphEnv(train=False); e.fijar_instancia(ordenes); o, _ = e.reset(); done = False
        while not done:
            ot, _ = model.policy.obs_to_tensor(o)
            lg = model.policy.get_distribution(ot).distribution.logits.detach().numpy().ravel()
            a = int(np.argmax(lg))
            o, _, t, tr, _ = e.step(a); done = t or tr
        out.append(case["evaluar"](ordenes, getattr(e, case["sched"]))[case["evalkey"]])
    return np.array(out, dtype=float)


def train_one(case, seed, demos):
    O, A, R = demos
    cfg, GraphEnv, Pol, pk = build(case)
    env = GraphEnv(train=True); env.rng = np.random.default_rng(seed)
    model = PPO(Pol, env, verbose=0, seed=seed, policy_kwargs=pk, **HP)
    bc_pretrain(model, O, A, BC_EPOCHS, seed)
    bc_vec = eval_masked(model, case, GraphEnv)
    warm_critic(model, O, R)
    model.learn(total_timesteps=int(case["steps"] * 0.4))
    ft_vec = eval_masked(model, case, GraphEnv)
    return bc_vec.tolist(), ft_vec.tolist()


def baselines(key):
    d = json.load(open(os.path.join(RESDIR, f"resultados_{key}.json")))
    r = d["resumen"]
    out = {m: r[m]["media"] for m in ["FIFO", "CP-SAT", "NSGA-II"]}
    out["raw_nsga"] = d["raw"]["NSGA-II"]
    return out


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print("  PPO con politica GNN (grafo disyuntivo) + warm-start GA", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    prog = salida.get("_prog", {})
    for case in CASES:
        k = case["key"]; prog.setdefault(k, {})
        if any(str(s) not in prog[k] for s in SEEDS):
            print(f"\n[{k}] demos del GA (warm-start)...", flush=True)
            _, GraphEnv, _, _ = build(case)
            demos = construir_demos(case, GraphEnv)
        for s in SEEDS:
            if str(s) in prog[k]:
                continue
            bc, ft = train_one(case, s, demos)
            prog[k][str(s)] = {"bc": bc, "ft": ft}
            salida["_prog"] = prog; json.dump(salida, open(PATH, "w"))
            print(f"   [{k}] semilla {s}: BC={np.mean(bc):.1f}  FT={np.mean(ft):.1f}", flush=True)
        bc_inst = np.mean([prog[k][str(s)]["bc"] for s in SEEDS], axis=0)
        ft_inst = np.mean([prog[k][str(s)]["ft"] for s in SEEDS], axis=0)
        base = baselines(k)
        nsga_raw = np.array(base["raw_nsga"], dtype=float)
        try:
            _, p_bc = stats.wilcoxon(bc_inst, nsga_raw)
        except ValueError:
            p_bc = 1.0
        salida[k] = {
            "gnn_bc": round(float(bc_inst.mean()), 1), "gnn_ft": round(float(ft_inst.mean()), 1),
            "fifo": round(base["FIFO"], 1), "cpsat": round(base["CP-SAT"], 1),
            "nsga": round(base["NSGA-II"], 1),
            "brecha_bc_a_nsga_pct": round(100 * (base["NSGA-II"] - bc_inst.mean()) / base["NSGA-II"], 1),
            "wilcoxon_bc_vs_nsga_p": float(p_bc),
        }
        json.dump(salida, open(PATH, "w"), indent=2)
        print(f"  -> [{k}] GNN-BC {bc_inst.mean():.1f}  GNN-PPO {ft_inst.mean():.1f}  | "
              f"NSGA-II {base['NSGA-II']:.0f}  (brecha BC {salida[k]['brecha_bc_a_nsga_pct']:+.1f}%, "
              f"p={p_bc:.3f})", flush=True)
    print("\n[OK] resultados_ppo_gnn.json", flush=True)
    if all(c["key"] in salida for c in CASES):
        figura(salida)
    _wake_lock(False)


def figura(salida):
    keys = [k for k in ["c1", "c2", "c3", "c4", "c5"] if k in salida]
    etiq = {f"c{i}": f"Caso {i}" for i in range(1, 6)}
    series = [("fifo", "#9aa0a6", "FIFO"), ("gnn_bc", "#7B2FBF", "PPO-GNN (BC)"),
              ("gnn_ft", "#1F6FB2", "PPO-GNN (fine-tune)"), ("nsga", "#2E8B57", "Genético")]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    for i, (m, c, lab) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [salida[k][m] for k in keys], w, label=lab, color=c,
               edgecolor="black", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([etiq[k] for k in keys])
    ax.set_ylabel("Throughput medio T (EUR)")
    ax.set_title("Agente PPO con política GNN (grafo disyuntivo) frente al genético")
    ax.legend(ncol=4, fontsize=8.5, loc="upper left"); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "rig_ppo_gnn.png"), dpi=150); plt.close()
    print("[OK] rig_ppo_gnn.png", flush=True)


if __name__ == "__main__":
    main()

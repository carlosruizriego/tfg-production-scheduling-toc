# -*- coding: utf-8 -*-
"""
PPO "PRO": warm-start por imitacion + fine-tune RL + inferencia enmascarada
===========================================================================
Tercera y mas ambiciosa mejora del agente. El reward shaping (estudio_ppo_toc) no
basto. Aqui se atacan las dos debilidades de raiz:

  1. WARM-START por imitacion (behavioral cloning): se preentrena la politica para
     IMITAR las buenas secuencias halladas por NSGA-II, lo que salta el problema de
     exploracion de la recompensa terminal.
  2. INFERENCIA ENMASCARADA: al decidir, el agente solo elige entre ordenes NO
     programadas (se enmascaran los logits de las ya despachadas), de modo que ninguna
     accion se desperdicia.
  3. (Se conserva el reward shaping por T/CU en el fine-tune.)

Se compara con el PPO-vanilla (estudio_ppo_toc) y con NSGA-II. Reanudable, wake-lock.
Salida: results/resultados_ppo_pro.json y figura rig_ppo_pro.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy import stats
from stable_baselines3 import PPO

import estudio_c1, estudio_c2, estudio_c3, estudio_c4
from estudio_ppo_toc import make_toc_env, _t_c1, _t_c2, _wake_lock

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
FIGDIR = os.path.join(_BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_ppo_pro.json")

SEEDS = [0, 1, 2]
N_DEMOS = 80              # instancias expertas (NSGA-II) por caso
BC_EPOCHS = 12
FINETUNE_FRAC = 0.5       # fine-tune = mitad de los pasos del estudio principal
HP = dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
          gamma=1.0, ent_coef=0.01)


def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.integers(0, n, 2)); h = [None] * n
    h[a:b] = p1[a:b]; resto = [x for x in p2 if x not in h]; pos = 0
    for i in range(n):
        if h[i] is None:
            h[i] = resto[pos]; pos += 1
    return h


def ga_secuencia(ordenes, evaluar, fitkey, n_gen=40, pop=40, seed=0):
    rng = np.random.default_rng(seed); n = len(ordenes)
    def fit(s): return evaluar(ordenes, s)[fitkey]
    pobl = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(s) for s in pobl]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2); p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2); p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            h = ox(p1, p2, rng)
            if rng.random() < 0.3:
                i, k = rng.integers(0, n, 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]; pobl = nueva; fits = [fit(s) for s in pobl]
    return pobl[int(np.argmax(fits))]


CASES = [
    dict(key="c1", env=estudio_c1.SeqConstructEnv, sched="scheduled",
         gen=lambda sd: estudio_c1.generar_instancia(seed=sd), evaluar=estudio_c1.evaluar,
         seeds_test=estudio_c1.SEEDS_TEST, steps=120_000, n_maq=estudio_c1.N_MACHINES,
         t=_t_c1, margen=lambda o: o.valor_venta - o.coste_material, fitkey="throughput"),
    dict(key="c2", env=estudio_c2.SeqEnv, sched="sched",
         gen=lambda sd: estudio_c2.generar_ordenes_c2(seed=sd), evaluar=estudio_c2.evaluar,
         seeds_test=estudio_c2.SEEDS_TEST, steps=150_000, n_maq=estudio_c2.N_MACHINES_C2,
         t=_t_c2, margen=lambda o: o.margen, fitkey="throughput_neto"),
    dict(key="c3", env=estudio_c3.SeqEnv, sched="sched", gen=estudio_c3.gen,
         evaluar=estudio_c3.evaluar, seeds_test=estudio_c3.SEEDS_TEST, steps=150_000,
         n_maq=estudio_c3.N_MACHINES_C3, t=lambda o, m: o.tiempos[m],
         margen=lambda o: o.margen, fitkey="throughput"),
    dict(key="c4", env=estudio_c4.SeqEnv, sched="sched", gen=estudio_c4.gen,
         evaluar=estudio_c4.evaluar, seeds_test=estudio_c4.SEEDS_TEST, steps=160_000,
         n_maq=estudio_c4.N_MACHINES_C4, t=lambda o, m: o.tiempos[m],
         margen=lambda o: o.margen, fitkey="throughput"),
]


def construir_demos(case, n_inst):
    """Replica secuencias expertas de NSGA-II para obtener pares (observacion, accion)."""
    obss, acts = [], []
    for sd in range(300, 300 + n_inst):
        ordenes = case["gen"](sd)
        seq = ga_secuencia(ordenes, case["evaluar"], case["fitkey"], seed=sd)
        e = case["env"](train=False); e.fijar_instancia(ordenes)
        obs, _ = e.reset()
        for a in seq:
            obss.append(np.asarray(obs, dtype=np.float32)); acts.append(int(a))
            obs, _, term, trunc, _ = e.step(int(a))
            if term or trunc:
                break
    return np.array(obss, dtype=np.float32), np.array(acts, dtype=np.int64)


def bc_pretrain(model, obss, acts, epochs, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = len(obss); bs = 256
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            ot, _ = model.policy.obs_to_tensor(obss[b])
            dist = model.policy.get_distribution(ot)
            loss = -dist.log_prob(torch.as_tensor(acts[b])).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def eval_masked(model, case, ordenes):
    e = case["env"](train=False); e.fijar_instancia(ordenes)
    obs, _ = e.reset(); done = False
    while not done:
        ot, _ = model.policy.obs_to_tensor(obs)
        logits = model.policy.get_distribution(ot).distribution.logits.detach().numpy().ravel()
        logits[e.mask == 1] = -1e9
        a = int(np.argmax(logits))
        obs, _, term, trunc, _ = e.step(a)
        done = term or trunc
    return case["evaluar"](ordenes, getattr(e, case["sched"]))["throughput"]


def train_one(case, seed, demos):
    obss, acts = demos
    EnvCls = make_toc_env(case)              # reward shaping TOC en el fine-tune
    env = EnvCls(train=True); env.rng = np.random.default_rng(seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, **HP)
    bc_pretrain(model, obss, acts, BC_EPOCHS, seed)
    ft = int(case["steps"] * FINETUNE_FRAC)
    if ft > 0:
        model.learn(total_timesteps=ft)
    return [float(eval_masked(model, case, case["gen"](sd))) for sd in case["seeds_test"]]


def vanilla_inst(key):
    d = json.load(open(os.path.join(RESDIR, "resultados_ppo_toc.json")))
    van = d["_prog"][key]["vanilla"]
    return np.mean([van[s] for s in van], axis=0)


def nsga_media(key):
    d = json.load(open(os.path.join(RESDIR, f"resultados_{key}.json")))
    return d["resumen"]["NSGA-II"]["media"], d["resumen"]["FIFO"]["media"]


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print("  PPO PRO: warm-start (imitacion) + masking + shaping", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    prog = salida.get("_prog", {})
    demos_cache = {}
    for case in CASES:
        k = case["key"]
        prog.setdefault(k, {})
        if any(str(s) not in prog[k] for s in SEEDS):
            if k not in demos_cache:
                print(f"\n[{k}] generando {N_DEMOS} demostraciones expertas (NSGA-II)...",
                      flush=True)
                demos_cache[k] = construir_demos(case, N_DEMOS)
        for s in SEEDS:
            if str(s) in prog[k]:
                continue
            vals = train_one(case, s, demos_cache[k])
            prog[k][str(s)] = vals
            salida["_prog"] = prog; json.dump(salida, open(PATH, "w"))
            print(f"   [{k}] semilla {s}: media {np.mean(vals):.1f}", flush=True)
        pro = np.mean([prog[k][str(s)] for s in SEEDS], axis=0)
        van = vanilla_inst(k)
        nsga, fifo = nsga_media(k)
        try:
            _, p = stats.wilcoxon(pro, van)
        except ValueError:
            p = 1.0
        salida[k] = {"ppo_vanilla": round(float(van.mean()), 1),
                     "ppo_pro": round(float(pro.mean()), 1),
                     "nsga": round(float(nsga), 1), "fifo": round(float(fifo), 1),
                     "mejora_vs_vanilla_pct": round(100 * (pro.mean() - van.mean()) / van.mean(), 1),
                     "brecha_a_nsga_pct": round(100 * (nsga - pro.mean()) / nsga, 1),
                     "wilcoxon_p": float(p)}
        json.dump(salida, open(PATH, "w"), indent=2)
        print(f"  -> [{k}] vanilla {van.mean():.1f}  PRO {pro.mean():.1f}  "
              f"({salida[k]['mejora_vs_vanilla_pct']:+.1f}%, p={p:.4f})  "
              f"NSGA-II {nsga:.0f} (brecha {salida[k]['brecha_a_nsga_pct']:.1f}%)", flush=True)
    print("\n[OK] resultados_ppo_pro.json", flush=True)
    if all(c["key"] in salida for c in CASES):
        figura(salida)
    _wake_lock(False)


def figura(salida):
    keys = ["c1", "c2", "c3", "c4"]; etiq = {"c1": "Caso 1", "c2": "Caso 2",
                                             "c3": "Caso 3", "c4": "Caso 4"}
    cols = {"fifo": "#9aa0a6", "ppo_vanilla": "#E8743B", "ppo_pro": "#1F6FB2",
            "nsga": "#2E8B57"}
    labs = {"fifo": "FIFO", "ppo_vanilla": "PPO vanilla", "ppo_pro": "PPO pro (imitación+masking)",
            "nsga": "Genético"}
    orden = ["fifo", "ppo_vanilla", "ppo_pro", "nsga"]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for i, met in enumerate(orden):
        ax.bar(x + (i - 1.5) * w, [salida[k][met] for k in keys], w, label=labs[met],
               color=cols[met], edgecolor="black", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([etiq[k] for k in keys])
    ax.set_ylabel("Throughput medio T (EUR)")
    ax.set_title("PPO pro (warm-start por imitación + inferencia enmascarada) frente al vanilla")
    ax.legend(ncol=4, fontsize=8.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "rig_ppo_pro.png"), dpi=150); plt.close()
    print("[OK] rig_ppo_pro.png", flush=True)


if __name__ == "__main__":
    main()

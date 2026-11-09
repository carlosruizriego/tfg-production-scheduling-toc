# -*- coding: utf-8 -*-
"""
PPO GUIADO POR LA TOC (reward shaping con T/CU)
===============================================
Hipotesis: el bajo rendimiento del PPO se debe a la recompensa TERMINAL escasa
(solo al final del episodio), que dificulta la exploracion. Se inyecta la logica
de la TOC como senal densa: en cada paso, ademas del throughput terminal, el agente
recibe una recompensa intermedia proporcional al T/CU (margen por unidad de tiempo
de cuello) de la orden que elige. Es decir, se usa la TOC para ENSENAR a la IA.

Compara, en los 4 casos y con varias semillas, el PPO-vanilla frente al PPO-TOC,
ambos evaluados sobre el mismo simulador y las mismas 30 instancias. La unica
diferencia entre ambos es la recompensa de entrenamiento (el espacio de observacion
y la red son identicos). Reanudable y con wake-lock (Modern Standby).

Salida: results/resultados_ppo_toc.json y figura rig_ppo_toc.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import PPO

import estudio_c1, estudio_c2, estudio_c3, estudio_c4

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
FIGDIR = os.path.join(_BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_ppo_toc.json")

SEEDS = [0, 1, 2]
SHAPE_W = 0.5         # peso del termino TOC (cumulado max ~SHAPE_W frente a terminal ~1)
HP = dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
          gamma=1.0, ent_coef=0.01)


def _wake_lock(on=True):
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 if on else 0))
        print(f"[wake-lock] {'activado' if on else 'liberado'}", flush=True)
    except Exception as e:
        print(f"[wake-lock] no disponible: {e}", flush=True)


def _t_c1(o, m):
    return sum(op.duracion for op in o.operaciones if op.maquina == m)


def _t_c2(o, m):
    return sum(t for mm, t in o.ruta if mm == m)


CASES = [
    dict(key="c1", env=estudio_c1.SeqConstructEnv, sched="scheduled",
         gen=lambda sd: estudio_c1.generar_instancia(seed=sd),
         metodo_ppo=estudio_c1.metodo_ppo, seeds_test=estudio_c1.SEEDS_TEST,
         steps=120_000, n_maq=estudio_c1.N_MACHINES, t=_t_c1,
         margen=lambda o: o.valor_venta - o.coste_material),
    dict(key="c2", env=estudio_c2.SeqEnv, sched="sched",
         gen=lambda sd: estudio_c2.generar_ordenes_c2(seed=sd),
         metodo_ppo=estudio_c2.metodo_ppo, seeds_test=estudio_c2.SEEDS_TEST,
         steps=150_000, n_maq=estudio_c2.N_MACHINES_C2, t=_t_c2,
         margen=lambda o: o.margen),
    dict(key="c3", env=estudio_c3.SeqEnv, sched="sched", gen=estudio_c3.gen,
         metodo_ppo=estudio_c3.metodo_ppo, seeds_test=estudio_c3.SEEDS_TEST,
         steps=150_000, n_maq=estudio_c3.N_MACHINES_C3,
         t=lambda o, m: o.tiempos[m], margen=lambda o: o.margen),
    dict(key="c4", env=estudio_c4.SeqEnv, sched="sched", gen=estudio_c4.gen,
         metodo_ppo=estudio_c4.metodo_ppo, seeds_test=estudio_c4.SEEDS_TEST,
         steps=160_000, n_maq=estudio_c4.N_MACHINES_C4,
         t=lambda o, m: o.tiempos[m], margen=lambda o: o.margen),
]


def make_toc_env(case):
    """Subclase del entorno canonico que anade recompensa intermedia por T/CU."""
    Base = case["env"]; sched_attr = case["sched"]
    tfn = case["t"]; mfn = case["margen"]; nmaq = case["n_maq"]; w = SHAPE_W

    class TOCEnv(Base):
        def _init_tcu(self):
            ords = self.ordenes
            carga = [sum(tfn(o, m) for o in ords) for m in range(nmaq)]
            cuello = int(np.argmax(carga))
            tcu = [float("inf") if tfn(o, cuello) <= 0 else mfn(o) / tfn(o, cuello)
                   for o in ords]
            fin = [x for x in tcu if x != float("inf")]
            mx = max(fin) if fin else 1.0
            self._tcun = [1.0 if x == float("inf") else x / mx for x in tcu]
            self._wstep = w / max(1, len(ords))

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            self._init_tcu()
            return obs, info

        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            sched = getattr(self, sched_attr)
            if sched and hasattr(self, "_tcun"):
                r = r + self._wstep * self._tcun[sched[-1]]
            return obs, r, term, trunc, info

    return TOCEnv


def train_eval(case, seed, shaped):
    EnvCls = make_toc_env(case) if shaped else case["env"]
    env = EnvCls(train=True)
    env.rng = np.random.default_rng(seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, **HP)
    model.learn(total_timesteps=case["steps"])
    vals = [case["metodo_ppo"](model, case["gen"](sd))["throughput"]
            for sd in case["seeds_test"]]
    return [float(v) for v in vals]


def baselines(key):
    d = json.load(open(os.path.join(RESDIR, f"resultados_{key}.json")))
    r = d["resumen"]
    return {"FIFO": r["FIFO"]["media"], "NSGA-II": r["NSGA-II"]["media"]}


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print("  PPO GUIADO POR LA TOC  (vanilla vs reward shaping T/CU)", flush=True)
    print(f"  {len(SEEDS)} semillas x 4 casos x 2 versiones", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    prog = salida.get("_prog", {})
    for case in CASES:
        k = case["key"]
        prog.setdefault(k, {"vanilla": {}, "toc": {}})
        for version in ("vanilla", "toc"):
            for s in SEEDS:
                if str(s) in prog[k][version]:
                    continue
                vals = train_eval(case, s, shaped=(version == "toc"))
                prog[k][version][str(s)] = vals
                salida["_prog"] = prog
                json.dump(salida, open(PATH, "w"))
                print(f"   [{k}] {version} semilla {s}: media {np.mean(vals):.1f}", flush=True)
        # agregados por instancia (media sobre semillas) -> Wilcoxon pareado n=30
        van = np.mean([prog[k]["vanilla"][str(s)] for s in SEEDS], axis=0)
        toc = np.mean([prog[k]["toc"][str(s)] for s in SEEDS], axis=0)
        base = baselines(k)
        try:
            _, p = stats.wilcoxon(toc, van)
        except ValueError:
            p = 1.0
        salida[k] = {
            "ppo_vanilla": round(float(van.mean()), 1),
            "ppo_toc": round(float(toc.mean()), 1),
            "mejora_pct": round(100 * (toc.mean() - van.mean()) / van.mean(), 1),
            "wilcoxon_p": float(p),
            "fifo": round(base["FIFO"], 1), "nsga": round(base["NSGA-II"], 1),
        }
        json.dump(salida, open(PATH, "w"), indent=2)
        print(f"  -> [{k}] vanilla {van.mean():.1f}  TOC {toc.mean():.1f}  "
              f"({salida[k]['mejora_pct']:+.1f}%, p={p:.4f})  "
              f"| FIFO {base['FIFO']:.0f}  NSGA-II {base['NSGA-II']:.0f}", flush=True)
    print("\n[OK] resultados_ppo_toc.json", flush=True)
    if all(c["key"] in salida for c in CASES):
        figura(salida)
    _wake_lock(False)


def figura(salida):
    keys = ["c1", "c2", "c3", "c4"]
    etiq = {"c1": "Caso 1", "c2": "Caso 2", "c3": "Caso 3", "c4": "Caso 4"}
    cols = {"fifo": "#9aa0a6", "ppo_vanilla": "#E8743B", "ppo_toc": "#7B2FBF",
            "nsga": "#2E8B57"}
    labs = {"fifo": "FIFO", "ppo_vanilla": "PPO vanilla", "ppo_toc": "PPO guiado por TOC",
            "nsga": "Genético"}
    orden = ["fifo", "ppo_vanilla", "ppo_toc", "nsga"]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for i, met in enumerate(orden):
        ax.bar(x + (i - 1.5) * w, [salida[k][met] for k in keys], w, label=labs[met],
               color=cols[met], edgecolor="black", zorder=3)
    for k, xi in zip(keys, x):
        mp = salida[k]["mejora_pct"]
        ax.annotate(f"{mp:+.0f}%", (xi + 0.5 * w, max(salida[k]["ppo_vanilla"],
                    salida[k]["ppo_toc"])), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="#7B2FBF", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([etiq[k] for k in keys])
    ax.set_ylabel("Throughput medio T (EUR)")
    ax.set_title("El PPO guiado por la TOC frente al PPO estándar\n"
                 "(reward shaping con T/CU; % = mejora del PPO-TOC sobre el vanilla)")
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "rig_ppo_toc.png"), dpi=150)
    plt.close()
    print("[OK] rig_ppo_toc.png", flush=True)


if __name__ == "__main__":
    main()

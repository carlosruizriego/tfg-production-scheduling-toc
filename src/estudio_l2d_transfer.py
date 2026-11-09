# -*- coding: utf-8 -*-
"""
TRANSFERENCIA / generalizacion del agente L2D (Caso 5).
=======================================================
Pregunta: ¿la politica aprendida a NIVEL DE OPERACION generaliza a instancias de TAMANO
DISTINTO al de entrenamiento, sin reentrenar (zero-shot)? Es la propiedad que de verdad
justifica el aprendizaje por refuerzo frente a la metaheuristica (Zhang 2020): el agente se
entrena UNA vez y se aplica a cualquier instancia, mientras que el genetico re-optimiza
desde cero en cada una.

Diseno:
  - El agente se entrena SOLO sobre instancias de 20 trabajos (las del Caso 5).
  - Se evalua zero-shot sobre instancias de 20 (control, en-distribucion), 25 y 30 trabajos.
  - Posible gracias a que la cabeza por-trabajo es COMPARTIDA (Linear(H,1)): el logit del
    trabajo j sale solo de su embedding, luego la politica es agnostica al numero de trabajos.
    Se fija un NJ_MAX/N_PAD_MAX comun; en entrenamiento los trabajos 20..NJ_MAX-1 van siempre
    enmascarados, y en evaluacion sobre 25/30 trabajos se activan.
  - Baselines por tamano (re-optimizan en cada instancia): regla voraz op-a-op y genetico
    op-a-op SEMBRADO (el bien calibrado de la Tabla 8.3).
  - Demostracion con 1 semilla de entrenamiento (la robustez multisemilla del agente ya se
    establece en el apartado 8.3); honesto al respecto.
Reanudable (checkpoint del modelo). wake-lock. Salida: resultados_l2d_transfer.json,
rig_l2d_transfer.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch as th
from gymnasium import spaces
import gymnasium as gym
from scipy import stats
from stable_baselines3 import PPO

import gnn_grafo as G
import l2d_c5 as L
import planta_caso5 as P5
from estudio_l2d_rl import (GNNExtractorL2D, FL2D, obs_dim_l2d, construir_grafo_l2d,
                            bc, warm_critic, HP)
from estudio_ppo_toc import _wake_lock

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results"); FIGDIR = os.path.join(_BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_l2d_transfer.json")
CKPT = os.path.join(RESDIR, "l2d_transfer_model")     # .zip lo anade SB3

NJ_MAX = 30          # capacidad maxima de trabajos del modelo (>= mayor instancia de test)
N_PAD_MAX = 192      # 30 trabajos x 6 maquinas = 180 operaciones <= 192
TRAIN_NJ = 20        # tamano de entrenamiento (el del Caso 5)
TEST_SIZES = [20, 25, 30]
STEPS = 80_000
SEED = 0
N_DEMOS = 60
K_SEARCH = 32


def cfg_T():
    c = L.cfg_c5(); c = dict(c); c["NJOBS"] = NJ_MAX; c["N_PAD"] = N_PAD_MAX
    return c


def gen_n(seed, n_jobs):
    pool = (list(P5.FAMILIAS) * (n_jobs // len(P5.FAMILIAS) + 1))[:n_jobs]
    return P5.generar_ordenes_c5(pool, seed)


class TransferEnv(gym.Env):
    """Entorno L2D op-a-op con NJ_MAX/N_PAD_MAX fijos; admite instancias de <= NJ_MAX trabajos."""
    def __init__(self, train=True, train_nj=TRAIN_NJ):
        super().__init__()
        self.cfg = cfg_T(); self.NJ = NJ_MAX; self.n_maq = self.cfg["n_maq"]
        self.SETUP = self.cfg["setup_por_maq"]; self.CUELLOS = self.cfg["cuellos"]
        self.H = self.cfg["horizonte"]; self.NORM = self.cfg["norm"]
        self.max_setup = self.cfg["max_setup"]
        self.OBS = obs_dim_l2d(self.cfg)
        self.observation_space = spaces.Box(-5.0, 5.0, (self.OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NJ)
        self.train = train; self.train_nj = train_nj
        self.rng = np.random.default_rng(0); self.ordenes_fijas = None
        self._reset_interno()

    def fijar_instancia(self, o):
        self.ordenes_fijas = o

    def _reset_interno(self):
        if self.train or self.ordenes_fijas is None:
            self.ordenes = gen_n(int(self.rng.integers(0, 1_000_000)), self.train_nj)
        else:
            self.ordenes = self.ordenes_fijas
        self.nj = len(self.ordenes)                       # trabajos REALES de esta instancia
        self.st = construir_grafo_l2d(self.ordenes, self.cfg)
        self.job_pos = [0] * self.nj; self.job_free = [0.0] * self.nj
        self.maq_free = [0.0] * self.n_maq
        self.last = {c: None for c in self.CUELLOS}; self.done_time = [None] * self.nj
        self.opstart = np.zeros(len(self.st["nodos"]), dtype=np.float32)
        self.n_done = 0; self.total_ops = sum(len(o.ruta) for o in self.ordenes)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_interno(); return self._obs(), {}

    def step(self, a):
        j = int(a)
        if j >= self.nj or self.job_pos[j] >= len(self.ordenes[j].ruta):
            j = next(i for i in range(self.nj) if self.job_pos[i] < len(self.ordenes[i].ruta))
        k = self.job_pos[j]; m, d = self.ordenes[j].ruta[k]
        s = max(self.job_free[j], self.maq_free[m]); setup = 0.0
        if m in self.CUELLOS:
            lf = self.last[m]
            if lf is not None and lf != self.ordenes[j].tipo:
                setup = self.SETUP[m][lf][self.ordenes[j].tipo]
            self.last[m] = self.ordenes[j].tipo
        ini = s + setup; fin = ini + d
        self.opstart[self.st["node_of"][(j, k)]] = ini
        self.maq_free[m] = fin; self.job_free[j] = fin; self.job_pos[j] += 1
        self.n_done += 1
        r = 0.0
        if self.job_pos[j] == len(self.ordenes[j].ruta):
            self.done_time[j] = fin
            if fin <= self.H:
                r = self.ordenes[j].margen / self.NORM
        term = (self.n_done == self.total_ops)
        return self._obs(), r, term, False, {}

    def throughput(self):
        return sum(o.margen for jj, o in enumerate(self.ordenes)
                   if self.done_time[jj] is not None and self.done_time[jj] <= self.H)

    def _obs(self):
        st = self.st; X = st["X0_14"].copy(); H = self.H
        for idx, (j, k, m, d) in enumerate(st["nodeinfo"]):
            jp = self.job_pos[j]
            if k < jp:
                est = self.opstart[idx]; X[idx, 9] = 1.0
            elif k == jp:
                est = max(self.maq_free[m], self.job_free[j])
                if m in self.CUELLOS and self.last[m] is not None and self.last[m] != self.ordenes[j].tipo:
                    su = self.SETUP[m][self.last[m]][self.ordenes[j].tipo]; est += su
                    X[idx, 13] = su / self.max_setup
            else:
                est = self.job_free[j]
            X[idx, 10] = min(est / H, 5.0); X[idx, 11] = min((est + d) / H, 5.0)
            X[idx, 12] = float(np.clip((H - (est + d)) / H, -1, 1))
        # mascara de NJ_MAX trabajos: ausentes (j>=nj) o completados -> 1 (enmascarado)
        jm = np.ones(self.NJ, dtype=np.float32)
        for j in range(self.nj):
            jm[j] = 1.0 if self.job_pos[j] >= len(self.ordenes[j].ruta) else 0.0
        return np.concatenate([X.ravel(), st["Cn"].ravel(), st["Dn"].ravel(),
                               st["J2O"].ravel(), jm]).astype(np.float32)


def build():
    cfg = cfg_T(); OBS = obs_dim_l2d(cfg)
    Pol = G.make_masked_policy(OBS - cfg["NJOBS"], cfg["NJOBS"])
    pk = dict(features_extractor_class=GNNExtractorL2D,
              features_extractor_kwargs=dict(N=cfg["N_PAD"], NJ=cfg["NJOBS"], H=64, Lc=3),
              net_arch=dict(pi=[], vf=[64]))
    return cfg, Pol, pk


def construir_demos():
    cfg = cfg_T(); O, A, R = [], [], []
    for sd in range(300, 300 + N_DEMOS):
        ordenes = gen_n(sd, TRAIN_NJ)
        exp = L.greedy_stream(ordenes, cfg)
        e = TransferEnv(train=False); e.fijar_instancia(ordenes); o, _ = e.reset()
        ep_o, ep_a, ep_r = [], [], []
        for a in exp:
            ep_o.append(o.astype(np.float32)); ep_a.append(int(a))
            o, r, t, tr, _ = e.step(int(a)); ep_r.append(r)
            if t or tr:
                break
        rtg = np.cumsum(ep_r[::-1])[::-1]
        O += ep_o; A += ep_a; R += list(rtg)
    return np.array(O, np.float32), np.array(A, np.int64), np.array(R, np.float32)


def eval_agent(model, size, search=0):
    out = []
    for sd in L.SEEDS_TEST:
        ordenes = gen_n(sd, size); best = -1.0
        for _ in range(max(1, search)):
            e = TransferEnv(train=False); e.fijar_instancia(ordenes); o, _ = e.reset(); done = False
            while not done:
                ot, _ = model.policy.obs_to_tensor(o)
                dist = model.policy.get_distribution(ot)
                a = int(dist.sample().item()) if search else int(dist.distribution.logits.argmax().item())
                o, _, t, tr, _ = e.step(a); done = t or tr
            best = max(best, e.throughput())
        out.append(best)
    return np.array(out, dtype=float)


def baselines_size(size):
    cfg = cfg_T(); gr, ga = [], []
    for sd in L.SEEDS_TEST:
        o = gen_n(sd, size)
        gr.append(L.greedy_op(o, cfg)); ga.append(L.ga_op(o, cfg, seed=sd))
    return np.array(gr), np.array(ga)


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print("  TRANSFERENCIA DEL AGENTE L2D (entrena 20 -> evalua 20/25/30)", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    cfg, Pol, pk = build()

    if os.path.exists(CKPT + ".zip"):
        print("[ckpt] modelo encontrado, se recarga (sin reentrenar)", flush=True)
        model = PPO.load(CKPT, env=TransferEnv(train=True))
    else:
        print("[demos] greedy op-a-op sobre 20 trabajos (warm-start)...", flush=True)
        O, A, Rr = construir_demos()
        env = TransferEnv(train=True); env.rng = np.random.default_rng(SEED)
        model = PPO(Pol, env, verbose=0, seed=SEED, policy_kwargs=pk, **HP)
        print("[bc] clonacion del experto...", flush=True)
        bc(model, O, A, 30, SEED)
        warm_critic(model, O, Rr)
        print(f"[ppo] afinado {STEPS} pasos...", flush=True)
        model.learn(total_timesteps=STEPS)
        model.save(CKPT)
        print("[ckpt] modelo guardado", flush=True)

    res = salida.get("por_tamano", {})
    for size in TEST_SIZES:
        if str(size) in res:
            continue
        gr, ga = baselines_size(size)
        rl_g = eval_agent(model, size, search=0)
        rl_s = eval_agent(model, size, search=K_SEARCH)
        try:
            _, p_ga = stats.wilcoxon(rl_s, ga)
        except ValueError:
            p_ga = 1.0
        try:
            _, p_gr = stats.wilcoxon(rl_s, gr)
        except ValueError:
            p_gr = 1.0
        res[str(size)] = {
            "greedy": float(gr.mean()), "ga": float(ga.mean()),
            "rl_greedy": float(rl_g.mean()), "rl_search": float(rl_s.mean()),
            "p_search_vs_ga": float(p_ga), "p_search_vs_greedy": float(p_gr),
            "rl_search_raw": rl_s.tolist(), "ga_raw": ga.tolist(), "greedy_raw": gr.tolist()}
        salida["por_tamano"] = res
        salida["meta"] = dict(train_nj=TRAIN_NJ, steps=STEPS, seed=SEED, k_search=K_SEARCH,
                              nj_max=NJ_MAX, n_pad_max=N_PAD_MAX)
        json.dump(salida, open(PATH, "w"), indent=2)
        marca = "EN-DISTRIBUCION" if size == TRAIN_NJ else "ZERO-SHOT"
        print(f"  [{size} trab. {marca}] greedy {gr.mean():.0f}  GA {ga.mean():.0f}  "
              f"RL-greedy {rl_g.mean():.0f}  RL+busqueda {rl_s.mean():.0f}  "
              f"(p vs GA={p_ga:.3f}, p vs voraz={p_gr:.3f})", flush=True)
    figura(res)
    print("\n[OK] resultados_l2d_transfer.json", flush=True)
    _wake_lock(False)


def figura(res):
    sizes = [s for s in TEST_SIZES if str(s) in res]
    x = np.arange(len(sizes)); w = 0.2
    series = [("greedy", "#C28F2C", "Regla voraz (re-optimiza)"),
              ("ga", "#7B5BA6", "Genético op-a-op (re-optimiza)"),
              ("rl_search", "#C0392B", "Agente L2D + búsqueda (zero-shot)")]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, (k, c, lab) in enumerate(series):
        ax.bar(x + (i - 1) * w, [res[str(s)][k] for s in sizes], w, label=lab, color=c,
               edgecolor="black", zorder=3)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.text(0, ax.get_ylim()[1] * 0.02 if False else 0, "")
    ax.set_xticks(x); ax.set_xticklabels([f"{s} trabajos" + ("\n(entrenamiento)" if s == TRAIN_NJ
                                          else "\n(zero-shot)") for s in sizes])
    ax.set_ylabel("Throughput medio T (EUR)")
    ax.set_title("Transferencia del agente L2D: entrenado en 20 trabajos, evaluado sin reentrenar")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "rig_l2d_transfer.png"), dpi=150); plt.close()
    print("[OK] rig_l2d_transfer.png", flush=True)


if __name__ == "__main__":
    main()

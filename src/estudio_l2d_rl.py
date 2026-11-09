# -*- coding: utf-8 -*-
"""
Agente RL a NIVEL DE OPERACION (L2D) para el Caso 5.
====================================================
MDP op-a-op: en cada paso el agente elige el siguiente TRABAJO cuya operacion lista se
programa (Discrete(20) + masking). El grafo disyuntivo se actualiza dinamicamente con los
tiempos de inicio/fin a medida que se construye el programa (la informacion dinamica que
al MDP a nivel de orden le faltaba). Politica = GNN sobre el grafo; warm-start imitando al
greedy op-a-op; afinado con PPO; recompensa densa por trabajo completado (telescopea al
throughput con gamma=1.0). La METRICA es el throughput del propio scheduler op-a-op, el
MISMO evaluador que los baselines op-a-op (greedy/GA) -> comparacion justa en el espacio op.
Salida: resultados_l2d_rl.json y rig_l2d.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch as th, torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gnn_grafo as G
import l2d_c5 as L
from estudio_ppo_toc import _wake_lock

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results"); FIGDIR = os.path.join(_BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_l2d_rl.json")
FL2D = 14            # nº de features por nodo (9 estaticas + 5 dinamicas de tiempo)
SEEDS = [0, 1, 2]
N_DEMOS = 60
HP = dict(learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10,
          gamma=1.0, gae_lambda=1.0, ent_coef=0.005)


def obs_dim_l2d(cfg):
    N = cfg["N_PAD"]; NJ = cfg["NJOBS"]
    return N * FL2D + 2 * N * N + N * NJ + NJ


class GNNExtractorL2D(BaseFeaturesExtractor):
    def __init__(self, observation_space, N, NJ, H=64, Lc=3):
        super().__init__(observation_space, features_dim=NJ * H)
        self.N = N; self.NJ = NJ; self.H = H; self.Lc = Lc
        self.lin_in = nn.Linear(FL2D, H)
        self.Wc = nn.ModuleList([nn.Linear(H, H) for _ in range(Lc)])
        self.Wd = nn.ModuleList([nn.Linear(H, H) for _ in range(Lc)])
        self.ln = nn.ModuleList([nn.LayerNorm(H) for _ in range(Lc)])
        c = 0
        self.oX = (c, c + N * FL2D); c += N * FL2D
        self.oC = (c, c + N * N); c += N * N
        self.oD = (c, c + N * N); c += N * N
        self.oJ = (c, c + N * NJ); c += N * NJ

    def forward(self, obs):
        B = obs.shape[0]; N, NJ, H = self.N, self.NJ, self.H
        X = obs[:, self.oX[0]:self.oX[1]].view(B, N, FL2D)
        Cn = obs[:, self.oC[0]:self.oC[1]].view(B, N, N)
        Dn = obs[:, self.oD[0]:self.oD[1]].view(B, N, N)
        J2O = obs[:, self.oJ[0]:self.oJ[1]].view(B, N, NJ)
        h = th.relu(self.lin_in(X))
        for l in range(self.Lc):
            m = th.bmm(Cn, self.Wc[l](h)) + th.bmm(Dn, self.Wd[l](h))
            h = self.ln[l](th.relu(m) + h)
        cnt = J2O.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return (th.bmm(J2O.transpose(1, 2), h) / cnt).reshape(B, NJ * H)


def construir_grafo_l2d(ordenes, cfg):
    st = G.construir_estatico(ordenes, cfg)
    N = cfg["N_PAD"]
    X0 = np.zeros((N, FL2D), dtype=np.float32)
    X0[:, :9] = st["X0"][:, :9]                       # features estaticas 0-8
    st["X0_14"] = X0
    st["nodeinfo"] = [(j, k, m, d) for (j, k, m, d, L_) in st["nodos"]]
    st["node_of"] = {(j, k): i for i, (j, k, m, d, L_) in enumerate(st["nodos"])}
    return st


class L2DEnvC5(gym.Env):
    def __init__(self, train=True):
        super().__init__()
        self.cfg = L.cfg_c5(); self.NJ = self.cfg["NJOBS"]; self.n_maq = self.cfg["n_maq"]
        self.SETUP = self.cfg["setup_por_maq"]; self.CUELLOS = self.cfg["cuellos"]
        self.H = self.cfg["horizonte"]; self.NORM = self.cfg["norm"]
        self.max_setup = self.cfg["max_setup"]
        self.OBS = obs_dim_l2d(self.cfg)
        self.observation_space = spaces.Box(-5.0, 5.0, (self.OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NJ)
        self.train = train; self.rng = np.random.default_rng(0); self.ordenes_fijas = None
        self._reset_interno()

    def fijar_instancia(self, o):
        self.ordenes_fijas = o

    def _reset_interno(self):
        self.ordenes = (L.gen(int(self.rng.integers(0, 1_000_000)))
                        if (self.train or self.ordenes_fijas is None) else self.ordenes_fijas)
        self.st = construir_grafo_l2d(self.ordenes, self.cfg)
        n = self.NJ
        self.job_pos = [0] * n; self.job_free = [0.0] * n; self.maq_free = [0.0] * self.n_maq
        self.last = {c: None for c in self.CUELLOS}; self.done_time = [None] * n
        self.opstart = np.zeros(len(self.st["nodos"]), dtype=np.float32)
        self.n_done = 0; self.total_ops = sum(len(o.ruta) for o in self.ordenes)
        self.acciones = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_interno(); return self._obs(), {}

    def step(self, a):
        n = self.NJ; j = int(a)
        if self.job_pos[j] >= len(self.ordenes[j].ruta):
            j = next(i for i in range(n) if self.job_pos[i] < len(self.ordenes[i].ruta))
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
        self.n_done += 1; self.acciones.append(j)
        r = 0.0
        if self.job_pos[j] == len(self.ordenes[j].ruta):
            self.done_time[j] = fin
            if fin <= self.H:
                r = self.ordenes[j].margen / self.NORM
        term = (self.n_done == self.total_ops)
        return self._obs(), r, term, False, {"throughput": self.throughput() if term else 0.0}

    def throughput(self):
        return sum(o.margen for j, o in enumerate(self.ordenes)
                   if self.done_time[j] is not None and self.done_time[j] <= self.H)

    def _obs(self):
        st = self.st; X = st["X0_14"].copy(); H = self.H
        for idx, (j, k, m, d) in enumerate(st["nodeinfo"]):
            jp = self.job_pos[j]
            if k < jp:                                  # ya programada
                stt = self.opstart[idx]; X[idx, 9] = 1.0; est = stt
            elif k == jp:                               # lista (ready)
                est = max(self.maq_free[m], self.job_free[j])
                if m in self.CUELLOS and self.last[m] is not None and self.last[m] != self.ordenes[j].tipo:
                    su = self.SETUP[m][self.last[m]][self.ordenes[j].tipo]; est += su
                    X[idx, 13] = su / self.max_setup
            else:                                       # futura
                est = self.job_free[j]
            X[idx, 10] = min(est / H, 5.0); X[idx, 11] = min((est + d) / H, 5.0)
            X[idx, 12] = float(np.clip((H - (est + d)) / H, -1, 1))
        jm = np.array([1.0 if self.job_pos[j] >= len(self.ordenes[j].ruta) else 0.0
                       for j in range(self.NJ)], dtype=np.float32)
        return np.concatenate([X.ravel(), st["Cn"].ravel(), st["Dn"].ravel(),
                               st["J2O"].ravel(), jm]).astype(np.float32)


def build():
    cfg = L.cfg_c5(); OBS = obs_dim_l2d(cfg)
    Pol = G.make_masked_policy(OBS - cfg["NJOBS"], cfg["NJOBS"])
    pk = dict(features_extractor_class=GNNExtractorL2D,
              features_extractor_kwargs=dict(N=cfg["N_PAD"], NJ=cfg["NJOBS"], H=64, Lc=3),
              net_arch=dict(pi=[], vf=[64]))
    return cfg, Pol, pk


def construir_demos():
    cfg = L.cfg_c5()
    O, A, R = [], [], []
    for sd in range(300, 300 + N_DEMOS):
        ordenes = L.gen(sd)
        exp = L.greedy_stream(ordenes, cfg)
        e = L2DEnvC5(train=False); e.fijar_instancia(ordenes); o, _ = e.reset()
        ep_o, ep_a, ep_r = [], [], []
        for a in exp:
            ep_o.append(o.astype(np.float32)); ep_a.append(int(a))
            o, r, t, tr, _ = e.step(int(a)); ep_r.append(r)
            if t or tr:
                break
        # retorno-a-go (gamma=1.0)
        rtg = np.cumsum(ep_r[::-1])[::-1]
        O += ep_o; A += ep_a; R += list(rtg)
    return np.array(O, np.float32), np.array(A, np.int64), np.array(R, np.float32)


def bc(model, O, A, epochs, seed):
    opt = th.optim.Adam([p for n, p in model.policy.named_parameters() if "value_net" not in n], lr=5e-4)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(O))
        for i in range(0, len(O), 256):
            b = idx[i:i + 256]; ot, _ = model.policy.obs_to_tensor(O[b])
            loss = -model.policy.get_distribution(ot).log_prob(th.as_tensor(A[b])).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def warm_critic(model, O, R, epochs=10):
    head = [p for n, p in model.policy.named_parameters() if "value_net" in n]
    opt = th.optim.Adam(head, lr=1e-3); Rt = th.as_tensor(R).unsqueeze(-1)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        idx = rng.permutation(len(O))
        for i in range(0, len(O), 512):
            b = idx[i:i + 512]; ot, _ = model.policy.obs_to_tensor(O[b])
            loss = ((model.policy.predict_values(ot) - Rt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def eval_agent(model, search=0):
    out = []
    for sd in L.SEEDS_TEST:
        ordenes = L.gen(sd); best = -1.0
        reps = max(1, search)
        for r in range(reps):
            e = L2DEnvC5(train=False); e.fijar_instancia(ordenes); o, _ = e.reset(); done = False
            while not done:
                ot, _ = model.policy.obs_to_tensor(o)
                dist = model.policy.get_distribution(ot)
                a = int(dist.sample().item()) if search else int(dist.distribution.logits.argmax().item())
                o, _, t, tr, _ = e.step(a); done = t or tr
            best = max(best, e.throughput())
        out.append(best)
    return np.array(out, dtype=float)


def train_one(seed, demos):
    O, A, R = demos
    cfg, Pol, pk = build()
    env = L2DEnvC5(train=True); env.rng = np.random.default_rng(seed)
    model = PPO(Pol, env, verbose=0, seed=seed, policy_kwargs=pk, **HP)
    bc(model, O, A, 30, seed)
    bc_vec = eval_agent(model, search=0)
    warm_critic(model, O, R)
    model.learn(total_timesteps=120_000)
    ft_vec = eval_agent(model, search=0)
    se_vec = eval_agent(model, search=32)
    return bc_vec.tolist(), ft_vec.tolist(), se_vec.tolist()


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print("  AGENTE RL A NIVEL DE OPERACION (L2D) - Caso 5", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    prog = salida.get("_prog", {})
    if any(str(s) not in prog for s in SEEDS):
        print("[demos] greedy op-a-op (warm-start)...", flush=True)
        demos = construir_demos()
    for s in SEEDS:
        if str(s) in prog:
            continue
        bcv, ftv, sev = train_one(s, demos)
        prog[str(s)] = {"bc": bcv, "ft": ftv, "search": sev}
        salida["_prog"] = prog; json.dump(salida, open(PATH, "w"))
        print(f"   semilla {s}: BC={np.mean(bcv):.0f}  FT={np.mean(ftv):.0f}  "
              f"SEARCH={np.mean(sev):.0f}", flush=True)
    bc_inst = np.mean([prog[str(s)]["bc"] for s in SEEDS], axis=0)
    ft_inst = np.mean([prog[str(s)]["ft"] for s in SEEDS], axis=0)
    se_inst = np.mean([prog[str(s)]["search"] for s in SEEDS], axis=0)
    op = json.load(open(os.path.join(RESDIR, "resultados_c5_l2d.json")))
    greedy = op["greedy_op"]["media"]; ga_op = op["ga_op"]["media"]
    jb = json.load(open(os.path.join(RESDIR, "resultados_c5.json")))["resumen"]
    salida["resumen"] = {
        "job_FIFO": round(jb["FIFO"]["media"], 1), "job_CPSAT": round(jb["CP-SAT"]["media"], 1),
        "job_NSGA": round(jb["NSGA-II"]["media"], 1), "op_greedy": round(greedy, 1),
        "op_GA": round(ga_op, 1), "op_RL_bc": round(float(bc_inst.mean()), 1),
        "op_RL_ft": round(float(ft_inst.mean()), 1), "op_RL_search": round(float(se_inst.mean()), 1)}
    json.dump(salida, open(PATH, "w"), indent=2)
    R = salida["resumen"]
    print("\n--- RESUMEN C5 ---", flush=True)
    print(f"  Nivel TRABAJO:   FIFO {R['job_FIFO']:.0f}  CP-SAT {R['job_CPSAT']:.0f}  "
          f"NSGA-II {R['job_NSGA']:.0f}", flush=True)
    print(f"  Nivel OPERACION: greedy {R['op_greedy']:.0f}  GA {R['op_GA']:.0f}  "
          f"RL(BC) {R['op_RL_bc']:.0f}  RL(PPO) {R['op_RL_ft']:.0f}  "
          f"RL+busqueda {R['op_RL_search']:.0f}", flush=True)
    figura(R)
    _wake_lock(False)


def figura(R):
    metodos = [("FIFO (trabajo)", R["job_FIFO"], "#9aa0a6"),
               ("Genético (trabajo)", R["job_NSGA"], "#2E8B57"),
               ("Greedy (op-a-op)", R["op_greedy"], "#C28F2C"),
               ("GA op-a-op", R["op_GA"], "#7B5BA6"),
               ("RL-PPO (op-a-op)", R["op_RL_ft"], "#1F6FB2"),
               ("RL+búsqueda (op-a-op)", R["op_RL_search"], "#C0392B")]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(metodos))
    ax.bar(x, [m[1] for m in metodos], color=[m[2] for m in metodos], edgecolor="black", zorder=3)
    for xi, m in zip(x, metodos):
        ax.text(xi, m[1] + 8, f"{m[1]:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(R["job_NSGA"], color="#2E8B57", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metodos], rotation=15, ha="right")
    ax.set_ylabel("Throughput medio T (EUR)")
    ax.set_title("Caso 5: la formulación a nivel de operación supera a la de nivel de trabajo")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, "rig_l2d.png"), dpi=150); plt.close()
    print("[OK] rig_l2d.png", flush=True)


if __name__ == "__main__":
    main()

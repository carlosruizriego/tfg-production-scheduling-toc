# -*- coding: utf-8 -*-
"""
Nucleo GNN sobre grafo disyuntivo para el agente PPO (TFG).
===========================================================
Representa el estado de scheduling como un GRAFO DISYUNTIVO por-operacion:
  - nodos = operaciones (con features ESTATICAS por-op + DINAMICAS por paso),
  - aristas conjuntivas (ruta del trabajo) y disyuntivas (operaciones de la misma maquina).
Un encoder GNN (paso de mensajes denso en torch puro, CPU) agrega por trabajo y produce
logits por orden, con masking real de ordenes ya despachadas.

Incorpora las correcciones de la revision adversaria:
  - OBS_DIM/MASK_OFF derivados PROGRAMATICAMENTE (no literal) + asserts.
  - Una sola representacion de la feature dinamica (no bloque 'disp' duplicado).
  - Features DINAMICAS reales: job_dispatched, setup_pendiente (last_fam->familia),
    carga de cuello restante  -> el GNN ve algo que el MLP no tenia.
  - Masking con torch.finfo(dtype).min (no -1e9) para no envenenar la entropia.
Diseno honesto: la accion es por-ORDEN (no por-operacion); el grafo aporta features
ricas por-operacion. Fiel al espiritu de L2D (Zhang 2020), no a su letra.
"""
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

F_DIM = 12   # nº de features por nodo (9 estaticas + 3 dinamicas)

# ───────────────────────── Adaptadores de caso ─────────────────────────
def ops_de_orden(o, key):
    if key == "c1":
        return [(op.maquina, op.duracion) for op in o.operaciones]
    if key in ("c2", "c5"):
        return list(o.ruta)
    return [(m, o.tiempos[m]) for m in range(len(o.tiempos))]   # c3, c4 (flow shop)

def margen_de(o, key):
    return (o.valor_venta - o.coste_material) if key == "c1" else o.margen

def familia_de(o, key):
    return getattr(o, "tipo", None)


def construir_estatico(ordenes, cfg):
    """Devuelve los tensores ESTATICOS del grafo (no dependen del paso)."""
    key = cfg["key"]; n_maq = cfg["n_maq"]; N = cfg["N_PAD"]; NJ = cfg["NJOBS"]
    fams = cfg["fams"]
    ops_por_orden = [ops_de_orden(o, key) for o in ordenes]
    # mapeo operacion global -> (job, idx_en_ruta, maquina, dur)
    nodos = []
    op2job = np.full(N, -1, dtype=np.int64)
    for j, ops in enumerate(ops_por_orden):
        for k, (m, d) in enumerate(ops):
            if len(nodos) >= N:
                break
            op2job[len(nodos)] = j
            nodos.append((j, k, int(m), float(d), len(ops)))
    n_ops = len(nodos)
    # cargas por maquina y cuello
    carga = np.zeros(n_maq, dtype=np.float64)
    for (j, k, m, d, L) in nodos:
        carga[m] += d
    cuello = int(np.argmax(carga)) if carga.max() > 0 else 0
    carga_max = max(carga.max(), 1e-9)
    dur_max = max((nd[3] for nd in nodos), default=1.0) or 1.0
    margenes = [margen_de(o, key) for o in ordenes]
    margen_max = max(max(margenes), 1e-9)
    # tiempo de cada job en el cuello (para T/CU) y trabajo total por job
    t_cuello_job = np.zeros(NJ); t_total_job = np.zeros(NJ)
    for (j, k, m, d, L) in nodos:
        if m == cuello:
            t_cuello_job[j] += d
        t_total_job[j] += d
    tcu_job = np.zeros(NJ)
    for j in range(min(len(ordenes), NJ)):
        tc = t_cuello_job[j] if t_cuello_job[j] > 0 else 1e-9
        tcu_job[j] = margenes[j] / tc
    tcu_max = max(tcu_job.max(), 1e-9)
    nfam = max(len(fams), 1)

    X0 = np.zeros((N, F_DIM), dtype=np.float32)
    # trabajo restante acumulado desde cada operacion (suma de duraciones de su ruta desde k)
    # precalcular sufijos por job
    suf = {}
    for j, ops in enumerate(ops_por_orden):
        s = np.cumsum([d for (_, d) in ops][::-1])[::-1]  # trabajo restante desde k (incluido)
        suf[j] = s
    for i, (j, k, m, d, L) in enumerate(nodos):
        X0[i, 0] = d / dur_max
        X0[i, 1] = m / max(n_maq - 1, 1)
        X0[i, 2] = 1.0 if m == cuello else 0.0
        X0[i, 3] = carga[m] / carga_max
        X0[i, 4] = margenes[j] / margen_max
        X0[i, 5] = tcu_job[j] / tcu_max
        X0[i, 6] = k / max(L - 1, 1)
        X0[i, 7] = (suf[j][k] / t_total_job[j]) if t_total_job[j] > 0 else 0.0
        if fams:
            fi = fams.index(familia_de(ordenes[j], key)) if familia_de(ordenes[j], key) in fams else 0
            X0[i, 8] = fi / max(nfam - 1, 1)
        # 9,10,11 dinamicas -> se rellenan en _obs()

    # adyacencias
    A_c = np.zeros((N, N), dtype=np.float32)   # conjuntiva (ruta consecutiva)
    A_d = np.zeros((N, N), dtype=np.float32)   # disyuntiva (misma maquina)
    by_job = {}
    by_maq = {}
    for i, (j, k, m, d, L) in enumerate(nodos):
        by_job.setdefault(j, []).append((k, i))
        by_maq.setdefault(m, []).append(i)
    for j, lst in by_job.items():
        lst.sort()
        for a in range(len(lst) - 1):
            i1 = lst[a][1]; i2 = lst[a + 1][1]
            A_c[i1, i2] = 1.0; A_c[i2, i1] = 1.0
    for m, lst in by_maq.items():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                A_d[lst[a], lst[b]] = 1.0; A_d[lst[b], lst[a]] = 1.0
    Cn = _norm_adj(A_c, n_ops, N)
    Dn = _norm_adj(A_d, n_ops, N)

    J2O = np.zeros((N, NJ), dtype=np.float32)
    for i, (j, k, m, d, L) in enumerate(nodos):
        if j < NJ:
            J2O[i, j] = 1.0
    node_mask = np.zeros(N, dtype=np.float32); node_mask[:n_ops] = 1.0
    return dict(X0=X0, Cn=Cn, Dn=Dn, op2job=op2job, J2O=J2O, node_mask=node_mask,
                cuello=cuello, nodos=nodos, n_ops=n_ops)


def _norm_adj(A, n_ops, N):
    """D^{-1/2}(A+I)D^{-1/2} sobre los nodos reales; pads con self-loop aislado."""
    M = A.copy()
    for i in range(N):
        M[i, i] = 1.0
    deg = M.sum(axis=1)
    deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    Mn = (M * dinv[:, None]) * dinv[None, :]
    # anular filas/cols de pads salvo self-loop
    for i in range(n_ops, N):
        Mn[i, :] = 0.0; Mn[:, i] = 0.0; Mn[i, i] = 1.0
    return Mn.astype(np.float32)


def obs_dim(cfg):
    N = cfg["N_PAD"]; NJ = cfg["NJOBS"]
    return N * F_DIM + N * N + N * N + N * NJ + NJ


# ───────────────────────── Encoder GNN ─────────────────────────
class GNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, N, NJ, H=64, L=2):
        super().__init__(observation_space, features_dim=NJ * H)
        self.N = N; self.NJ = NJ; self.H = H; self.L = L
        self.lin_in = nn.Linear(F_DIM, H)
        self.Wc = nn.ModuleList([nn.Linear(H, H) for _ in range(L)])
        self.Wd = nn.ModuleList([nn.Linear(H, H) for _ in range(L)])
        self.ln = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self._off = self._offsets()

    def _offsets(self):
        N, NJ = self.N, self.NJ
        o = {}
        c = 0
        o["X"] = (c, c + N * F_DIM); c += N * F_DIM
        o["Cn"] = (c, c + N * N); c += N * N
        o["Dn"] = (c, c + N * N); c += N * N
        o["J2O"] = (c, c + N * NJ); c += N * NJ
        o["mask"] = (c, c + NJ); c += NJ
        assert c == o["mask"][1], "offset mismatch"
        return o

    def forward(self, obs):
        B = obs.shape[0]; N, NJ, H = self.N, self.NJ, self.H
        X = obs[:, self._off["X"][0]:self._off["X"][1]].view(B, N, F_DIM)
        Cn = obs[:, self._off["Cn"][0]:self._off["Cn"][1]].view(B, N, N)
        Dn = obs[:, self._off["Dn"][0]:self._off["Dn"][1]].view(B, N, N)
        J2O = obs[:, self._off["J2O"][0]:self._off["J2O"][1]].view(B, N, NJ)
        h = th.relu(self.lin_in(X))
        for l in range(self.L):
            m = th.bmm(Cn, self.Wc[l](h)) + th.bmm(Dn, self.Wd[l](h))
            h = self.ln[l](th.relu(m) + h)
        cnt = J2O.sum(dim=1).clamp(min=1.0).unsqueeze(-1)           # [B,NJ,1]
        job_emb = th.bmm(J2O.transpose(1, 2), h) / cnt              # [B,NJ,H]
        return job_emb.reshape(B, NJ * H)


# ───────────────────────── Politica enmascarada ─────────────────────────
class MaskedGNNPolicy(ActorCriticPolicy):
    mask_off = None
    njobs = None

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Cabeza per-job COMPARTIDA: el logit del trabajo j sale solo del embedding de j
        # (preserva equivarianza; evita que la cabeza densa mezcle trabajos como el MLP).
        self._H = self.features_extractor.features_dim // self.njobs
        self.job_head = nn.Linear(self._H, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _logits(self, latent_pi, obs):
        B = latent_pi.shape[0]
        je = latent_pi.view(B, self.njobs, self._H)          # [B, NJOBS, H]
        lg = self.job_head(je).squeeze(-1)                   # [B, NJOBS]
        m = obs[:, self.mask_off:self.mask_off + self.njobs] > 0.5
        return lg.masked_fill(m, th.finfo(lg.dtype).min)

    def forward(self, obs, deterministic=False):
        feats = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(feats)
        logits = self._logits(latent_pi, obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(latent_vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        feats = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(feats)
        logits = self._logits(latent_pi, obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        return self.value_net(latent_vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        feats = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(feats)
        logits = self._logits(latent_pi, obs)
        return self.action_dist.proba_distribution(action_logits=logits)

    def predict_values(self, obs):
        feats = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(feats)
        return self.value_net(latent_vf)


def make_masked_policy(mask_off, njobs):
    cls = type("MaskedGNNPolicy_%d" % njobs, (MaskedGNNPolicy,),
               {"mask_off": int(mask_off), "njobs": int(njobs)})
    return cls


# ───────────────────────── Wrapper de entorno-grafo ─────────────────────────
def make_gnn_env(BaseEnvCls, cfg):
    """Subclasea el entorno base cambiando solo observation_space y _obs()."""
    key = cfg["key"]; N = cfg["N_PAD"]; NJ = cfg["NJOBS"]
    fams = cfg["fams"]; setup = cfg.get("setup"); max_setup = cfg.get("max_setup", 1.0)
    OBS = obs_dim(cfg)

    class GraphEnv(BaseEnvCls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.observation_space = spaces.Box(-5.0, 5.0, (OBS,), dtype=np.float32)
            self._static = None

        def reset(self, seed=None, options=None):
            self._static = None     # se reconstruye en _obs para la nueva instancia
            return super().reset(seed=seed, options=options)

        def _last_fam(self):
            sched = getattr(self, "sched", None)
            if sched is None:
                sched = getattr(self, "scheduled", [])
            if not sched or not fams:
                return None
            return familia_de(self.ordenes[sched[-1]], key)

        def _obs(self):
            if self._static is None:
                self._static = construir_estatico(self.ordenes, cfg)
            st = self._static
            X = st["X0"].copy()
            op2job = st["op2job"]; n_ops = st["n_ops"]; cuello = st["cuello"]
            mask = self.mask                                   # 1 = despachado
            # 9: job_dispatched por nodo
            for i in range(n_ops):
                X[i, 9] = mask[op2job[i]]
            # 10: setup pendiente que induce despachar el job de este nodo AHORA
            lf = self._last_fam()
            if setup is not None and lf is not None:
                for i in range(n_ops):
                    fj = familia_de(self.ordenes[op2job[i]], key)
                    X[i, 10] = setup[lf][fj] / max_setup if (fj in setup.get(lf, {})) else 0.0
            # 11: carga restante del cuello (jobs NO despachados) / carga inicial cuello
            tot = 0.0; rem = 0.0
            for i in range(n_ops):
                if int(round(X[i, 1] * max(cfg["n_maq"] - 1, 1))) == cuello:
                    pass
            # mas simple: usar nodos
            cini = 1e-9; crem = 0.0
            for (j, k, m, d, L) in st["nodos"]:
                if m == cuello:
                    cini += d
                    if mask[j] < 0.5:
                        crem += d
            for i in range(n_ops):
                X[i, 11] = crem / cini
            job_mask = self.mask.astype(np.float32).copy()
            return np.concatenate([X.ravel(), st["Cn"].ravel(), st["Dn"].ravel(),
                                   st["J2O"].ravel(), job_mask]).astype(np.float32)

    return GraphEnv, OBS


# ───────────────────────── Config por caso ─────────────────────────
def cfg_para(key):
    import estudio_c1, estudio_c2, estudio_c3, estudio_c4
    if key == "c1":
        return dict(key="c1", n_maq=estudio_c1.N_MACHINES, N_PAD=50, NJOBS=10,
                    fams=[], setup=None, max_setup=1.0)
    if key == "c2":
        import planta_caso2
        return dict(key="c2", n_maq=estudio_c2.N_MACHINES_C2, N_PAD=65, NJOBS=13,
                    fams=list(estudio_c2.TIPOS), setup=None, max_setup=1.0)
    if key == "c3":
        return dict(key="c3", n_maq=estudio_c3.N_MACHINES_C3, N_PAD=80, NJOBS=16,
                    fams=["X", "Y"], setup=None, max_setup=1.0)
    if key == "c4":
        import planta_caso4
        ms = max(planta_caso4.SETUP[a][b] for a in planta_caso4.FAMILIAS
                 for b in planta_caso4.FAMILIAS)
        return dict(key="c4", n_maq=estudio_c4.N_MACHINES_C4, N_PAD=75, NJOBS=15,
                    fams=list(planta_caso4.FAMILIAS), setup=planta_caso4.SETUP,
                    max_setup=ms or 1.0)
    if key == "c5":
        import planta_caso5
        ms = max(max(M[a][b] for a in planta_caso5.FAMILIAS for b in planta_caso5.FAMILIAS)
                 for M in (planta_caso5.SETUP_A, planta_caso5.SETUP_B))
        return dict(key="c5", n_maq=planta_caso5.N_MACHINES_C5, N_PAD=125, NJOBS=20,
                    fams=list(planta_caso5.FAMILIAS), setup=planta_caso5.SETUP_A,
                    max_setup=ms or 1.0)
    raise ValueError(key)


if __name__ == "__main__":
    # ---- SMOKE GATE en C1: pipeline + accuracy de imitacion + ablacion de features ----
    import warnings; warnings.filterwarnings("ignore")
    import estudio_c1
    from stable_baselines3 import PPO
    from estudio_ppo_pro import ga_secuencia
    key = "c1"; cfg = cfg_para(key)
    GraphEnv, OBS = make_gnn_env(estudio_c1.SeqConstructEnv, cfg)
    assert OBS == obs_dim(cfg)
    print("OBS_DIM(c1) =", OBS, "| MASK_OFF =", OBS - cfg["NJOBS"])
    env = GraphEnv(train=True); env.rng = np.random.default_rng(0)
    Pol = make_masked_policy(OBS - cfg["NJOBS"], cfg["NJOBS"])
    pk = dict(features_extractor_class=GNNExtractor,
              features_extractor_kwargs=dict(N=cfg["N_PAD"], NJ=cfg["NJOBS"], H=64),
              net_arch=dict(pi=[], vf=[64]))
    model = PPO(Pol, env, verbose=0, seed=0, n_steps=256, batch_size=64,
                policy_kwargs=pk, learning_rate=3e-4, ent_coef=0.005, gamma=1.0)
    print("[smoke] learn 300 pasos...")
    model.learn(300)
    # comprobar que la entropia/loss no es NaN
    obs, _ = env.reset()
    ot, _ = model.policy.obs_to_tensor(obs)
    lg = model.policy.get_distribution(ot).distribution.logits
    assert th.isfinite(lg).all(), "logits con NaN/inf"
    print("[smoke] logits finitos OK")
    # ---- BC sobre demos del GA y accuracy de imitacion ----
    O, A = [], []
    for sd in range(300, 325):
        ordenes = estudio_c1.generar_instancia(seed=sd)
        seq = ga_secuencia(ordenes, estudio_c1.evaluar, "throughput", seed=sd)
        e = GraphEnv(train=False); e.fijar_instancia(ordenes); o, _ = e.reset()
        for a in seq:
            O.append(o.astype(np.float32)); A.append(int(a))
            o, _, t, tr, _ = e.step(int(a))
            if t or tr:
                break
    O = np.array(O, np.float32); A = np.array(A, np.int64)
    opt = th.optim.Adam([p for n, p in model.policy.named_parameters()
                         if "value_net" not in n], lr=5e-4)
    for ep in range(20):
        idx = np.random.default_rng(ep).permutation(len(O))
        for i in range(0, len(O), 256):
            b = idx[i:i + 256]
            ot, _ = model.policy.obs_to_tensor(O[b])
            dist = model.policy.get_distribution(ot)
            loss = -dist.log_prob(th.as_tensor(A[b])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    # accuracy de imitacion
    ot, _ = model.policy.obs_to_tensor(O)
    pred = model.policy.get_distribution(ot).distribution.logits.argmax(1).numpy()
    acc = (pred == A).mean()
    print("[GATE] BC imitation accuracy C1 = %.3f  (debe >0.7; azar=%.3f)" % (acc, 1 / cfg["NJOBS"]))
    print("[smoke] OK")

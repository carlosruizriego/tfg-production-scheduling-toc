# -*- coding: utf-8 -*-
"""
ESTUDIO RIGUROSO - CASO 4 (Flow Shop con tiempos de cambio en el cuello)
=======================================================================
Mismo protocolo que los casos anteriores:
  - Evaluador UNICO (PlantaC4): los 4 metodos se miden con el mismo simulador.
  - CP-SAT alineado a throughput, con los setups del cuello modelados mediante una
    restriccion de CIRCUITO (AddCircuit) que solo penaliza los cambios consecutivos.
  - NSGA-II (GA) y PPO buscan/evaluan secuencias en el mismo simulador.
  - 30 instancias (semillas), media +/- desv. y test de Wilcoxon.
El objetivo es maximizar el throughput dentro del horizonte; como los setups roban
capacidad al cuello, la buena secuencia AGRUPA familias para amortizarlos.
"""
import numpy as np, json, warnings, os
warnings.filterwarnings("ignore")
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
from ortools.sat.python import cp_model
from scipy import stats
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from planta_caso4 import (generar_ordenes_c4, PlantaC4, SETUP, SETUP_M,
                          N_MACHINES_C4, FAMILIAS, POOL_DEFAULT)

HORIZONTE = 320
N_TEST = 30
SEEDS_TEST = list(range(100, 100 + N_TEST))
NORD = len(POOL_DEFAULT)          # 15
MAX_SETUP = max(SETUP[a][b] for a in FAMILIAS for b in FAMILIAS)


def gen(seed):
    return generar_ordenes_c4(pool=POOL_DEFAULT[:], seed=seed)

def evaluar(ordenes, seq):
    return PlantaC4(ordenes, list(seq), HORIZONTE).ejecutar()

def metodo_fifo(ordenes):
    return evaluar(ordenes, list(range(len(ordenes))))

# ───────── CP-SAT: flow shop + setups en el cuello via circuito ─────────
def metodo_cpsat(ordenes, tiempo_limite=12.0):
    model = cp_model.CpModel()
    n = len(ordenes)
    H = int(sum(sum(o.tiempos) for o in ordenes)) + n * MAX_SETUP + 1
    task = {}
    for j, o in enumerate(ordenes):
        for m in range(N_MACHINES_C4):
            dur = max(1, int(round(o.tiempos[m])))
            s = model.NewIntVar(0, H, f"s_{j}_{m}")
            e = model.NewIntVar(0, H, f"e_{j}_{m}")
            iv = model.NewIntervalVar(s, dur, e, f"i_{j}_{m}")
            task[j, m] = (s, e, iv)
    # precedencias flow shop
    for j in range(n):
        for m in range(N_MACHINES_C4 - 1):
            model.Add(task[j, m + 1][0] >= task[j, m][1])
    # no solapamiento en todas las maquinas
    for m in range(N_MACHINES_C4):
        model.AddNoOverlap([task[j, m][2] for j in range(n)])
    # circuito en el cuello: secuencia de familias con setups consecutivos
    depot = n
    arcs = []
    lit = {}
    start_lit = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            b = model.NewBoolVar(f"arc_{i}_{j}")
            lit[i, j] = b
            arcs.append((i, j, b))
            su = SETUP[ordenes[i].tipo][ordenes[j].tipo]
            model.Add(task[j, SETUP_M][0] >= task[i, SETUP_M][1] + su).OnlyEnforceIf(b)
    for j in range(n):
        sb = model.NewBoolVar(f"start_{j}")
        start_lit[j] = sb
        arcs.append((depot, j, sb))
        arcs.append((j, depot, model.NewBoolVar(f"end_{j}")))
    model.AddCircuit(arcs)
    # done_j = orden j finalizada dentro del horizonte (objetivo = throughput)
    done = {}
    for j, o in enumerate(ordenes):
        cj = task[j, N_MACHINES_C4 - 1][1]
        d = model.NewBoolVar(f"done_{j}")
        model.Add(cj <= HORIZONTE).OnlyEnforceIf(d)
        model.Add(cj > HORIZONTE).OnlyEnforceIf(d.Not())
        done[j] = d
    model.Maximize(sum(int(o.margen) * done[j] for j, o in enumerate(ordenes)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_limite
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # reconstruir el orden del cuello siguiendo el circuito (gobierna los setups)
        cur = next(j for j in range(n) if solver.Value(start_lit[j]) == 1)
        seq = [cur]
        for _ in range(n - 1):
            cur = next(j for j in range(n) if j != cur and solver.Value(lit[cur, j]) == 1)
            seq.append(cur)
    else:
        seq = list(range(n))
    return evaluar(ordenes, seq)

# ───────── GA mono-objetivo (throughput) ─────────
def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.integers(0, n, 2)); h = [None] * n
    h[a:b] = p1[a:b]; resto = [x for x in p2 if x not in h]; pos = 0
    for i in range(n):
        if h[i] is None: h[i] = resto[pos]; pos += 1
    return h

def metodo_ga(ordenes, n_gen=45, pop=50, seed=0):
    rng = np.random.default_rng(seed); n = len(ordenes)
    def fit(seq): return evaluar(ordenes, seq)["throughput"]
    pobl = [list(rng.permutation(n)) for _ in range(pop)]; fits = [fit(s) for s in pobl]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2)
            p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            h = ox(p1, p2, rng)
            if rng.random() < 0.3:
                i, k = rng.integers(0, n, 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]; pobl = nueva; fits = [fit(s) for s in pobl]
    return evaluar(ordenes, pobl[int(np.argmax(fits))])

# ───────── PPO: construccion de secuencia sobre el mismo simulador ─────────
class SeqEnv(gym.Env):
    def __init__(self, train=True):
        super().__init__()
        self.n = NORD; self.train = train
        self.nf = len(FAMILIAS)
        self.observation_space = spaces.Box(0.0, 1.0,
            shape=(self.n + N_MACHINES_C4 + self.nf + self.nf + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n)
        self.rng = np.random.default_rng(0); self.ordenes_fijas = None
        self._reset_interno()
    def fijar_instancia(self, o): self.ordenes_fijas = o
    def _reset_interno(self):
        self.ordenes = (gen(int(self.rng.integers(0, 1_000_000)))
                        if (self.train or self.ordenes_fijas is None) else self.ordenes_fijas)
        self.mask = np.zeros(self.n, dtype=np.float32); self.sched = []
        self.last_fam = -1
        self.carga = np.zeros(N_MACHINES_C4, dtype=np.float32)
        for o in self.ordenes:
            for m in range(N_MACHINES_C4): self.carga[m] += o.tiempos[m]
        self.carga_norm = self.carga / (self.carga.max() + 1e-6)
    def _obs(self):
        cnt = np.array([sum(1 for j in self.sched if self.ordenes[j].tipo == f)
                        / max(1, self.n) for f in FAMILIAS], dtype=np.float32)
        lf = np.zeros(self.nf, dtype=np.float32)
        if self.last_fam >= 0: lf[self.last_fam] = 1.0
        prog = len(self.sched) / self.n
        return np.concatenate([self.mask, self.carga_norm, cnt, lf, [prog]]).astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_interno(); return self._obs(), {}
    def step(self, a):
        if self.mask[a] == 1.0: a = int(np.where(self.mask == 0.0)[0][0])
        self.mask[a] = 1.0; self.sched.append(int(a))
        self.last_fam = FAMILIAS.index(self.ordenes[a].tipo)
        if len(self.sched) < self.n: return self._obs(), 0.0, False, False, {}
        r = PlantaC4(self.ordenes, self.sched, HORIZONTE).ejecutar()
        return self._obs(), r["throughput"] / 900.0, True, False, {"throughput": r["throughput"]}

def entrenar_ppo(pasos=160_000):
    env = SeqEnv(train=True)
    m = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=512,
            batch_size=64, n_epochs=10, gamma=1.0, ent_coef=0.01, seed=0)
    m.learn(total_timesteps=pasos); return m

def metodo_ppo(modelo, ordenes):
    env = SeqEnv(train=False); env.fijar_instancia(ordenes)
    obs, _ = env.reset(); done = False
    while not done:
        a, _ = modelo.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(a)); done = term or trunc
    return evaluar(ordenes, env.sched)

def main():
    print("=" * 64)
    print("  ESTUDIO RIGUROSO - CASO 4 (Flow Shop con setups en el cuello)")
    print(f"  Pool 5A+5B+5C | Horizonte={HORIZONTE} | {N_TEST} instancias")
    print("=" * 64)
    print("\n[PPO] Entrenando (160k pasos)...")
    modelo = entrenar_ppo(160_000)
    metodos = ["FIFO", "CP-SAT", "NSGA-II", "PPO"]
    res = {m: [] for m in metodos}
    setups = {m: [] for m in metodos}
    compl = {m: [] for m in metodos}
    for i, sd in enumerate(SEEDS_TEST):
        ordenes = gen(sd)
        for nombre, fn in [("FIFO", lambda o: metodo_fifo(o)),
                           ("CP-SAT", lambda o: metodo_cpsat(o)),
                           ("NSGA-II", lambda o: metodo_ga(o, seed=sd)),
                           ("PPO", lambda o: metodo_ppo(modelo, o))]:
            r = fn(ordenes)
            res[nombre].append(r["throughput"])
            setups[nombre].append(r["setup_total"])
            compl[nombre].append(r["ordenes_completadas"])
        if (i + 1) % 10 == 0: print(f"  ...{i+1}/{N_TEST}")
    print("\n--- RESULTADOS (throughput EUR, media +/- desv.) ---")
    resumen = {}
    for m in metodos:
        arr = np.array(res[m])
        resumen[m] = {"media": float(arr.mean()), "std": float(arr.std()),
                      "setup_medio": float(np.mean(setups[m])),
                      "compl_media": float(np.mean(compl[m]))}
        print(f"  {m:<9}: {arr.mean():7.1f} +/- {arr.std():5.1f}   "
              f"setups {np.mean(setups[m]):5.1f} u.t.   compl {np.mean(compl[m]):.1f}/{NORD}")
    print("\n--- WILCOXON ---")
    wil = {}
    for a, b in [("NSGA-II", "FIFO"), ("CP-SAT", "FIFO"), ("NSGA-II", "CP-SAT"),
                 ("PPO", "FIFO"), ("NSGA-II", "PPO")]:
        try:
            _, p = stats.wilcoxon(res[a], res[b])
        except ValueError:
            p = 1.0
        wil[f"{a} vs {b}"] = float(p)
        print(f"  {a:<9} vs {b:<9}: p={p:.4f} [{'SIG' if p < 0.05 else 'ns'}]")
    json.dump({"horizonte": HORIZONTE, "resumen": resumen, "wilcoxon": wil, "raw": res},
              open(os.path.join(RESDIR, "resultados_c4.json"), "w"), indent=2)
    print("\n[OK] resultados_c4.json")

if __name__ == "__main__":
    main()

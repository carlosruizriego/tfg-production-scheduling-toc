# -*- coding: utf-8 -*-
"""
ESTUDIO RIGUROSO - CASO 2 (Job Shop multiproducto con minimos)
==============================================================
Mismo evaluador unico (PlantaC2), CP-SAT alineado a throughput con minimos como
restriccion dura, GA y PPO con penalizacion por minimos. 30 instancias + Wilcoxon.
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
from planta_caso2 import (generar_ordenes_c2, PlantaC2, PRODUCTOS, MINIMOS,
                          MIX_ORDENES, N_MACHINES_C2)

HORIZONTE = 230
N_TEST = 30
SEEDS_TEST = list(range(100, 100 + N_TEST))
NORD = len(MIX_ORDENES)   # 13
TIPOS = ['A', 'B', 'C']

def evaluar(ordenes, seq):
    return PlantaC2(ordenes, list(seq), HORIZONTE).ejecutar()

def metodo_fifo(ordenes):
    return evaluar(ordenes, list(range(len(ordenes))))

def metodo_cpsat(ordenes, tiempo_limite=10.0):
    model = cp_model.CpModel()
    n = len(ordenes)
    horizon_sched = int(sum(t for o in ordenes for _, t in o.ruta)) + 1
    task = {}
    for j, o in enumerate(ordenes):
        for k, (m, t) in enumerate(o.ruta):
            dur = max(1, int(round(t)))
            s = model.NewIntVar(0, horizon_sched, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon_sched, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, dur, e, f"i_{j}_{k}")
            task[j, k] = (s, e, iv, m)
    for j, o in enumerate(ordenes):
        for k in range(len(o.ruta) - 1):
            model.Add(task[j, k + 1][0] >= task[j, k][1])
    for mm in range(N_MACHINES_C2):
        ivs = [task[j, k][2] for j, o in enumerate(ordenes) for k in range(len(o.ruta))
               if task[j, k][3] == mm]
        if ivs:
            model.AddNoOverlap(ivs)
    done = {}
    for j, o in enumerate(ordenes):
        cj = task[j, len(o.ruta) - 1][1]
        d = model.NewBoolVar(f"done_{j}")
        model.Add(cj <= HORIZONTE).OnlyEnforceIf(d)
        model.Add(cj > HORIZONTE).OnlyEnforceIf(d.Not())
        done[j] = d
    # minimos como restriccion dura (fortaleza de CP-SAT)
    for t in TIPOS:
        idx = [j for j, o in enumerate(ordenes) if o.tipo == t]
        model.Add(sum(done[j] for j in idx) >= MINIMOS[t])
    # maximizar throughput (margen de las completadas)
    model.Maximize(sum(int(o.margen) * done[j] for j, o in enumerate(ordenes)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_limite
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        inicios = sorted([(j, solver.Value(task[j, 0][0])) for j in range(n)],
                         key=lambda x: x[1])
        seq = [j for j, _ in inicios]
    else:
        seq = list(range(n))
    return evaluar(ordenes, seq)

def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.integers(0, n, 2)); h = [None]*n
    h[a:b] = p1[a:b]; resto = [x for x in p2 if x not in h]; pos = 0
    for i in range(n):
        if h[i] is None: h[i] = resto[pos]; pos += 1
    return h

def metodo_ga(ordenes, n_gen=45, pop=50, seed=0):
    rng = np.random.default_rng(seed); n = len(ordenes)
    def fit(seq):
        r = evaluar(ordenes, seq)
        return r["throughput_neto"]   # penaliza minimos incumplidos
    pobl = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(s) for s in pobl]
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
        nueva[0] = pobl[int(np.argmax(fits))][:]
        pobl = nueva; fits = [fit(s) for s in pobl]
    return evaluar(ordenes, pobl[int(np.argmax(fits))])

class SeqEnv(gym.Env):
    def __init__(self, train=True):
        super().__init__()
        self.n = NORD; self.train = train
        self.observation_space = spaces.Box(0.0, 1.0,
            shape=(self.n + N_MACHINES_C2 + 3 + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n)
        self.rng = np.random.default_rng(0)
        self.ordenes_fijas = None
        self._reset_interno()
    def fijar_instancia(self, o): self.ordenes_fijas = o
    def _reset_interno(self):
        if self.train or self.ordenes_fijas is None:
            self.ordenes = generar_ordenes_c2(seed=int(self.rng.integers(0, 1_000_000)))
        else:
            self.ordenes = self.ordenes_fijas
        self.mask = np.zeros(self.n, dtype=np.float32)
        self.sched = []
        self.carga = np.zeros(N_MACHINES_C2, dtype=np.float32)
        for o in self.ordenes:
            for m, t in o.ruta: self.carga[m] += t
        self.carga_norm = self.carga / (self.carga.max() + 1e-6)
    def _obs(self):
        cnt = np.array([sum(1 for j in self.sched if self.ordenes[j].tipo == t)
                        / max(1, self.n) for t in TIPOS], dtype=np.float32)
        prog = len(self.sched) / self.n
        return np.concatenate([self.mask, self.carga_norm, cnt, [prog]]).astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_interno(); return self._obs(), {}
    def step(self, a):
        if self.mask[a] == 1.0:
            a = int(np.where(self.mask == 0.0)[0][0])
        self.mask[a] = 1.0; self.sched.append(int(a))
        if len(self.sched) < self.n:
            return self._obs(), 0.0, False, False, {}
        r = PlantaC2(self.ordenes, self.sched, HORIZONTE).ejecutar()
        return self._obs(), r["throughput_neto"] / 800.0, True, False, {}

def entrenar_ppo(pasos=150_000):
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
    print("  ESTUDIO RIGUROSO - CASO 2 (multiproducto con minimos)")
    print(f"  Horizonte={HORIZONTE}  |  {N_TEST} instancias  |  evaluador unico SimPy")
    print("=" * 64)
    print("\n[PPO] Entrenando (150k pasos)...")
    modelo = entrenar_ppo(150_000)
    res = {m: [] for m in ["FIFO", "CP-SAT", "NSGA-II", "PPO"]}
    minok = {m: [] for m in res}
    for i, sd in enumerate(SEEDS_TEST):
        ordenes = generar_ordenes_c2(seed=sd)
        for nombre, fn in [("FIFO", lambda o: metodo_fifo(o)),
                           ("CP-SAT", lambda o: metodo_cpsat(o)),
                           ("NSGA-II", lambda o: metodo_ga(o, seed=sd)),
                           ("PPO", lambda o: metodo_ppo(modelo, o))]:
            r = fn(ordenes)
            res[nombre].append(r["throughput"])
            minok[nombre].append(1 if r["minimos_cumplidos"] else 0)
        if (i + 1) % 10 == 0: print(f"  ...{i+1}/{N_TEST}")
    print("\n--- RESULTADOS (throughput EUR, media +/- desv.) ---")
    resumen = {}
    for m in ["FIFO", "CP-SAT", "NSGA-II", "PPO"]:
        arr = np.array(res[m])
        resumen[m] = {"media": float(arr.mean()), "std": float(arr.std()),
                      "minimos_pct": float(100*np.mean(minok[m]))}
        print(f"  {m:<9}: {arr.mean():7.1f} +/- {arr.std():5.1f}   "
              f"minimos OK {100*np.mean(minok[m]):.0f}%")
    print("\n--- WILCOXON ---")
    wil = {}
    for a, b in [("NSGA-II","FIFO"),("CP-SAT","FIFO"),("NSGA-II","CP-SAT"),
                 ("PPO","FIFO"),("CP-SAT","PPO")]:
        try: _, p = stats.wilcoxon(res[a], res[b])
        except ValueError: p = 1.0
        wil[f"{a} vs {b}"] = float(p)
        print(f"  {a:<9} vs {b:<9}: p={p:.4f} [{'SIG' if p<0.05 else 'ns'}]")
    json.dump({"horizonte":HORIZONTE,"resumen":resumen,"wilcoxon":wil,
               "raw":res}, open(os.path.join(RESDIR, "resultados_c2.json"),"w"), indent=2)
    print("\n[OK] resultados_c2.json")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
ESTUDIO RIGUROSO - CASO 1 (Job Shop homogeneo)
==============================================
Correcciones metodologicas:
  1. Simulador UNICO (PlantaSimPy) para los 4 metodos -> comparacion valida.
  2. CP-SAT realineado: maximiza ordenes completadas dentro del horizonte (throughput),
     no el makespan.
  3. PPO formulado como CONSTRUCCION DE SECUENCIA evaluada en la misma PlantaSimPy.
  4. 30 instancias (semillas), media +/- desv. y test de Mann-Whitney (Wilcoxon).
  5. Horizonte ajustado para que la capacidad sea restrictiva.
  6. Validacion sobre instancia benchmark ft06 (OR-Library, optimo conocido = 55).
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rutas de salida relativas a la raiz del proyecto (este script vive en src)
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
FIGDIR = os.path.join(_BASE, "figures")

from ortools.sat.python import cp_model
from scipy import stats
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class CurvaCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.throughputs = []
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "throughput" in info:
                self.throughputs.append(info["throughput"])
        return True

from planta_simulada import generar_instancia, PlantaSimPy, N_MACHINES, M_JOBS

HORIZONTE = 150          # < makespan tipico (~167) -> capacidad restrictiva, ~50-60% completadas
N_TEST = 30              # instancias de evaluacion
SEEDS_TEST = list(range(100, 100 + N_TEST))
MARGEN = 70

# ───────────────────────── Evaluador unico ─────────────────────────
def evaluar(ordenes, secuencia):
    """Unica fuente de verdad: simula la secuencia en SimPy y mide throughput."""
    res = PlantaSimPy(ordenes, list(secuencia), HORIZONTE).ejecutar()
    return res

# ───────────────────────── FIFO ─────────────────────────
def metodo_fifo(ordenes):
    return evaluar(ordenes, list(range(len(ordenes))))

# ───────────────────────── CP-SAT alineado a throughput ─────────────────────────
def metodo_cpsat(ordenes, tiempo_limite=10.0):
    model = cp_model.CpModel()
    n = len(ordenes)
    horizon_sched = int(sum(op.duracion for o in ordenes for op in o.operaciones)) + 1
    task = {}
    for j, o in enumerate(ordenes):
        for k, op in enumerate(o.operaciones):
            dur = max(1, int(round(op.duracion)))
            s = model.NewIntVar(0, horizon_sched, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon_sched, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, dur, e, f"i_{j}_{k}")
            task[j, k] = (s, e, iv)
    # precedencias
    for j, o in enumerate(ordenes):
        for k in range(len(o.operaciones) - 1):
            model.Add(task[j, k + 1][0] >= task[j, k][1])
    # no solapamiento por maquina
    maq = {m: [] for m in range(N_MACHINES)}
    for j, o in enumerate(ordenes):
        for k, op in enumerate(o.operaciones):
            maq[op.maquina].append(task[j, k][2])
    for m in range(N_MACHINES):
        if maq[m]:
            model.AddNoOverlap(maq[m])
    # done_j = (fin_ultima_op_j <= HORIZONTE)  -> objetivo alineado con throughput
    done = []
    for j, o in enumerate(ordenes):
        cj = task[j, len(o.operaciones) - 1][1]
        d = model.NewBoolVar(f"done_{j}")
        model.Add(cj <= HORIZONTE).OnlyEnforceIf(d)
        model.Add(cj > HORIZONTE).OnlyEnforceIf(d.Not())
        done.append(d)
    model.Maximize(sum(done))   # maximizar nº de ordenes completadas (margen homogeneo)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_limite
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        inicios = [(j, solver.Value(task[j, 0][0])) for j in range(n)]
        seq = [j for j, _ in sorted(inicios, key=lambda x: x[1])]
    else:
        seq = list(range(n))
    return evaluar(ordenes, seq)

# ───────────────────────── GA mono-objetivo (throughput) ─────────────────────────
def metodo_ga(ordenes, n_gen=40, pop=40, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ordenes)
    def fit(seq):
        return evaluar(ordenes, seq)["throughput"]
    poblacion = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(s) for s in poblacion]
    for _ in range(n_gen):
        # seleccion por torneo + OX + swap
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            p1 = poblacion[a] if fits[a] >= fits[b] else poblacion[b]
            c, d = rng.integers(0, pop, 2)
            p2 = poblacion[c] if fits[c] >= fits[d] else poblacion[d]
            hijo = ox(p1, p2, rng)
            if rng.random() < 0.3:
                i, k = rng.integers(0, n, 2)
                hijo[i], hijo[k] = hijo[k], hijo[i]
            nueva.append(hijo)
        # elitismo
        mejor = poblacion[int(np.argmax(fits))]
        nueva[0] = mejor[:]
        poblacion = nueva
        fits = [fit(s) for s in poblacion]
    mejor = poblacion[int(np.argmax(fits))]
    return evaluar(ordenes, mejor)

def ox(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.integers(0, n, 2))
    h = [None] * n
    h[a:b] = p1[a:b]
    resto = [x for x in p2 if x not in h]
    pos = 0
    for i in range(n):
        if h[i] is None:
            h[i] = resto[pos]; pos += 1
    return h

# ───────────────────────── PPO: construccion de secuencia sobre SimPy ─────────────
class SeqConstructEnv(gym.Env):
    """El agente construye la secuencia de despacho (un job por paso).
    Recompensa terminal = throughput real medido en PlantaSimPy (mismo evaluador)."""
    def __init__(self, train=True):
        super().__init__()
        self.n = M_JOBS
        self.train = train
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.n + N_MACHINES + 1,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.n)
        self.rng = np.random.default_rng(0)
        self.ordenes_fijas = None
        self._reset_interno(seed_inst=None)

    def fijar_instancia(self, ordenes):
        self.ordenes_fijas = ordenes

    def _reset_interno(self, seed_inst):
        if self.train or self.ordenes_fijas is None:
            s = int(self.rng.integers(0, 1_000_000))
            self.ordenes = generar_instancia(seed=s)
        else:
            self.ordenes = self.ordenes_fijas
        self.scheduled = []
        self.mask = np.zeros(self.n, dtype=np.float32)
        self.carga = np.zeros(N_MACHINES, dtype=np.float32)
        for o in self.ordenes:
            for op in o.operaciones:
                self.carga[op.maquina] += op.duracion
        self.carga_norm = self.carga / (self.carga.max() + 1e-6)

    def _obs(self):
        prog = len(self.scheduled) / self.n
        return np.concatenate([self.mask, self.carga_norm, [prog]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_interno(seed)
        return self._obs(), {}

    def step(self, action):
        if self.mask[action] == 1.0:
            disponibles = np.where(self.mask == 0.0)[0]
            action = int(disponibles[0])
        self.mask[action] = 1.0
        self.scheduled.append(int(action))
        if len(self.scheduled) < self.n:
            return self._obs(), 0.0, False, False, {}
        # secuencia completa -> evaluar en SimPy
        res = PlantaSimPy(self.ordenes, self.scheduled, HORIZONTE).ejecutar()
        reward = res["throughput"] / 700.0
        return self._obs(), reward, True, False, {"throughput": res["throughput"]}

def entrenar_ppo(pasos=120_000):
    env = SeqConstructEnv(train=True)
    modelo = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=512,
                 batch_size=64, n_epochs=10, gamma=1.0, ent_coef=0.01, seed=0)
    cb = CurvaCallback()
    modelo.learn(total_timesteps=pasos, callback=cb)
    # Curva de aprendizaje (throughput terminal por episodio de entrenamiento)
    if len(cb.throughputs) > 30:
        v = 30
        suav = np.convolve(cb.throughputs, np.ones(v) / v, mode="valid")
        plt.figure(figsize=(8, 4.2))
        plt.plot(suav, color="#E8743B", lw=1.5)
        plt.title("Caso 1 - PPO (entorno unificado): throughput por episodio de entrenamiento")
        plt.xlabel("Episodio de entrenamiento"); plt.ylabel("Throughput T (EUR)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, "rig_aprendizaje_c1.png"), dpi=150); plt.close()
        print("[PPO] Curva de aprendizaje guardada: rig_aprendizaje_c1.png")
    return modelo

def metodo_ppo(modelo, ordenes):
    env = SeqConstructEnv(train=False)
    env.fijar_instancia(ordenes)
    obs, _ = env.reset()
    done = False
    while not done:
        a, _ = modelo.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
    return evaluar(ordenes, env.scheduled)

# ───────────────────────── Benchmark ft06 (OR-Library) ─────────────────────────
FT06 = [
    [(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],
    [(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],
    [(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],
    [(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],
    [(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],
    [(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)],
]
def benchmark_ft06():
    model = cp_model.CpModel()
    nj, nm = 6, 6
    H = sum(t for job in FT06 for _, t in job)
    task = {}
    for j, job in enumerate(FT06):
        for k, (m, t) in enumerate(job):
            s = model.NewIntVar(0, H, f"s_{j}_{k}")
            e = model.NewIntVar(0, H, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, t, e, f"i_{j}_{k}")
            task[j, k] = (s, e, iv, m)
    for j, job in enumerate(FT06):
        for k in range(len(job) - 1):
            model.Add(task[j, k + 1][0] >= task[j, k][1])
    for mm in range(nm):
        ivs = [task[j, k][2] for j, job in enumerate(FT06) for k in range(len(job)) if task[j, k][3] == mm]
        model.AddNoOverlap(ivs)
    cmax = model.NewIntVar(0, H, "cmax")
    model.AddMaxEquality(cmax, [task[j, len(FT06[j]) - 1][1] for j in range(nj)])
    model.Minimize(cmax)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    st = solver.Solve(model)
    return solver.Value(cmax), solver.StatusName(st)

# ───────────────────────── Main ─────────────────────────
def main():
    print("=" * 64)
    print("  ESTUDIO RIGUROSO - CASO 1 (Job Shop homogeneo)")
    print(f"  Horizonte={HORIZONTE} u.t.  |  {N_TEST} instancias  |  evaluador unico SimPy")
    print("=" * 64)

    # Validacion benchmark
    mk, status = benchmark_ft06()
    print(f"\n[Benchmark ft06] Makespan CP-SAT = {mk} (optimo conocido = 55) [{status}]")

    print("\n[PPO] Entrenando politica de construccion de secuencia (120k pasos)...")
    modelo = entrenar_ppo(120_000)

    res = {m: [] for m in ["FIFO", "CP-SAT", "NSGA-II", "PPO"]}
    compl = {m: [] for m in res}
    for i, sd in enumerate(SEEDS_TEST):
        ordenes = generar_instancia(seed=sd)
        for nombre, fn in [("FIFO", lambda o: metodo_fifo(o)),
                           ("CP-SAT", lambda o: metodo_cpsat(o)),
                           ("NSGA-II", lambda o: metodo_ga(o, seed=sd)),
                           ("PPO", lambda o: metodo_ppo(modelo, o))]:
            r = fn(ordenes)
            res[nombre].append(r["throughput"])
            compl[nombre].append(r["ordenes_completadas"])
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{N_TEST} instancias")

    print("\n--- RESULTADOS (throughput EUR, media +/- desv. sobre 30 instancias) ---")
    resumen = {}
    for m in ["FIFO", "CP-SAT", "NSGA-II", "PPO"]:
        arr = np.array(res[m])
        resumen[m] = {"media": float(arr.mean()), "std": float(arr.std()),
                      "compl_media": float(np.mean(compl[m]))}
        print(f"  {m:<9}: {arr.mean():7.1f} +/- {arr.std():5.1f}   "
              f"(compl. {np.mean(compl[m]):.1f}/10)")

    print("\n--- TEST DE WILCOXON (pareado, throughput) ---")
    pares = [("NSGA-II", "FIFO"), ("CP-SAT", "FIFO"), ("NSGA-II", "CP-SAT"),
             ("PPO", "FIFO"), ("NSGA-II", "PPO")]
    wilcox = {}
    for a, b in pares:
        try:
            stat, p = stats.wilcoxon(res[a], res[b])
        except ValueError:
            stat, p = float("nan"), 1.0
        sig = "SIGNIFICATIVO" if p < 0.05 else "no significativo"
        wilcox[f"{a} vs {b}"] = {"p": float(p)}
        print(f"  {a:<9} vs {b:<9}: p = {p:.4f}  [{sig}]")

    salida = {"horizonte": HORIZONTE, "n_test": N_TEST, "ft06_makespan": mk,
              "resumen": resumen, "wilcoxon": wilcox,
              "raw": {m: res[m] for m in res}}
    with open(os.path.join(RESDIR, "resultados_c1.json"), "w") as f:
        json.dump(salida, f, indent=2)
    print("\n[OK] Guardado resultados_c1.json")

if __name__ == "__main__":
    main()

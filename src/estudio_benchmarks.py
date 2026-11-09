# -*- coding: utf-8 -*-
"""
ESTUDIO DE BENCHMARKS Y ESCALABILIDAD (makespan)
================================================
Objetivo: contrastar la validez externa y la ESCALABILIDAD de los metodos mas alla
de los casos de juguete del estudio principal. A diferencia del resto del trabajo
(que optimiza el throughput), aqui el objetivo es el MAKESPAN, que es el criterio
estandar de los benchmarks clasicos de Job Shop y el que tiene optimos publicados.

Dos bloques:
  1. VALIDACION con optimos conocidos: ft06 (optimo 55) y ft10 (optimo 930) de
     Fisher-Thompson. Que CP-SAT los recupere valida solver y datos.
  2. ESCALABILIDAD: instancias estilo Taillard (tiempos U[1,99], orden de maquinas
     aleatorio, Taillard 1993), reproducibles, de tamano creciente. Se mide la
     brecha de optimalidad y el tiempo de CP-SAT frente al genetico (GA).

Reanudable (checkpoint por bloque) y con wake-lock para Modern Standby.
Salida: results/resultados_benchmarks.json y figura rig_benchmarks.png.
"""
import numpy as np, json, os, time, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
FIGDIR = os.path.join(_BASE, "figures")
PATH = os.path.join(RESDIR, "resultados_benchmarks.json")


def _wake_lock(on=True):
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
        print(f"[wake-lock] {'activado' if on else 'liberado'}", flush=True)
    except Exception as e:
        print(f"[wake-lock] no disponible: {e}", flush=True)


# ─────────── Instancias de la literatura (machine, duracion) ───────────
FT06 = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
]
FT10 = [
    [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
    [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
    [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
    [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
    [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
    [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
    [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
    [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
    [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)],
    [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)],
]
OPTIMOS = {"ft06": 55, "ft10": 930}


def gen_taillard_like(n, m, seed):
    """Instancia estilo Taillard: orden de maquinas aleatorio y tiempos U[1,99]."""
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(n):
        maquinas = list(rng.permutation(m))
        tiempos = rng.integers(1, 100, size=m)
        jobs.append([(int(mm), int(t)) for mm, t in zip(maquinas, tiempos)])
    return jobs


def n_machines(jobs):
    return max(mm for job in jobs for mm, _ in job) + 1


# ─────────── CP-SAT: minimizar makespan ───────────
def cpsat_makespan(jobs, time_limit):
    model = cp_model.CpModel()
    n = len(jobs); m = n_machines(jobs)
    H = sum(d for job in jobs for _, d in job)
    task = {}
    for j, job in enumerate(jobs):
        for k, (mm, d) in enumerate(job):
            s = model.NewIntVar(0, H, f"s_{j}_{k}")
            e = model.NewIntVar(0, H, f"e_{j}_{k}")
            task[j, k] = (s, e, model.NewIntervalVar(s, d, e, f"i_{j}_{k}"), mm)
    for j, job in enumerate(jobs):
        for k in range(len(job) - 1):
            model.Add(task[j, k + 1][0] >= task[j, k][1])
    for mm in range(m):
        model.AddNoOverlap([task[j, k][2] for j, job in enumerate(jobs)
                            for k in range(len(job)) if task[j, k][3] == mm])
    cmax = model.NewIntVar(0, H, "cmax")
    model.AddMaxEquality(cmax, [task[j, len(jobs[j]) - 1][1] for j in range(n)])
    model.Minimize(cmax)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    t0 = time.perf_counter()
    st = solver.Solve(model)
    secs = time.perf_counter() - t0
    obj = solver.ObjectiveValue()
    bound = solver.BestObjectiveBound()
    proved = (st == cp_model.OPTIMAL)
    return int(round(obj)), float(bound), proved, secs


# ─────────── GA (codificacion basada en operaciones) ───────────
def makespan_decode(opseq, jobs, m):
    next_op = [0] * len(jobs)
    maq_libre = [0] * m
    job_libre = [0] * len(jobs)
    cmax = 0
    for j in opseq:
        k = next_op[j]; next_op[j] += 1
        mm, d = jobs[j][k]
        ini = max(maq_libre[mm], job_libre[j])
        fin = ini + d
        maq_libre[mm] = fin; job_libre[j] = fin
        if fin > cmax:
            cmax = fin
    return cmax


def jox(p1, p2, rng, jobset):
    elegidos = set(j for j in jobset if rng.random() < 0.5)
    hijo = [g if g in elegidos else None for g in p1]
    resto = (g for g in p2 if g not in elegidos)
    for i in range(len(hijo)):
        if hijo[i] is None:
            hijo[i] = next(resto)
    return hijo


def ga_makespan(jobs, gens=80, pop=60, seed=0, time_limit=None):
    """GA de makespan. Si time_limit se da, itera generaciones hasta agotar ese
    presupuesto de tiempo (comparacion justa frente a CP-SAT con el mismo limite)."""
    rng = np.random.default_rng(seed)
    n = len(jobs); m = n_machines(jobs)
    base = []
    for j, job in enumerate(jobs):
        base += [j] * len(job)
    jobset = list(range(n))
    pobl = []
    for _ in range(pop):
        c = base[:]; rng.shuffle(c); pobl.append(c)
    fits = [makespan_decode(c, jobs, m) for c in pobl]
    mejor = min(fits)
    t0 = time.perf_counter()
    g = 0
    while True:
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            p1 = pobl[a] if fits[a] <= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2)
            p2 = pobl[c] if fits[c] <= fits[d] else pobl[d]
            h = jox(p1, p2, rng, jobset)
            if rng.random() < 0.3:
                i, k = rng.integers(0, len(h), 2)
                h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmin(fits))][:]
        pobl = nueva
        fits = [makespan_decode(c, jobs, m) for c in pobl]
        mejor = min(mejor, min(fits))
        g += 1
        if time_limit is not None:
            if time.perf_counter() - t0 >= time_limit:
                break
        elif g >= gens:
            break
    secs = time.perf_counter() - t0
    return int(mejor), secs


def baseline_makespan(jobs):
    """Regla de despacho simple: operaciones en orden de job (referencia)."""
    n = len(jobs); m = n_machines(jobs)
    base = []
    for j, job in enumerate(jobs):
        base += [j] * len(job)
    return makespan_decode(base, jobs, m)


# ─────────── Ejecucion ───────────
def _save(d):
    json.dump(d, open(PATH, "w"), indent=2)


LADDER = [(10, 10), (15, 15), (20, 15), (20, 20), (30, 20)]
SEEDS = [0, 1, 2]
TL_LADDER = 30.0


def main():
    _wake_lock(True)
    print("=" * 64, flush=True)
    print("  BENCHMARKS Y ESCALABILIDAD (makespan, reanudable)", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}

    # --- Bloque 1: validacion con optimos conocidos ---
    if "validacion" not in salida:
        print("\n[VALIDACION] ft06 y ft10 (optimos 55 y 930)", flush=True)
        val = {}
        for nombre, jobs, tl in [("ft06", FT06, 15.0), ("ft10", FT10, 60.0)]:
            obj, bound, proved, secs = cpsat_makespan(jobs, tl)
            ga, gsec = ga_makespan(jobs, gens=120, pop=80, seed=0)
            val[nombre] = {"optimo_conocido": OPTIMOS[nombre], "cpsat": obj,
                           "cpsat_demostrado": proved, "cpsat_s": round(secs, 1),
                           "ga": ga, "ga_s": round(gsec, 1)}
            ok = "OK" if obj == OPTIMOS[nombre] else "!! DISCREPA"
            print(f"   {nombre}: CP-SAT={obj} (opt {OPTIMOS[nombre]}, {ok}, "
                  f"demostrado={proved}, {secs:.1f}s) | GA={ga}", flush=True)
        salida["validacion"] = val
        _save(salida)

    # --- Bloque 2: escalabilidad ---
    prog = salida.get("escalabilidad", {})
    for (n, m) in LADDER:
        key = f"{n}x{m}"
        if key in prog:
            print(f"\n[{key}] (checkpoint)", flush=True)
            continue
        print(f"\n[{key}] {len(SEEDS)} instancias, CP-SAT {TL_LADDER:.0f}s c/u", flush=True)
        filas = []
        for sd in SEEDS:
            jobs = gen_taillard_like(n, m, seed=sd)
            obj, bound, proved, secs = cpsat_makespan(jobs, TL_LADDER)
            ga, gsec = ga_makespan(jobs, pop=80, seed=sd, time_limit=TL_LADDER)
            base = baseline_makespan(jobs)
            mejor = min(obj, ga)
            gap_cpsat = 100 * (obj - bound) / obj if obj > 0 else 0.0
            filas.append({"seed": sd, "cpsat": obj, "cpsat_bound": round(bound, 1),
                          "cpsat_gap_pct": round(gap_cpsat, 1), "cpsat_proved": proved,
                          "cpsat_s": round(secs, 1), "ga": ga, "ga_s": round(gsec, 2),
                          "baseline": base, "mejor": mejor})
            print(f"   seed {sd}: CP-SAT={obj} (gap {gap_cpsat:.1f}%, {secs:.1f}s, "
                  f"opt={proved}) | GA={ga} ({gsec:.1f}s) | FIFO={base}", flush=True)
        agg = {
            "n": n, "m": m, "filas": filas,
            "cpsat_gap_medio": round(float(np.mean([f["cpsat_gap_pct"] for f in filas])), 1),
            "cpsat_s_medio": round(float(np.mean([f["cpsat_s"] for f in filas])), 1),
            "cpsat_proved": sum(f["cpsat_proved"] for f in filas),
            "ga_s_medio": round(float(np.mean([f["ga_s"] for f in filas])), 2),
            # brecha del GA respecto a la mejor solucion conocida (min CP-SAT/GA) por instancia
            "ga_gap_medio": round(float(np.mean(
                [100 * (f["ga"] - f["mejor"]) / f["mejor"] for f in filas])), 1),
            "cpsat_gap_vs_mejor": round(float(np.mean(
                [100 * (f["cpsat"] - f["mejor"]) / f["mejor"] for f in filas])), 1),
        }
        prog[key] = agg
        salida["escalabilidad"] = prog
        _save(salida)
        print(f"   -> CP-SAT gap medio {agg['cpsat_gap_medio']}% en {agg['cpsat_s_medio']}s "
              f"(demostrados {agg['cpsat_proved']}/{len(SEEDS)}) | GA {agg['ga_s_medio']}s",
              flush=True)

    print("\n[OK] resultados_benchmarks.json", flush=True)
    if all(f"{n}x{m}" in salida.get("escalabilidad", {}) for n, m in LADDER):
        figura(salida)
    _wake_lock(False)


def figura(salida):
    esc = salida["escalabilidad"]
    keys = [f"{n}x{m}" for n, m in LADDER]
    gap = [esc[k]["cpsat_gap_medio"] for k in keys]
    secs = [esc[k]["cpsat_s_medio"] for k in keys]
    proved = [esc[k]["cpsat_proved"] for k in keys]
    x = np.arange(len(keys))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    # Panel 1: brecha de optimalidad de CP-SAT
    bars = a1.bar(x, gap, 0.55, color="#2E74B5", edgecolor="black", zorder=3)
    for xi, g, pr in zip(x, gap, proved):
        txt = "óptimo\ndemostrado" if pr == len(SEEDS) and g < 0.05 else f"{g:.0f}%"
        a1.text(xi, g + max(gap) * 0.02 + 0.2, txt, ha="center", va="bottom", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(keys, rotation=20)
    a1.set_ylabel("Brecha de optimalidad de CP-SAT (%)")
    a1.set_xlabel("Tamaño de la instancia (n×m)")
    a1.set_title("El exacto deja de garantizar el óptimo")
    a1.grid(True, axis="y", alpha=0.3, zorder=0)
    # Panel 2: tiempo de CP-SAT
    a2.plot(x, secs, "o-", color="#C0392B", lw=1.8, label="CP-SAT")
    a2.axhline(TL_LADDER, color="#999", ls="--", lw=1, label=f"límite {TL_LADDER:.0f} s")
    a2.set_xticks(x); a2.set_xticklabels(keys, rotation=20)
    a2.set_ylabel("Tiempo de cálculo medio (s)")
    a2.set_xlabel("Tamaño de la instancia (n×m)")
    a2.set_title("…y agota el presupuesto de tiempo")
    a2.set_ylim(0, TL_LADDER * 1.12)
    a2.legend(fontsize=8.5); a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "rig_benchmarks.png"), dpi=150)
    plt.close()
    print("[OK] rig_benchmarks.png", flush=True)


if __name__ == "__main__":
    main()

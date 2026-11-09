# -*- coding: utf-8 -*-
"""
Nucleo del duelo a NIVEL DE OPERACION (L2D) para el Caso 5.
===========================================================
El agente a nivel de orden no bate al GA en C5. La hipotesis (Zhang 2020) es que el
espacio de OPERACION (intercalar operaciones de distintos trabajos, no solo permutar
trabajos) es mas expresivo. Para una comparacion HONESTA, todos los metodos (greedy,
GA y RL) deben evaluarse con el MISMO scheduler op-a-op y en el MISMO espacio.

Este modulo provee:
  - `schedule_priority(ordenes, rank, cfg)`: list-scheduler op-a-op dirigido por una
    prioridad de TRABAJO. Con rank = posicion en una permutacion de trabajos reproduce
    EXACTAMENTE PlantaC5 (Gate 1) -> legitima comparar el espacio op-a-op con los baselines.
  - `schedule_stream(ordenes, stream, cfg)`: decodifica una secuencia de OPERACIONES
    (codificacion basada en operaciones: cada trabajo aparece len(ruta) veces) -> throughput.
    Es el espacio libre op-a-op (genoma del GA op-a-op).
  - cfg de C5 y la auditoria de coherencia con PlantaC5.
"""
import numpy as np
import planta_caso5 as P5
from planta_caso5 import generar_ordenes_c5, PlantaC5, POOL_DEFAULT, SETUP_A, SETUP_B, FAMILIAS

HORIZONTE = 550
NORM_C5 = 1200.0
SEEDS_TEST = list(range(100, 130))


def cfg_c5():
    ms = max(max(M[a][b] for a in FAMILIAS for b in FAMILIAS) for M in (SETUP_A, SETUP_B))
    return dict(key="c5", n_maq=P5.N_MACHINES_C5, NJOBS=len(POOL_DEFAULT),
                setup_por_maq={P5.CUELLO_A: SETUP_A, P5.CUELLO_B: SETUP_B},
                cuellos=(P5.CUELLO_A, P5.CUELLO_B), horizonte=HORIZONTE, norm=NORM_C5,
                max_setup=ms or 1.0, fams=list(FAMILIAS), N_PAD=125)


def gen(seed):
    return generar_ordenes_c5(POOL_DEFAULT[:], seed)


def _throughput(ordenes, done_time, H):
    return sum(o.margen for j, o in enumerate(ordenes)
               if done_time[j] is not None and done_time[j] <= H)


def schedule_priority(ordenes, rank, cfg):
    """List-scheduler op-a-op por prioridad de trabajo (rank). Reproduce PlantaC5 si
    rank = posicion en la permutacion de trabajos."""
    n = len(ordenes); n_maq = cfg["n_maq"]; SETUP = cfg["setup_por_maq"]
    CUELLOS = cfg["cuellos"]; H = cfg["horizonte"]
    job_pos = [0] * n; job_free = [0.0] * n; maq_free = [0.0] * n_maq
    last = {c: None for c in CUELLOS}; done_time = [None] * n
    total = sum(len(o.ruta) for o in ordenes)
    for _ in range(total):
        best = None
        for j in range(n):
            if job_pos[j] >= len(ordenes[j].ruta):
                continue
            m, dur = ordenes[j].ruta[job_pos[j]]
            s = max(job_free[j], maq_free[m]); setup = 0.0
            if m in CUELLOS:
                lf = last[m]
                if lf is not None and lf != ordenes[j].tipo:
                    setup = SETUP[m][lf][ordenes[j].tipo]
            ini = s + setup
            key = (ini, job_free[j], rank[j])
            if best is None or key < best[0]:
                best = (key, j, m, dur, ini, setup)
        _, j, m, dur, ini, setup = best
        fin = ini + dur
        maq_free[m] = fin; job_free[j] = fin
        if m in CUELLOS:
            last[m] = ordenes[j].tipo
        job_pos[j] += 1
        if job_pos[j] == len(ordenes[j].ruta):
            done_time[j] = fin
    return _throughput(ordenes, done_time, H)


def schedule_stream(ordenes, stream, cfg):
    """Decodifica una secuencia de OPERACIONES (cada trabajo aparece len(ruta) veces)."""
    n = len(ordenes); n_maq = cfg["n_maq"]; SETUP = cfg["setup_por_maq"]
    CUELLOS = cfg["cuellos"]; H = cfg["horizonte"]
    job_pos = [0] * n; job_free = [0.0] * n; maq_free = [0.0] * n_maq
    last = {c: None for c in CUELLOS}; done_time = [None] * n
    for j in stream:
        if job_pos[j] >= len(ordenes[j].ruta):
            nxt = next((i for i in range(n) if job_pos[i] < len(ordenes[i].ruta)), None)
            if nxt is None:
                break
            j = nxt
        m, dur = ordenes[j].ruta[job_pos[j]]
        s = max(job_free[j], maq_free[m]); setup = 0.0
        if m in CUELLOS:
            lf = last[m]
            if lf is not None and lf != ordenes[j].tipo:
                setup = SETUP[m][lf][ordenes[j].tipo]
            last[m] = ordenes[j].tipo
        fin = s + setup + dur
        maq_free[m] = fin; job_free[j] = fin; job_pos[j] += 1
        if job_pos[j] == len(ordenes[j].ruta):
            done_time[j] = fin
    return _throughput(ordenes, done_time, H)


# ───────── Greedy op-a-op (regla de despacho earliest-start) ─────────
def greedy_op(ordenes, cfg):
    """Despacho op-a-op: en cada paso, la operacion lista de inicio mas temprano."""
    n = len(ordenes)
    return schedule_priority(ordenes, np.arange(n, dtype=float), cfg)


def greedy_stream(ordenes, cfg):
    """Devuelve la SECUENCIA DE ACCIONES (jobs) que despacha el greedy earliest-start;
    sirve de experto para el warm-start del agente L2D."""
    n = len(ordenes); n_maq = cfg["n_maq"]; SETUP = cfg["setup_por_maq"]
    CUELLOS = cfg["cuellos"]; rank = np.arange(n, dtype=float)
    job_pos = [0] * n; job_free = [0.0] * n; maq_free = [0.0] * n_maq
    last = {c: None for c in CUELLOS}
    total = sum(len(o.ruta) for o in ordenes); acciones = []
    for _ in range(total):
        best = None
        for j in range(n):
            if job_pos[j] >= len(ordenes[j].ruta):
                continue
            m, dur = ordenes[j].ruta[job_pos[j]]
            s = max(job_free[j], maq_free[m]); setup = 0.0
            if m in CUELLOS:
                lf = last[m]
                if lf is not None and lf != ordenes[j].tipo:
                    setup = SETUP[m][lf][ordenes[j].tipo]
            key = (s + setup, job_free[j], rank[j])
            if best is None or key < best[0]:
                best = (key, j, m, dur, s + setup, setup)
        _, j, m, dur, ini, setup = best
        acciones.append(j)
        fin = ini + dur; maq_free[m] = fin; job_free[j] = fin
        if m in CUELLOS:
            last[m] = ordenes[j].tipo
        job_pos[j] += 1
    return acciones


# ───────── GA op-a-op (codificacion basada en operaciones) ─────────
def _ox(p1, p2, rng):
    """JOX para codificacion basada en operaciones: conserva el segmento de p1 y rellena
    con el resto de p2 en orden, preservando el multiconjunto (cada trabajo len(ruta) veces)."""
    nn = len(p1); a, b = sorted(rng.integers(0, nn, 2)); h = [None] * nn
    h[a:b] = p1[a:b]
    quitar = {}
    for g in p1[a:b]:
        quitar[g] = quitar.get(g, 0) + 1
    resto = []
    for g in p2:
        if quitar.get(g, 0) > 0:
            quitar[g] -= 1
        else:
            resto.append(g)
    pos = 0
    for i in range(nn):
        if h[i] is None:
            h[i] = resto[pos]; pos += 1
    return h


def ga_op(ordenes, cfg, n_gen=120, pop=60, seed=0):
    """Genoma = secuencia de OPERACIONES (cada trabajo aparece len(ruta) veces).

    SEMBRADO con la solucion voraz op-a-op: sin esta siembra el GA partia de
    permutaciones aleatorias de operaciones (psimas) y, pese a >3.600 evaluaciones,
    no alcanzaba siquiera a la regla voraz (1617). Inyectar la voraz como elite
    inicial garantiza que el GA arranca >=1617 y solo puede mejorar; deja el
    competidor metaheuristico BIEN CALIBRADO en el espacio de operacion antes de
    compararlo con el agente RL (comparacion justa)."""
    rng = np.random.default_rng(seed); n = len(ordenes)
    base = []
    for j, o in enumerate(ordenes):
        base += [j] * len(o.ruta)
    def fit(s): return schedule_stream(ordenes, s, cfg)
    semilla = greedy_stream(ordenes, cfg)        # secuencia voraz earliest-start
    pobl = [semilla[:]]                            # elite: la voraz
    for _ in range(max(1, pop // 10)):             # variantes perturbadas (diversidad local)
        c = semilla[:]
        for _ in range(int(rng.integers(1, 4))):
            i, k = rng.integers(0, len(c), 2); c[i], c[k] = c[k], c[i]
        pobl.append(c)
    for _ in range(pop - len(pobl)):               # resto aleatorio (exploracion global)
        c = base[:]; rng.shuffle(c); pobl.append(c)
    fits = [fit(s) for s in pobl]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2); p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2); p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            h = _ox(p1, p2, rng)
            if rng.random() < 0.3:
                i, k = rng.integers(0, len(h), 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]; pobl = nueva; fits = [fit(s) for s in pobl]
    return float(max(fits))


def main():
    import warnings, json, os; warnings.filterwarnings("ignore")
    try:
        from estudio_ppo_toc import _wake_lock; _wake_lock(True)
    except Exception:
        pass
    cfg = cfg_c5()
    print("=" * 64)
    print("  BASELINES A NIVEL DE OPERACION (op-a-op) - Caso 5")
    print("=" * 64, flush=True)
    gr, ga = [], []
    for i, sd in enumerate(SEEDS_TEST):
        o = gen(sd)
        gr.append(greedy_op(o, cfg))
        ga.append(ga_op(o, cfg, seed=sd))
        if (i + 1) % 5 == 0:
            print(f"  ...{i+1}/{len(SEEDS_TEST)}", flush=True)
    gr = np.array(gr); ga = np.array(ga)
    print(f"\n  Greedy op-a-op : {gr.mean():7.1f} +/- {gr.std():5.1f}")
    print(f"  GA op-a-op     : {ga.mean():7.1f} +/- {ga.std():5.1f}")
    print("  (referencia nivel-orden / PlantaC5: FIFO 1040, CP-SAT 1085, NSGA-II 1377)")
    out = {"greedy_op": {"media": float(gr.mean()), "std": float(gr.std()), "raw": gr.tolist()},
           "ga_op": {"media": float(ga.mean()), "std": float(ga.std()), "raw": ga.tolist()}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json.dump(out, open(os.path.join(base, "results", "resultados_c5_l2d.json"), "w"), indent=2)
    print("\n[OK] resultados_c5_l2d.json")


if __name__ == "__main__":
    main()

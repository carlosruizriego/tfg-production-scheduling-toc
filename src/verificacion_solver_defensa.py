# -*- coding: utf-8 -*-
"""
VERIFICACION DEL ARGUMENTO "DOS PROBLEMAS" (comentarios #91/#92/#94 del tutor)
=============================================================================
NO forma parte de la memoria del TFG (no tiene anexo): es material de respaldo
para la DEFENSA, por si se cuestiona por que la secuencia derivada de un metodo
exacto queda por debajo de la del genetico.

Idea: en instancias lo bastante pequenas para enumerar TODAS las secuencias de
lanzamiento (n = 7 -> 5 040 permutaciones), se comparan cuatro magnitudes:

  M      = optimo del MODELO CP-SAT (max nº de ordenes con fin <= H en un
           programa completo, con instantes de inicio y esperas deliberadas).
  B      = optimo REAL del problema de secuencias (fuerza bruta sobre las 5 040).
  CPproj = secuencia proyectada de CP-SAT (orden por instante de inicio),
           re-simulada en el mismo evaluador que el resto de metodos.
  GA     = mejor secuencia hallada por el genetico.

Invariantes que deben cumplirse SIEMPRE (si alguna falla, el argumento del TFG
seria falso y habria que revisarlo):

  (I1) B_ordenes  <= M       nadie supera el optimo demostrado del metodo exacto
  (I2) GA_ordenes <= M       el genetico NO bate al exacto en el problema del exacto
  (I3) GA_th      <= B_th    el genetico no supera el optimo real de secuencias
  (I4) CPproj_th  <= B_th    la proyeccion es un punto subotimo de ese espacio

Y el fenomeno que el tutor senalaba queda reproducido: GA_th >= CPproj_th, a
menudo de forma estricta. Conclusion defendible: el genetico no bate al metodo
exacto, bate a la PROYECCION del metodo exacto a un espacio de decision mas pobre.

Ejecucion:  python verificacion_solver_defensa.py     (~2-3 min)
"""
import os, sys, itertools, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ortools.sat.python import cp_model
from planta_simulada import generar_instancia, PlantaSimPy

N = 7                          # 7! = 5 040 permutaciones enumerables por instancia
SEEDS = list(range(11, 27))    # 16 instancias independientes


def evaluar_h(ordenes, seq, H):
    """Mismo evaluador unico que usan todos los metodos del estudio."""
    return PlantaSimPy(ordenes, list(seq), H).ejecutar()


def cpsat_modelo(ordenes, H, limite=10.0):
    """Resuelve el modelo exacto. Devuelve (objetivo, status, secuencia proyectada)."""
    model = cp_model.CpModel()
    n = len(ordenes)
    horizon = int(sum(op.duracion for o in ordenes for op in o.operaciones)) + 1
    task = {}
    for j, o in enumerate(ordenes):
        for k, op in enumerate(o.operaciones):
            dur = max(1, int(round(op.duracion)))
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            task[j, k] = (s, e, model.NewIntervalVar(s, dur, e, f"i_{j}_{k}"))
    for j, o in enumerate(ordenes):
        for k in range(len(o.operaciones) - 1):
            model.Add(task[j, k + 1][0] >= task[j, k][1])
    maq = {}
    for j, o in enumerate(ordenes):
        for k, op in enumerate(o.operaciones):
            maq.setdefault(op.maquina, []).append(task[j, k][2])
    for ivs in maq.values():
        model.AddNoOverlap(ivs)
    done = []
    for j, o in enumerate(ordenes):
        cj = task[j, len(o.operaciones) - 1][1]
        d = model.NewBoolVar(f"done_{j}")
        model.Add(cj <= H).OnlyEnforceIf(d)
        model.Add(cj > H).OnlyEnforceIf(d.Not())
        done.append(d)
    model.Maximize(sum(done))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limite
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    # proyeccion: del programa optimo solo se conserva el ORDEN de lanzamiento
    inicios = [(j, solver.Value(task[j, 0][0])) for j in range(n)]
    seq = [j for j, _ in sorted(inicios, key=lambda x: x[1])]
    return int(round(solver.ObjectiveValue())), solver.StatusName(st), seq


def ga_local(ordenes, H, n_gen=60, pop=40, seed=0):
    """Genetico identico en operadores al del estudio (torneo + OX + swap + elitismo)."""
    rng = np.random.default_rng(seed)
    n = len(ordenes)
    fit = lambda s: evaluar_h(ordenes, s, H)["throughput"]
    pobl = [list(rng.permutation(n)) for _ in range(pop)]
    fits = [fit(s) for s in pobl]
    for _ in range(n_gen):
        nueva = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2); p1 = pobl[a] if fits[a] >= fits[b] else pobl[b]
            c, d = rng.integers(0, pop, 2); p2 = pobl[c] if fits[c] >= fits[d] else pobl[d]
            x, y = sorted(rng.integers(0, n, 2)); h = [None] * n
            h[x:y] = p1[x:y]
            resto = [z for z in p2 if z not in h]; pos = 0
            for i in range(n):
                if h[i] is None:
                    h[i] = resto[pos]; pos += 1
            if rng.random() < 0.3:
                i, k = rng.integers(0, n, 2); h[i], h[k] = h[k], h[i]
            nueva.append(h)
        nueva[0] = pobl[int(np.argmax(fits))][:]          # elitismo
        pobl = nueva; fits = [fit(s) for s in pobl]
    return evaluar_h(ordenes, pobl[int(np.argmax(fits))], H)


def fuerza_bruta(ordenes, H):
    """Optimo REAL del problema de secuencias: enumera las n! permutaciones."""
    n = len(ordenes); mejor_th, mejor_c = -1.0, -1
    for perm in itertools.permutations(range(n)):
        r = evaluar_h(ordenes, perm, H)
        if r["throughput"] > mejor_th:
            mejor_th, mejor_c = r["throughput"], r["ordenes_completadas"]
    return mejor_th, mejor_c


def calibra_H(seed, n):
    """Horizonte que deja a FIFO en ~55 % de ordenes completadas (regimen restrictivo)."""
    ordenes = generar_instancia(n_jobs=n, seed=seed)
    objetivo, mejor_H, mejor_d = 0.55 * n, 120, 1e9
    for H in range(80, 221, 10):
        d = abs(evaluar_h(ordenes, range(n), H)["ordenes_completadas"] - objetivo)
        if d < mejor_d:
            mejor_d, mejor_H = d, H
    return mejor_H


def main():
    print("n = %d ordenes  (%d permutaciones enumeradas por instancia)\n"
          % (N, math.factorial(N)))
    print("%-5s %-4s | %-7s %-8s | %-10s %-10s | %-9s %-6s | invariantes"
          % ("seed", "H", "M(mod)", "status", "B_th/ord", "GA_th/ord", "CPproj", "GA>CP"))
    print("-" * 104)

    fallos = 0
    d_ga_cp, d_B_cp, d_M_B, n_ga_gt_cp = [], [], [], 0

    for sd in SEEDS:
        H = calibra_H(sd, N)
        ordenes = generar_instancia(n_jobs=N, seed=sd)
        M, st, seqcp = cpsat_modelo(ordenes, H)
        rcp = evaluar_h(ordenes, seqcp, H)
        rga = ga_local(ordenes, H, seed=sd)
        Bth, Bc = fuerza_bruta(ordenes, H)

        i1 = Bc <= M
        i2 = rga["ordenes_completadas"] <= M
        i3 = rga["throughput"] <= Bth + 1e-6
        i4 = rcp["throughput"] <= Bth + 1e-6
        ok = all([i1, i2, i3, i4])
        fallos += (not ok)

        d_ga_cp.append(rga["throughput"] - rcp["throughput"])
        d_B_cp.append(Bth - rcp["throughput"])
        d_M_B.append(M - Bc)
        n_ga_gt_cp += (rga["throughput"] > rcp["throughput"] + 1e-6)

        print("%-5d %-4d | %-7d %-8s | %6.0f/%d %6.0f/%d | %6.0f/%d %-6s | I1=%d I2=%d I3=%d I4=%d %s"
              % (sd, H, M, st, Bth, Bc, rga["throughput"], rga["ordenes_completadas"],
                 rcp["throughput"], rcp["ordenes_completadas"],
                 "SI" if rga["throughput"] >= rcp["throughput"] else "no",
                 i1, i2, i3, i4, "" if ok else "  <-- FALLA"))

    print("-" * 104)
    print("Invariantes: I1 B_ord<=M | I2 GA_ord<=M | I3 GA_th<=B_th | I4 CPproj_th<=B_th")
    print("FALLOS: %d de %d instancias" % (fallos, len(SEEDS)))
    print("\nFENOMENO (media sobre %d instancias):" % len(SEEDS))
    print("  GA supera a la proyeccion CP-SAT en %d/%d instancias" % (n_ga_gt_cp, len(SEEDS)))
    print("  ventaja media GA - CPproj         : %.1f EUR" % np.mean(d_ga_cp))
    print("  ventaja optimo real - CPproj      : %.1f EUR   (lo que pierde la proyeccion)"
          % np.mean(d_B_cp))
    print("  holgura modelo - optimo real      : %.2f ordenes   (esperas deliberadas que el "
          "programa usa y una secuencia no puede reproducir)" % np.mean(d_M_B))


if __name__ == "__main__":
    main()

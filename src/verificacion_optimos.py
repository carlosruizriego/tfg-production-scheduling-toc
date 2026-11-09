# -*- coding: utf-8 -*-
"""
VERIFICACIONES SOBRE EL METODO EXACTO
=====================================
Reune las tres comprobaciones cuyos resultados se citan en la memoria:

  techo()          Valor del programa que obtiene CP-SAT en cada caso (apartado 7.2 y
                   tablas 7.4-7.7): media, desviacion tipica, numero de instancias en
                   las que demuestra la optimalidad y contraste frente a FIFO. Se
                   reutilizan los modelos originales de estudio_c1..c5 mediante un
                   solver que registra el valor objetivo, de modo que la cifra procede
                   exactamente de la misma formulacion que emplea el estudio.

  discretizacion() Efecto del redondeo de duraciones a enteros que exige el modelo
                   entero de CP-SAT (apartado 6.6). Resuelve cada instancia con la
                   resolucion del trabajo y con una resolucion cien veces mas fina.

  escala()         Degradacion del metodo exacto al crecer la instancia (apartado 8.3):
                   valor y distancia al optimo certificada sobre las instancias de 20,
                   25 y 30 trabajos del estudio de transferencia.

Uso:  python verificacion_optimos.py [techo|discretizacion|escala|todo]
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
from ortools.sat.python import cp_model

RESDIR = os.path.join(_BASE, "results")

# ── Solver que registra el valor objetivo sin alterar los modelos originales ──
_Base = cp_model.CpSolver


class SolverRegistrador(_Base):
    ultimo = {}

    def Solve(self, modelo, *a, **k):
        estado = _Base.Solve(self, modelo, *a, **k)
        try:
            SolverRegistrador.ultimo = {"obj": self.ObjectiveValue(),
                                        "cota": self.BestObjectiveBound(),
                                        "estado": self.StatusName(estado)}
        except Exception:
            SolverRegistrador.ultimo = {"obj": float("nan"), "cota": float("nan"),
                                        "estado": self.StatusName(estado)}
        return estado


cp_model.CpSolver = SolverRegistrador

import estudio_c1 as E1, estudio_c2 as E2, estudio_c3 as E3
import estudio_c4 as E4, estudio_c5 as E5
import planta_caso3, planta_caso4, planta_caso5 as P5, l2d_c5 as L
from planta_simulada import generar_instancia, PlantaSimPy

# (modulo, generador de instancias, escala a EUR del objetivo)
CASOS = [
    ("C1", E1, lambda s: generar_instancia(seed=s), 70.0),
    ("C2", E2, lambda s: E2.generar_ordenes_c2(seed=s), 1.0),
    ("C3", E3, lambda s: planta_caso3.generar_ordenes_c3(mix=E3.POOL[:], seed=s), 1.0),
    ("C4", E4, lambda s: planta_caso4.generar_ordenes_c4(seed=s), 1.0),
    ("C5", E5, lambda s: P5.generar_ordenes_c5(P5.POOL_DEFAULT[:] if hasattr(P5, "POOL_DEFAULT")
                                               else E5.POOL_DEFAULT[:], s), 1.0),
]


def techo(salida="optimos_cpsat.json"):
    """Valor del programa de CP-SAT en cada caso, con su contraste frente a FIFO."""
    res = {}
    for nombre, mod, gen, escala in CASOS:
        opt, fifo, n_opt = [], [], 0
        for sd in mod.SEEDS_TEST:
            ordenes = gen(sd)
            mod.metodo_cpsat(ordenes)
            u = SolverRegistrador.ultimo
            opt.append(float(u["obj"]) * escala)
            n_opt += (u["estado"] == "OPTIMAL")
            fifo.append(mod.metodo_fifo(ordenes)["throughput"])
        opt, fifo = np.array(opt), np.array(fifo)
        res[nombre] = {"media": round(float(opt.mean()), 1), "std": round(float(opt.std()), 1),
                       "optimal": n_opt, "n": len(opt), "fifo": round(float(fifo.mean()), 1),
                       "p_vs_fifo": float(stats.wilcoxon(opt, fifo).pvalue)}
        r = res[nombre]
        print("  %s  %8.1f +- %5.1f EUR | optimo demostrado %2d/%d | FIFO %7.1f | p = %.2g"
              % (nombre, r["media"], r["std"], r["optimal"], r["n"], r["fifo"], r["p_vs_fifo"]),
              flush=True)
    json.dump(res, open(os.path.join(RESDIR, salida), "w"), indent=1)
    return res


def _modelo_c1(ordenes, escala, limite=60.0):
    """Modelo del Caso 1 con resolucion temporal parametrizable."""
    m = cp_model.CpModel()
    H = int(round(E1.HORIZONTE * escala))
    horizonte = int(sum(op.duracion for o in ordenes for op in o.operaciones) * escala) + escala + 1
    tarea = {}
    for j, o in enumerate(ordenes):
        for k, op in enumerate(o.operaciones):
            d = max(1, int(round(op.duracion * escala)))
            ini = m.NewIntVar(0, horizonte, ""); fin = m.NewIntVar(0, horizonte, "")
            tarea[j, k] = (ini, fin, m.NewIntervalVar(ini, d, fin, ""), op.maquina)
    for j, o in enumerate(ordenes):
        for k in range(len(o.operaciones) - 1):
            m.Add(tarea[j, k + 1][0] >= tarea[j, k][1])
    maq = {}
    for j, o in enumerate(ordenes):
        for k, op in enumerate(o.operaciones):
            maq.setdefault(op.maquina, []).append(tarea[j, k][2])
    for ivs in maq.values():
        m.AddNoOverlap(ivs)
    hechos = []
    for j, o in enumerate(ordenes):
        fin_j = tarea[j, len(o.operaciones) - 1][1]
        b = m.NewBoolVar("")
        m.Add(fin_j <= H).OnlyEnforceIf(b); m.Add(fin_j > H).OnlyEnforceIf(b.Not())
        hechos.append(b)
    m.Maximize(sum(hechos))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = limite
    s.parameters.num_search_workers = 8
    estado = s.Solve(m)
    return int(round(s.ObjectiveValue())), s.StatusName(estado)


def discretizacion(n_inst=15):
    """El redondeo de duraciones a enteros, altera el optimo del modelo?"""
    iguales = 0
    for sd in list(E1.SEEDS_TEST)[:n_inst]:
        ordenes = generar_instancia(seed=sd)
        a, _ = _modelo_c1(ordenes, 1)      # resolucion del trabajo (1 u.t.)
        b, _ = _modelo_c1(ordenes, 100)    # resolucion cien veces mas fina
        iguales += (a == b)
        print("  semilla %d: %d ordenes con resolucion 1 | %d con resolucion 100  -> %s"
              % (sd, a, b, "coincide" if a == b else "DIFIERE"), flush=True)
    print("  Coinciden %d de %d instancias" % (iguales, n_inst))
    return iguales, n_inst


def _instancia_n(seed, n_trabajos):
    """Instancia del Caso 5 con un numero arbitrario de trabajos (estudio de transferencia)."""
    pool = (list(P5.FAMILIAS) * (n_trabajos // len(P5.FAMILIAS) + 1))[:n_trabajos]
    return P5.generar_ordenes_c5(pool, seed)


def escala(tamanos=(20, 25, 30), salida="escala_cpsat.json"):
    """Como se degrada el metodo exacto al crecer la instancia."""
    res = {}
    for nj in tamanos:
        vals, gaps, n_opt = [], [], 0
        for sd in L.SEEDS_TEST:
            E5.metodo_cpsat(_instancia_n(sd, nj))
            u = SolverRegistrador.ultimo
            vals.append(float(u["obj"])); n_opt += (u["estado"] == "OPTIMAL")
            if u["obj"] and u["obj"] > 0 and np.isfinite(u["cota"]):
                gaps.append(abs(u["cota"] - u["obj"]) / u["obj"])
        v = np.array(vals)
        res[nj] = {"media": round(float(v.mean()), 1), "std": round(float(v.std()), 1),
                   "optimal": n_opt, "n": len(v),
                   "gap_medio": round(float(np.mean(gaps)) * 100, 1) if gaps else None}
        r = res[nj]
        print("  %2d trabajos: %7.1f +- %5.1f EUR | optimo demostrado %2d/%d | gap medio %.1f %%"
              % (nj, r["media"], r["std"], r["optimal"], r["n"], r["gap_medio"]), flush=True)
    json.dump(res, open(os.path.join(RESDIR, salida), "w"), indent=1)
    return res


if __name__ == "__main__":
    que = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if que in ("techo", "todo"):
        print("\n[1] Valor del programa de CP-SAT por caso"); techo()
    if que in ("discretizacion", "todo"):
        print("\n[2] Efecto de la discretizacion temporal (Caso 1)"); discretizacion()
    if que in ("escala", "todo"):
        print("\n[3] Degradacion del metodo exacto con el tamano"); escala()

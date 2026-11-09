# -*- coding: utf-8 -*-
"""Techo (optimo demostrado del modelo CP-SAT) en C2, C3 y C4, reutilizando
los modelos originales de cada estudio via un solver que registra el objetivo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ortools.sat.python import cp_model

_orig = cp_model.CpSolver
class Rec(_orig):
    ult = {}
    def Solve(self, model, *a, **kw):
        st = _orig.Solve(self, model, *a, **kw)
        try: Rec.ult = {"obj": self.ObjectiveValue(), "st": self.StatusName(st)}
        except Exception: Rec.ult = {"obj": float("nan"), "st": self.StatusName(st)}
        return st
cp_model.CpSolver = Rec

import estudio_c2 as E2, estudio_c3 as E3, estudio_c4 as E4
import planta_caso3, planta_caso4

CASOS = [
 ("Caso 2", E2, lambda s: E2.generar_ordenes_c2(seed=s)),
 ("Caso 3", E3, lambda s: planta_caso3.generar_ordenes_c3(mix=E3.POOL[:], seed=s)),
 ("Caso 4", E4, lambda s: planta_caso4.generar_ordenes_c4(seed=s)),
]
for nombre, mod, gen in CASOS:
    techo=[]; proy=[]; ga=[]; nopt=0
    for sd in mod.SEEDS_TEST:
        o = gen(sd)
        r = mod.metodo_cpsat(o)
        techo.append(float(Rec.ult["obj"])); nopt += (Rec.ult["st"]=="OPTIMAL")
        proy.append(r["throughput"])
        ga.append(mod.metodo_ga(o, seed=sd)["throughput"])
    dom = sum(1 for a,b in zip(techo,ga) if a >= b-1e-9)
    print("%s  (n=%d)" % (nombre, len(techo)))
    print("   Techo: optimo demostrado del modelo : %7.1f EUR   (OPTIMAL en %d/%d)"
          % (np.mean(techo), nopt, len(techo)))
    print("   Genetico                            : %7.1f EUR" % np.mean(ga))
    print("   CP-SAT proyectado                   : %7.1f EUR" % np.mean(proy))
    print("   El techo domina al genetico en %d de %d instancias" % (dom, len(techo)))
    print()

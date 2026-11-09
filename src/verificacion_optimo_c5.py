# -*- coding: utf-8 -*-
"""Valor del modelo CP-SAT en el Caso 5 (20 s por instancia). A esta escala no
se espera demostrar optimalidad: la cifra sera la mejor solucion hallada."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_RES_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "optimos_cpsat.json")
import numpy as np
from scipy import stats
from ortools.sat.python import cp_model
_o = cp_model.CpSolver
class Rec(_o):
    ult={}
    def Solve(self, m, *a, **k):
        st=_o.Solve(self,m,*a,**k)
        try: Rec.ult={"obj":self.ObjectiveValue(),"st":self.StatusName(st)}
        except Exception: Rec.ult={"obj":float("nan"),"st":self.StatusName(st)}
        return st
cp_model.CpSolver = Rec
import estudio_c5 as E5, planta_caso5
seeds = getattr(E5, "SEEDS_TEST", list(range(100,130)))
mod=[]; proy=[]; fifo=[]; nopt=0
for sd in seeds:
    o = planta_caso5.generar_ordenes_c5(pool=E5.POOL_DEFAULT[:], seed=sd)
    r = E5.metodo_cpsat(o)
    mod.append(float(Rec.ult["obj"])); nopt += (Rec.ult["st"]=="OPTIMAL")
    proy.append(r["throughput"]); fifo.append(E5.metodo_fifo(o)["throughput"])
mod=np.array(mod); proy=np.array(proy); fifo=np.array(fifo)
p = stats.wilcoxon(mod, fifo).pvalue
print("Caso 5 (n=%d, limite 20 s)" % len(mod))
print("  Modelo CP-SAT (mejor hallado) : %7.1f +- %5.1f EUR   (OPTIMAL en %d/%d)"
      % (mod.mean(), mod.std(), nopt, len(mod)))
print("  Secuencia derivada (lo actual): %7.1f EUR" % proy.mean())
print("  FIFO                          : %7.1f EUR   (p modelo vs FIFO = %.2g)" % (fifo.mean(), p))
print("  Genetico (referencia doc)     :  1377   EUR")
d=json.load(open(_RES_JSON))
d["C5"]={"media":round(float(mod.mean()),1),"std":round(float(mod.std()),1),
         "optimal":nopt,"n":len(mod),"p_vs_fifo":float(p),
         "fifo":round(float(fifo.mean()),1),"proyeccion":round(float(proy.mean()),1),"ordenes":None}
json.dump(d,open(_RES_JSON,"w"),indent=1)
print("\n[OK] anadido C5 a optimos_cpsat.json")

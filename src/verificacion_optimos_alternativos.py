# -*- coding: utf-8 -*-
"""Entre los MUCHOS programas optimos del modelo, cuanto varia el valor de su proyeccion?
Y cuanto se recupera eligiendo el mejor? Caso 1, 30 instancias del estudio."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ortools.sat.python import cp_model
from planta_simulada import generar_instancia, PlantaSimPy
import estudio_c1 as E
H = E.HORIZONTE; MARGEN = 70.0; R = 20

def construir(ordenes):
    m = cp_model.CpModel(); n=len(ordenes)
    hz=int(sum(op.duracion for o in ordenes for op in o.operaciones))+1
    task={}
    for j,o in enumerate(ordenes):
        for k,op in enumerate(o.operaciones):
            d=max(1,int(round(op.duracion)))
            s=m.NewIntVar(0,hz,''); e=m.NewIntVar(0,hz,'')
            task[j,k]=(s,e,m.NewIntervalVar(s,d,e,''))
    for j,o in enumerate(ordenes):
        for k in range(len(o.operaciones)-1): m.Add(task[j,k+1][0]>=task[j,k][1])
    maq={}
    for j,o in enumerate(ordenes):
        for k,op in enumerate(o.operaciones): maq.setdefault(op.maquina,[]).append(task[j,k][2])
    for ivs in maq.values(): m.AddNoOverlap(ivs)
    done=[]
    for j,o in enumerate(ordenes):
        cj=task[j,len(o.operaciones)-1][1]; d=m.NewBoolVar('')
        m.Add(cj<=H).OnlyEnforceIf(d); m.Add(cj>H).OnlyEnforceIf(d.Not()); done.append(d)
    return m, task, done, n, hz

def orden_de(sol, task, n):
    return [j for j,_ in sorted([(j, sol.Value(task[j,0][0])) for j in range(n)], key=lambda x:x[1])]

base=[]; mejor=[]; peor=[]; ga=[]; techo=[]
for sd in E.SEEDS_TEST:
    o=generar_instancia(seed=sd)
    m,task,done,n,hz = construir(o)
    m.Maximize(sum(done))
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=10; s.parameters.num_search_workers=8
    s.Solve(m); opt=int(round(s.ObjectiveValue()))
    techo.append(opt*MARGEN)
    base.append(PlantaSimPy(o, orden_de(s,task,n), H).ejecutar()["throughput"])
    # muestrear el frente de optimos: mismo valor objetivo, semillas de busqueda distintas
    vals=[]
    for r in range(R):
        m2,task2,done2,n2,hz2 = construir(o)
        m2.Add(sum(done2)==opt)
        s2=cp_model.CpSolver(); s2.parameters.max_time_in_seconds=5
        s2.parameters.num_search_workers=1; s2.parameters.random_seed=r*7+1
        s2.parameters.randomize_search=True
        if s2.Solve(m2) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            vals.append(PlantaSimPy(o, orden_de(s2,task2,n2), H).ejecutar()["throughput"])
    if vals: mejor.append(max(vals)); peor.append(min(vals))
    ga.append(E.metodo_ga(o, seed=sd)["throughput"])

print("Caso 1, 30 instancias, %d programas optimos muestreados por instancia\n" % R)
print("  Techo: optimo demostrado del modelo        : %6.1f EUR" % np.mean(techo))
print("  Genetico                                   : %6.1f EUR" % np.mean(ga))
print("  ---")
print("  MEJOR proyeccion entre los optimos         : %6.1f EUR" % np.mean(mejor))
print("  Proyeccion actual (la que reporta el TFG)  : %6.1f EUR" % np.mean(base))
print("  PEOR proyeccion entre los optimos          : %6.1f EUR" % np.mean(peor))
print()
print("  Horquilla media entre optimos: %.1f EUR" % (np.mean(mejor)-np.mean(peor)))
print("  Ganancia por elegir bien     : %.1f EUR (%+.0f%%)" %
      (np.mean(mejor)-np.mean(base), 100*(np.mean(mejor)-np.mean(base))/np.mean(base)))
gan=sum(1 for a,b in zip(mejor,ga) if a>b+1e-9)
print("  Aun eligiendo el mejor optimo, supera al genetico en %d de 30 instancias" % gan)

# -*- coding: utf-8 -*-
"""Cuanto vale el OPTIMO DEL MODELO de CP-SAT frente a lo que la tabla le atribuye.
Sobre las 30 instancias de prueba del Caso 1 (las mismas del estudio)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ortools.sat.python import cp_model
from planta_simulada import generar_instancia, PlantaSimPy
import estudio_c1 as E

H = E.HORIZONTE
MARGEN = 70.0

def modelo(ordenes, limite=10.0):
    m = cp_model.CpModel(); n=len(ordenes)
    horizon=int(sum(op.duracion for o in ordenes for op in o.operaciones))+1
    task={}
    for j,o in enumerate(ordenes):
        for k,op in enumerate(o.operaciones):
            d=max(1,int(round(op.duracion)))
            s=m.NewIntVar(0,horizon,''); e=m.NewIntVar(0,horizon,'')
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
    m.Maximize(sum(done))
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=limite
    s.parameters.num_search_workers=8
    st=s.Solve(m)
    seq=[j for j,_ in sorted([(j,s.Value(task[j,0][0])) for j in range(n)], key=lambda x:x[1])]
    return int(round(s.ObjectiveValue())), s.StatusName(st), seq

tech=[]; proy=[]; ga=[]; opt=0
for sd in E.SEEDS_TEST:
    o=generar_instancia(seed=sd)
    obj,st,seq=modelo(o)
    opt += (st=="OPTIMAL")
    tech.append(obj*MARGEN)
    proy.append(PlantaSimPy(o,seq,H).ejecutar()["throughput"])
    ga.append(E.metodo_ga(o, seed=sd)["throughput"])
print("Caso 1, 30 instancias de prueba (mismas del estudio)")
print("  Optimo DEMOSTRADO del modelo CP-SAT : %6.1f EUR   (OPTIMAL en %d/30)" % (np.mean(tech), opt))
print("  Genetico (lo que gana la tabla)     : %6.1f EUR" % np.mean(ga))
print("  CP-SAT proyectado (lo que reporta)  : %6.1f EUR" % np.mean(proy))
print()
dom=sum(1 for a,b in zip(tech,ga) if a>=b-1e-9)
print("  El optimo del modelo domina al genetico en %d de 30 instancias" % dom)
print("  Coste medio de la proyeccion: %.1f EUR (%.0f%% del techo)" %
      (np.mean(tech)-np.mean(proy), 100*(np.mean(tech)-np.mean(proy))/np.mean(tech)))

# -*- coding: utf-8 -*-
"""La proyeccion actual (orden por inicio de la 1a operacion) es la mejor regla posible?
Compara cuatro reglas de proyeccion del MISMO programa optimo. Caso 1, 30 instancias."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ortools.sat.python import cp_model
from planta_simulada import generar_instancia, PlantaSimPy, N_MACHINES
import estudio_c1 as E
H=E.HORIZONTE; MARGEN=70.0

def resolver(o):
    m=cp_model.CpModel(); n=len(o)
    hz=int(sum(op.duracion for x in o for op in x.operaciones))+1
    task={}
    for j,x in enumerate(o):
        for k,op in enumerate(x.operaciones):
            d=max(1,int(round(op.duracion)))
            s=m.NewIntVar(0,hz,''); e=m.NewIntVar(0,hz,'')
            task[j,k]=(s,e,m.NewIntervalVar(s,d,e,''),op.maquina)
    for j,x in enumerate(o):
        for k in range(len(x.operaciones)-1): m.Add(task[j,k+1][0]>=task[j,k][1])
    maq={}
    for j,x in enumerate(o):
        for k,op in enumerate(x.operaciones): maq.setdefault(op.maquina,[]).append(task[j,k][2])
    for ivs in maq.values(): m.AddNoOverlap(ivs)
    done=[]
    for j,x in enumerate(o):
        cj=task[j,len(x.operaciones)-1][1]; d=m.NewBoolVar('')
        m.Add(cj<=H).OnlyEnforceIf(d); m.Add(cj>H).OnlyEnforceIf(d.Not()); done.append(d)
    m.Maximize(sum(done))
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=10; s.parameters.num_search_workers=8
    s.Solve(m)
    return s, task, n, int(round(s.ObjectiveValue()))

def cuello(o):
    carga=np.zeros(N_MACHINES)
    for x in o:
        for op in x.operaciones: carga[op.maquina]+=op.duracion
    return int(np.argmax(carga))

res={k:[] for k in ("inicio 1a op","fin ultima op","inicio en cuello","completadas primero")}
ga=[]; techo=[]
for sd in E.SEEDS_TEST:
    o=generar_instancia(seed=sd)
    s,task,n,obj=resolver(o); techo.append(obj*MARGEN)
    c=cuello(o)
    ini=[(j,s.Value(task[j,0][0])) for j in range(n)]
    fin=[(j,s.Value(task[j,len(o[j].operaciones)-1][1])) for j in range(n)]
    inicuello=[]
    for j in range(n):
        t=[s.Value(task[j,k][0]) for k in range(len(o[j].operaciones)) if task[j,k][3]==c]
        inicuello.append((j, t[0] if t else 10**6))
    # completadas primero: las que caben dentro del horizonte, por fin; luego el resto
    dentro=sorted([p for p in fin if p[1]<=H], key=lambda x:x[1])
    fuera =sorted([p for p in fin if p[1]> H], key=lambda x:x[1])
    reglas={
      "inicio 1a op":       [j for j,_ in sorted(ini, key=lambda x:x[1])],
      "fin ultima op":      [j for j,_ in sorted(fin, key=lambda x:x[1])],
      "inicio en cuello":   [j for j,_ in sorted(inicuello, key=lambda x:x[1])],
      "completadas primero":[j for j,_ in dentro]+[j for j,_ in fuera],
    }
    for k,seq in reglas.items():
        res[k].append(PlantaSimPy(o,seq,H).ejecutar()["throughput"])
    ga.append(E.metodo_ga(o, seed=sd)["throughput"])

print("Caso 1, 30 instancias. Mismo programa optimo, distintas reglas de proyeccion:\n")
print("  Techo (optimo demostrado)        : %6.1f EUR" % np.mean(techo))
print("  Genetico                         : %6.1f EUR" % np.mean(ga))
print("  " + "-"*44)
for k,v in sorted(res.items(), key=lambda x:-np.mean(x[1])):
    marca = "  <- la del TFG" if k=="inicio 1a op" else ""
    print("  %-32s : %6.1f EUR%s" % (k, np.mean(v), marca))
mejor=max(res, key=lambda k: np.mean(res[k]))
print("\n  Mejor regla: '%s' (%+.1f EUR sobre la actual)" %
      (mejor, np.mean(res[mejor])-np.mean(res["inicio 1a op"])))
gan=sum(1 for a,b in zip(res[mejor],ga) if a>b+1e-9)
print("  Con la mejor regla, supera al genetico en %d de 30 instancias" % gan)

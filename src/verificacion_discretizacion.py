# -*- coding: utf-8 -*-
"""VERIFICACION DE LA DISCRETIZACION TEMPORAL (comentario #92 del tutor)
========================================================================
Respalda la afirmacion del apartado 6.6 de la memoria. Mide si el redondeo de duraciones a enteros del modelo CP-SAT altera su optimo.
Compara resolver con duraciones redondeadas a 1 u.t. (lo que hace el TFG) frente a
resolver con resolucion 100x (equivalente a 0,01 u.t., practicamente continuo)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ortools.sat.python import cp_model
from planta_simulada import generar_instancia
HOR = 150

def resolver(ordenes, escala, limite=60.0):
    m = cp_model.CpModel()
    H = int(round(HOR*escala))
    horizon = int(sum(op.duracion for o in ordenes for op in o.operaciones)*escala)+escala+1
    task={}
    for j,o in enumerate(ordenes):
        for k,op in enumerate(o.operaciones):
            dur=max(1,int(round(op.duracion*escala)))
            s=m.NewIntVar(0,horizon,'s%d_%d'%(j,k)); e=m.NewIntVar(0,horizon,'e%d_%d'%(j,k))
            task[j,k]=(s,e,m.NewIntervalVar(s,dur,e,'i%d_%d'%(j,k)))
    for j,o in enumerate(ordenes):
        for k in range(len(o.operaciones)-1): m.Add(task[j,k+1][0]>=task[j,k][1])
    maq={}
    for j,o in enumerate(ordenes):
        for k,op in enumerate(o.operaciones): maq.setdefault(op.maquina,[]).append(task[j,k][2])
    for ivs in maq.values(): m.AddNoOverlap(ivs)
    done=[]
    for j,o in enumerate(ordenes):
        cj=task[j,len(o.operaciones)-1][1]; d=m.NewBoolVar('d%d'%j)
        m.Add(cj<=H).OnlyEnforceIf(d); m.Add(cj>H).OnlyEnforceIf(d.Not()); done.append(d)
    m.Maximize(sum(done))
    sol=cp_model.CpSolver(); sol.parameters.max_time_in_seconds=limite
    sol.parameters.num_search_workers=8
    st=sol.Solve(m)
    return int(round(sol.ObjectiveValue())), sol.StatusName(st)

print("%-6s %-16s %-16s %s" % ("seed","escala 1 (TFG)","escala 100","coincide"))
print("-"*58)
ig=0; tot=0
for sd in range(100,115):
    o=generar_instancia(seed=sd)
    a,sa=resolver(o,1); b,sb=resolver(o,100)
    tot+=1; ig+= (a==b)
    print("%-6d %-16s %-16s %s" % (sd,"%d (%s)"%(a,sa),"%d (%s)"%(b,sb),"si" if a==b else "NO"))
print("-"*58)
print("Coinciden %d de %d instancias" % (ig,tot))

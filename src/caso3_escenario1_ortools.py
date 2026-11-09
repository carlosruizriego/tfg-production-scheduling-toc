"""
CASO 3 - ESCENARIO 1: OR-Tools CP-SAT Flow Shop 2 Productos
============================================================
Flow Shop puro: secuencia fija M0->M1->M2->M3->M4.
OR-Tools minimiza el makespan y calcula el coste de oportunidad
del mix X/Y elegido respecto al mix optimo segun T/CU.
"""

from ortools.sat.python import cp_model
from planta_caso3 import (
    generar_ordenes_c3, PlantaC3, imprimir_metricas_c3,
    coste_oportunidad_c3, N_MACHINES_C3, SIM_TIME_C3, MIX_DEFAULT
)
from analisis_oportunidad import imprimir_coste_oportunidad, comparar_co_escenarios
import time


def resolver_flowshop_cpsat(ordenes, tiempo_limite=30.0):
    model = cp_model.CpModel()
    horizon = int(sum(sum(o.tiempos) for o in ordenes)) + 1

    task_vars = {}
    for j, orden in enumerate(ordenes):
        for m in range(N_MACHINES_C3):
            dur = max(1, int(orden.tiempos[m]))
            s  = model.NewIntVar(0, horizon, f"s_{j}_{m}")
            e  = model.NewIntVar(0, horizon, f"e_{j}_{m}")
            iv = model.NewIntervalVar(s, dur, e, f"i_{j}_{m}")
            task_vars[j, m] = (s, e, iv)

    # Precedencia dentro de cada orden (Flow Shop: M0->M1->...->M4)
    for j in range(len(ordenes)):
        for m in range(N_MACHINES_C3 - 1):
            model.Add(task_vars[j, m + 1][0] >= task_vars[j, m][1])

    # No solapamiento por maquina
    for m in range(N_MACHINES_C3):
        model.AddNoOverlap([task_vars[j, m][2] for j in range(len(ordenes))])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan,
        [task_vars[j, N_MACHINES_C3 - 1][1] for j in range(len(ordenes))])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_limite
    solver.parameters.log_search_progress = False

    print(f"\n[OR-Tools C3] Flow Shop: {len(ordenes)} ordenes, {N_MACHINES_C3} maquinas...")
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        mk = solver.Value(makespan)
        print(f"[OR-Tools C3] {'OPTIMO' if status==cp_model.OPTIMAL else 'FACTIBLE'}"
              f" - Makespan: {mk} u.t. - Tiempo: {elapsed:.2f}s")
        inicios = [(j, solver.Value(task_vars[j, 0][0])) for j in range(len(ordenes))]
        return [j for j, _ in sorted(inicios, key=lambda x: x[1])], mk
    print("[OR-Tools C3] Sin solucion - usando FIFO")
    return list(range(len(ordenes))), None


def main():
    print("=" * 60)
    print("  CASO 3 - ESCENARIO 1: OR-Tools Flow Shop + CO")
    print("=" * 60)

    ordenes = generar_ordenes_c3()

    fifo = PlantaC3(ordenes, list(range(len(ordenes))), SIM_TIME_C3)
    res_fifo = fifo.ejecutar()
    imprimir_metricas_c3(res_fifo, "FIFO (baseline)")
    co_fifo = coste_oportunidad_c3(res_fifo, ordenes)
    imprimir_coste_oportunidad(co_fifo, "CO - FIFO")

    seq, mk = resolver_flowshop_cpsat(ordenes, tiempo_limite=30.0)
    planta_opt = PlantaC3(ordenes, seq, SIM_TIME_C3)
    res_opt = planta_opt.ejecutar()
    imprimir_metricas_c3(res_opt, "OR-Tools CP-SAT")
    co_opt = coste_oportunidad_c3(res_opt, ordenes)
    imprimir_coste_oportunidad(co_opt, "CO - OR-Tools")

    comparar_co_escenarios([co_fifo, co_opt], ["FIFO", "OR-Tools"])

    print(f"\n  Mejora Throughput : {res_opt['throughput']-res_fifo['throughput']:+.2f} EUR")
    print(f"  Reduccion CO      : {co_fifo['CO_total']-co_opt['CO_total']:+.2f} EUR")
    print(f"  Mix recomendado   : {co_opt['mix_recomendado']}")
    print(f"  T/CU X={co_opt['tcu_por_producto'].get('X',0):.2f}  "
          f"Y={co_opt['tcu_por_producto'].get('Y',0):.2f}\n")


if __name__ == "__main__":
    main()

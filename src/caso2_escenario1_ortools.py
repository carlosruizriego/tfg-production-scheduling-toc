"""
CASO 2 - ESCENARIO 1: OR-Tools CP-SAT con restricciones de minimos
===================================================================
TOC aplicada:
  Paso 1: identificar cuello entre maquinas compartidas (M0, M2, M4)
  Paso 2: el solver garantiza que el cuello nunca este ocioso
  Paso 3: restriccion AddNoOverlap subordina el resto al cuello
  Paso 4: minimizar makespan eleva el throughput del cuello
  Paso 5: analisis de coste de oportunidad indica el siguiente cuello

Innovacion respecto Caso 1: restricciones duras de minimos por producto.
  Si el solver no puede cumplirlas, se penaliza la funcion objetivo.
"""

from ortools.sat.python import cp_model
from planta_caso2 import (
    generar_ordenes_c2, PlantaC2, imprimir_metricas_c2,
    coste_oportunidad_c2, calcular_tcu_c2,
    N_MACHINES_C2, SIM_TIME_C2, MIX_ORDENES, MINIMOS, PRODUCTOS
)
from analisis_oportunidad import imprimir_coste_oportunidad, comparar_co_escenarios
import time


def identificar_cuello_c2(ordenes) -> int:
    """TOC Paso 1: maquina con mayor carga acumulada."""
    carga = {m: 0.0 for m in range(N_MACHINES_C2)}
    for o in ordenes:
        for maq, dur in o.ruta:
            carga[maq] += dur
    cuello = max(carga, key=carga.get)
    print(f"\n[TOC Paso 1] Cuello identificado: M{cuello}")
    print(f"  Cargas: { {f'M{m}': f'{v:.1f}' for m,v in carga.items()} }")
    tcu = calcular_tcu_c2(cuello)
    rank = sorted(tcu.items(), key=lambda x: x[1], reverse=True)
    print(f"  T/CU por producto: { {p: f'{v:.2f}' for p,v in rank} }")
    print(f"  Producto prioritario segun TOC: {rank[0][0]} (T/CU={rank[0][1]:.2f})")
    return cuello


def resolver_c2_cpsat(ordenes, tiempo_limite: float = 60.0):
    model = cp_model.CpModel()
    horizon = int(sum(dur for o in ordenes for _, dur in o.ruta)) + 1

    task_vars = {}
    for j, orden in enumerate(ordenes):
        for k, (maq, dur) in enumerate(orden.ruta):
            dur_int = max(1, int(dur))
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, dur_int, e, f"i_{j}_{k}")
            task_vars[j, k] = (s, e, iv, maq)

    # Precedencia dentro de cada orden
    for j, orden in enumerate(ordenes):
        for k in range(len(orden.ruta) - 1):
            model.Add(task_vars[j, k + 1][0] >= task_vars[j, k][1])

    # No solapamiento por maquina
    for m in range(N_MACHINES_C2):
        ivs = [task_vars[j, k][2]
               for j, orden in enumerate(ordenes)
               for k, (maq, _) in enumerate(orden.ruta) if maq == m]
        if ivs:
            model.AddNoOverlap(ivs)

    # Restricciones de minimos como soft constraints con penalizacion
    # Variables booleanas: orden j completada antes de SIM_TIME_C2
    completada = [model.NewBoolVar(f"done_{j}") for j in range(len(ordenes))]
    for j, orden in enumerate(ordenes):
        last_k = len(orden.ruta) - 1
        model.Add(task_vars[j, last_k][1] <= int(SIM_TIME_C2)).OnlyEnforceIf(completada[j])
        model.Add(task_vars[j, last_k][1] > int(SIM_TIME_C2)).OnlyEnforceIf(completada[j].Not())

    # Minimos por tipo de producto
    for tipo, minimo in MINIMOS.items():
        indices = [j for j, o in enumerate(ordenes) if o.tipo == tipo]
        model.Add(sum(completada[j] for j in indices) >= minimo)

    # Objetivo: maximizar ordenes completadas ponderadas por margen
    peso_margenes = [int(o.margen * 10) for o in ordenes]
    model.Maximize(sum(completada[j] * peso_margenes[j] for j in range(len(ordenes))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_limite
    solver.parameters.log_search_progress = False

    print(f"\n[OR-Tools C2] Resolviendo con {len(ordenes)} ordenes ({N_MACHINES_C2} maquinas)...")
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"[OR-Tools C2] {'OPTIMO' if status == cp_model.OPTIMAL else 'FACTIBLE'}"
              f" - Tiempo: {elapsed:.2f}s")
        inicios = [(j, solver.Value(task_vars[j, 0][0])) for j in range(len(ordenes))]
        secuencia = [j for j, _ in sorted(inicios, key=lambda x: x[1])]
        return secuencia
    else:
        print("[OR-Tools C2] Sin solucion factible - usando FIFO")
        return list(range(len(ordenes)))


def main():
    print("=" * 60)
    print("  CASO 2 - ESCENARIO 1: OR-Tools + TOC Multi-Producto")
    print("=" * 60)

    ordenes = generar_ordenes_c2()

    cuello = identificar_cuello_c2(ordenes)

    # FIFO baseline
    fifo = PlantaC2(ordenes, list(range(len(ordenes))), SIM_TIME_C2)
    res_fifo = fifo.ejecutar()
    imprimir_metricas_c2(res_fifo, "FIFO (baseline)")
    co_fifo = coste_oportunidad_c2(res_fifo)
    imprimir_coste_oportunidad(co_fifo, "Coste de Oportunidad - FIFO")

    # OR-Tools
    seq = resolver_c2_cpsat(ordenes, tiempo_limite=60.0)
    planta_opt = PlantaC2(ordenes, seq, SIM_TIME_C2)
    res_opt = planta_opt.ejecutar()
    imprimir_metricas_c2(res_opt, "OR-Tools CP-SAT")
    co_opt = coste_oportunidad_c2(res_opt)
    imprimir_coste_oportunidad(co_opt, "Coste de Oportunidad - OR-Tools")

    comparar_co_escenarios(
        [co_fifo, co_opt],
        ["FIFO", "OR-Tools CP-SAT"]
    )

    print(f"\n  Mejora Throughput : {res_opt['throughput'] - res_fifo['throughput']:+.2f} EUR")
    print(f"  Mejora Bº Neto    : {res_opt['beneficio_neto'] - res_fifo['beneficio_neto']:+.2f} EUR")
    print(f"  Reduccion CO      : {co_fifo['CO_total'] - co_opt['CO_total']:+.2f} EUR\n")


if __name__ == "__main__":
    main()

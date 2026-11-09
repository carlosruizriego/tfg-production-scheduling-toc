from ortools.sat.python import cp_model
from planta_simulada import (
    generar_instancia, PlantaSimPy, imprimir_metricas,
    N_MACHINES, M_JOBS, SIM_TIME
)
import time


# ──────────────────────────────────────────────
# Paso 1 TOC: identificar restriccion antes de optimizar
# ──────────────────────────────────────────────
def identificar_cuello(ordenes) -> int:
    """
    Calcula la carga total por maquina sumando duraciones
    de todas las operaciones que la requieren.
    La maquina con mayor carga es el cuello de botella inicial.
    """
    carga = {m: 0.0 for m in range(N_MACHINES)}
    for orden in ordenes:
        for op in orden.operaciones:
            carga[op.maquina] += op.duracion
    cuello = max(carga, key=carga.get)
    print(f"\n[TOC Paso 1] Restriccion identificada: Maquina {cuello}")
    print(f"  Cargas por maquina: { {f'M{m}': f'{v:.1f}' for m,v in carga.items()} }")
    return cuello


# ──────────────────────────────────────────────
# Solver CP-SAT: Job Shop Scheduling
# ──────────────────────────────────────────────
def resolver_jobshop_cpsat(ordenes, tiempo_limite: float = 30.0):
    """
    Formula el Job Shop como un problema CP-SAT.
    Minimiza el makespan (tiempo hasta completar todas las ordenes).
    Esto implementa TOC pasos 2-4: explotar y subordinar al cuello.
    """
    model = cp_model.CpModel()

    # Horizonte temporal maximo (suma de todas las duraciones)
    horizon = int(sum(op.duracion for o in ordenes for op in o.operaciones)) + 1

    # Variables de intervalo para cada operacion
    # task_vars[j][k] = (start, end, interval_var)
    task_vars = {}
    for j, orden in enumerate(ordenes):
        for k, op in enumerate(orden.operaciones):
            dur = int(op.duracion)
            suffix = f"_{j}_{k}"
            start = model.NewIntVar(0, horizon, "s" + suffix)
            end   = model.NewIntVar(0, horizon, "e" + suffix)
            interval = model.NewIntervalVar(start, dur, end, "i" + suffix)
            task_vars[j, k] = (start, end, interval)

    # Restriccion: operaciones del mismo job en secuencia
    for j, orden in enumerate(ordenes):
        for k in range(len(orden.operaciones) - 1):
            model.Add(task_vars[j, k + 1][0] >= task_vars[j, k][1])

    # Restriccion: no solapamiento en la misma maquina
    maquina_intervals = {m: [] for m in range(N_MACHINES)}
    for j, orden in enumerate(ordenes):
        for k, op in enumerate(orden.operaciones):
            maquina_intervals[op.maquina].append(task_vars[j, k][2])

    for m in range(N_MACHINES):
        if maquina_intervals[m]:
            model.AddNoOverlap(maquina_intervals[m])

    # Funcion objetivo: minimizar makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan,
        [task_vars[j, len(o.operaciones) - 1][1] for j, o in enumerate(ordenes)]
    )
    model.Minimize(makespan)

    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tiempo_limite
    solver.parameters.log_search_progress = False

    print(f"\n[OR-Tools] Resolviendo Job Shop con {len(ordenes)} ordenes y {N_MACHINES} maquinas...")
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        makespan_val = solver.Value(makespan)
        print(f"[OR-Tools] {'oPTIMO' if status == cp_model.OPTIMAL else 'FACTIBLE'} "
              f"- Makespan: {makespan_val} u.t. - Tiempo solver: {elapsed:.2f}s")

        # Extraer secuencia de inicio por orden (para simular con SimPy)
        inicios = [(j, solver.Value(task_vars[j, 0][0])) for j in range(len(ordenes))]
        secuencia = [j for j, _ in sorted(inicios, key=lambda x: x[1])]
        return secuencia, makespan_val
    else:
        print("[OR-Tools] No se encontro solucion factible.")
        return list(range(len(ordenes))), None


# ──────────────────────────────────────────────
# Comparativa: secuencia FIFO vs OR-Tools
# ──────────────────────────────────────────────
def ejecutar_fifo(ordenes):
    """Baseline: ordenes en orden de llegada (FIFO)."""
    secuencia = list(range(len(ordenes)))
    planta = PlantaSimPy(ordenes, secuencia, SIM_TIME)
    return planta.ejecutar()


def main():
    print("=" * 60)
    print("  ESCENARIO 1: OR-Tools CP-SAT + TOC")
    print("=" * 60)

    ordenes = generar_instancia()

    # TOC Paso 1: identificar restriccion
    cuello = identificar_cuello(ordenes)

    # Baseline FIFO
    resultado_fifo = ejecutar_fifo(ordenes)
    imprimir_metricas(resultado_fifo, "FIFO (sin optimizacion)")

    # OR-Tools
    secuencia_opt, makespan = resolver_jobshop_cpsat(ordenes, tiempo_limite=30.0)
    planta_opt = PlantaSimPy(ordenes, secuencia_opt, SIM_TIME)
    resultado_opt = planta_opt.ejecutar()
    imprimir_metricas(resultado_opt, "OR-Tools CP-SAT (optimizado)")

    # Mejora
    mejora_t = resultado_opt["throughput"] - resultado_fifo["throughput"]
    mejora_bn = resultado_opt["beneficio_neto"] - resultado_fifo["beneficio_neto"]
    print(f"\n  Mejora en Throughput  : {mejora_t:+.2f} EUR")
    print(f"  Mejora en Bo Neto     : {mejora_bn:+.2f} EUR")
    print(f"  Reduccion WIP         : {resultado_fifo['inventario_wip'] - resultado_opt['inventario_wip']} ordenes\n")


if __name__ == "__main__":
    main()

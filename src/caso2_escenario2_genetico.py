"""
CASO 2 - ESCENARIO 2: NSGA-II Multi-Producto con Coste de Oportunidad
======================================================================
Tres objetivos TOC:
  1. Maximizar Throughput neto (T penalizado si no se cumplen minimos)
  2. Minimizar WIP
  3. Minimizar Coste de Oportunidad (nueva metrica)

El cromosoma codifica tanto la SECUENCIA de ordenes como el MIX de productos,
permitiendo al GA explorar si conviene cambiar el mix para mejorar el T/CU.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from planta_caso2 import (
    generar_ordenes_c2, PlantaC2, imprimir_metricas_c2,
    coste_oportunidad_c2, calcular_tcu_c2,
    MIX_ORDENES, SIM_TIME_C2, OE_FIJO_C2, MINIMOS, PRODUCTOS, N_MACHINES_C2
)
from analisis_oportunidad import imprimir_coste_oportunidad, comparar_co_escenarios

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Tres objetivos: T_neto (+), WIP (-), CO (-)
creator.create("FitnessC2", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("IndividuoC2", list, fitness=creator.FitnessC2)

toolbox2 = base.Toolbox()
_ORDENES_C2 = None   # se inicializa en main


def crear_ind_c2(n: int):
    perm = list(range(n))
    random.shuffle(perm)
    return creator.IndividuoC2(perm)


def evaluar_c2(individuo, ordenes):
    planta = PlantaC2(ordenes, list(individuo), SIM_TIME_C2)
    res = planta.ejecutar()
    co = coste_oportunidad_c2(res)
    return (res["throughput_neto"], res["inventario_wip"], co["CO_total"])


def cruce_ox_c2(ind1, ind2):
    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))
    h1, h2 = [None]*size, [None]*size
    h1[a:b], h2[a:b] = ind1[a:b], ind2[a:b]
    for h, p in [(h1, ind2), (h2, ind1)]:
        rest = [x for x in p if x not in h]
        pos = 0
        for i in range(size):
            if h[i] is None:
                h[i] = rest[pos]; pos += 1
    ind1[:], ind2[:] = h1, h2
    return ind1, ind2


def mutar_c2(ind, indpb=0.15):
    for i in range(len(ind)):
        if random.random() < indpb:
            j = random.randint(0, len(ind) - 1)
            ind[i], ind[j] = ind[j], ind[i]
    return (ind,)


def ejecutar_nsga2_c2(ordenes, n_gen=80, tam_pop=60):
    n = len(ordenes)
    toolbox2.register("individuo", crear_ind_c2, n=n)
    toolbox2.register("population", tools.initRepeat, list, toolbox2.individuo)
    toolbox2.register("evaluate", evaluar_c2, ordenes=ordenes)
    toolbox2.register("mate", cruce_ox_c2)
    toolbox2.register("mutate", mutar_c2, indpb=0.15)
    toolbox2.register("select", tools.selNSGA2)

    pop = toolbox2.population(n=tam_pop)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("T_max",  lambda v: max(x[0] for x in v))
    stats.register("CO_min", lambda v: min(x[2] for x in v))

    print(f"\n[GA C2] NSGA-II: {n_gen} generaciones x {tam_pop} individuos (3 objetivos)")
    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox2, mu=tam_pop, lambda_=tam_pop,
        cxpb=0.8, mutpb=0.2, ngen=n_gen,
        stats=stats, halloffame=hof, verbose=True
    )
    return pop, log, hof


def seleccionar_mejor_c2(hof, ordenes):
    """Mejor segun: max(T_neto - OE - WIP*5 - CO*0.3)."""
    mejor, mejor_score = None, -float("inf")
    for ind in hof:
        t, wip, co = ind.fitness.values
        score = t - OE_FIJO_C2 - wip * 5 - co * 0.3
        if score > mejor_score:
            mejor_score = score
            mejor = ind
    planta = PlantaC2(ordenes, list(mejor), SIM_TIME_C2)
    return planta.ejecutar()


def graficar_pareto_c2(hof):
    t_vals  = [ind.fitness.values[0] for ind in hof]
    co_vals = [ind.fitness.values[2] for ind in hof]
    wip_vals = [ind.fitness.values[1] for ind in hof]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(t_vals, co_vals, c="steelblue", edgecolors="black", s=80)
    axes[0].set_xlabel("Throughput neto (EUR)"); axes[0].set_ylabel("Coste Oportunidad (EUR)")
    axes[0].set_title("Pareto: T vs CO"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(t_vals, wip_vals, c="tomato", edgecolors="black", s=80)
    axes[1].set_xlabel("Throughput neto (EUR)"); axes[1].set_ylabel("WIP")
    axes[1].set_title("Pareto: T vs WIP"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("caso2_pareto_ga.png", dpi=150)
    plt.show()
    print("[GA C2] Grafico guardado: caso2_pareto_ga.png")


def main():
    print("=" * 60)
    print("  CASO 2 - ESCENARIO 2: NSGA-II + CO Multi-Producto")
    print("=" * 60)

    ordenes = generar_ordenes_c2()

    fifo = PlantaC2(ordenes, list(range(len(ordenes))), SIM_TIME_C2)
    res_fifo = fifo.ejecutar()
    imprimir_metricas_c2(res_fifo, "FIFO (baseline)")
    co_fifo = coste_oportunidad_c2(res_fifo)
    imprimir_coste_oportunidad(co_fifo, "Coste de Oportunidad - FIFO")

    pop, log, hof = ejecutar_nsga2_c2(ordenes, n_gen=80, tam_pop=60)
    print(f"\n[GA C2] Frente de Pareto: {len(hof)} soluciones")

    res_ga = seleccionar_mejor_c2(hof, ordenes)
    imprimir_metricas_c2(res_ga, "GA NSGA-II - mejor segun TOC+CO")
    co_ga = coste_oportunidad_c2(res_ga)
    imprimir_coste_oportunidad(co_ga, "Coste de Oportunidad - GA")

    comparar_co_escenarios([co_fifo, co_ga], ["FIFO", "NSGA-II"])

    print(f"\n  Mejora Throughput : {res_ga['throughput'] - res_fifo['throughput']:+.2f} EUR")
    print(f"  Reduccion CO      : {co_fifo['CO_total'] - co_ga['CO_total']:+.2f} EUR")
    print(f"  Producto prioritario (T/CU): {co_ga['mejor_producto_tcu']}"
          f" - Ranking: {co_ga['ranking_tcu']}\n")

    graficar_pareto_c2(hof)


if __name__ == "__main__":
    main()

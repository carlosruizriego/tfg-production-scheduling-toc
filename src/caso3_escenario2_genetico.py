"""
CASO 3 - ESCENARIO 2: NSGA-II Flow Shop + Mix + Coste de Oportunidad
=====================================================================
El GA optimiza simultaneamente:
  1. Throughput (maximizar)
  2. WIP (minimizar)
  3. Coste de Oportunidad (minimizar)

Y explora el MIX optimo de X/Y: el cromosoma codifica tanto
la secuencia como el tipo de producto de cada slot, permitiendo
al GA descubrir el ratio X/Y que maximiza el T/CU en el cuello.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from planta_caso3 import (
    generar_ordenes_c3, PlantaC3, imprimir_metricas_c3,
    coste_oportunidad_c3, SIM_TIME_C3, OE_FIJO_C3,
    N_MACHINES_C3, MIX_DEFAULT, PRECIO, MATERIAL, TIEMPOS_BASE
)
from analisis_oportunidad import imprimir_coste_oportunidad, comparar_co_escenarios

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
N_ORDENES_C3 = len(MIX_DEFAULT)

creator.create("FitnessC3", base.Fitness, weights=(1.0, -1.0, -0.5))
creator.create("IndividuoC3", list, fitness=creator.FitnessC3)

toolbox3 = base.Toolbox()


def crear_ind_c3():
    # Cromosoma: permutacion con mix variable (ratio X/Y aleatorio)
    n_x = random.randint(2, N_ORDENES_C3 - 2)
    mix = ['X'] * n_x + ['Y'] * (N_ORDENES_C3 - n_x)
    random.shuffle(mix)
    return creator.IndividuoC3(mix)


def evaluar_c3(individuo):
    mix = list(individuo)
    ordenes = generar_ordenes_c3(mix=mix, seed=RANDOM_SEED)
    seq = list(range(len(ordenes)))
    planta = PlantaC3(ordenes, seq, SIM_TIME_C3)
    res = planta.ejecutar()
    co = coste_oportunidad_c3(res, ordenes)
    return (res["throughput"], res["inventario_wip"], co["CO_total"])


def mutar_tipo(ind, indpb=0.1):
    """Muta el tipo de producto (X<->Y) en posiciones aleatorias."""
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] = 'Y' if ind[i] == 'X' else 'X'
    return (ind,)


def cruce_uniforme_c3(ind1, ind2):
    """Cruce uniforme para cromosomas de tipos de producto."""
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def ejecutar_nsga2_c3(n_gen=80, tam_pop=60):
    toolbox3.register("individuo", crear_ind_c3)
    toolbox3.register("population", tools.initRepeat, list, toolbox3.individuo)
    toolbox3.register("evaluate", evaluar_c3)
    toolbox3.register("mate", cruce_uniforme_c3)
    toolbox3.register("mutate", mutar_tipo, indpb=0.1)
    toolbox3.register("select", tools.selNSGA2)

    pop = toolbox3.population(n=tam_pop)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("T_max",  lambda v: max(x[0] for x in v))
    stats.register("CO_min", lambda v: min(x[2] for x in v))

    print(f"\n[GA C3] NSGA-II Flow Shop: {n_gen} gen x {tam_pop} ind (mix variable)")
    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox3, mu=tam_pop, lambda_=tam_pop,
        cxpb=0.7, mutpb=0.3, ngen=n_gen,
        stats=stats, halloffame=hof, verbose=True
    )
    return pop, log, hof


def seleccionar_mejor_c3(hof):
    mejor, mejor_score = None, -float("inf")
    for ind in hof:
        t, wip, co = ind.fitness.values
        score = t - OE_FIJO_C3 - wip * 5 - co * 0.3
        if score > mejor_score:
            mejor_score = score; mejor = ind
    mix = list(mejor)
    ordenes = generar_ordenes_c3(mix=mix, seed=RANDOM_SEED)
    planta = PlantaC3(ordenes, list(range(len(ordenes))), SIM_TIME_C3)
    return planta.ejecutar(), ordenes, mix


def graficar_pareto_c3(hof):
    t_vals  = [ind.fitness.values[0] for ind in hof]
    co_vals = [ind.fitness.values[2] for ind in hof]
    nx_vals = [ind.count('X') / len(ind) for ind in hof]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(t_vals, co_vals, c=nx_vals, cmap="RdYlGn",
                          edgecolors="black", s=80)
    plt.colorbar(sc, ax=axes[0], label="Ratio X/(X+Y)")
    axes[0].set_xlabel("Throughput (EUR)"); axes[0].set_ylabel("Coste Oportunidad (EUR)")
    axes[0].set_title("Pareto C3: T vs CO (color=ratio X)"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(nx_vals, t_vals, c="steelblue", edgecolors="black", s=80)
    axes[1].set_xlabel("Ratio X en el mix"); axes[1].set_ylabel("Throughput (EUR)")
    axes[1].set_title("Throughput vs Mix X/Y"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("caso3_pareto_ga.png", dpi=150); plt.show()
    print("[GA C3] Grafico guardado: caso3_pareto_ga.png")


def main():
    print("=" * 60)
    print("  CASO 3 - ESCENARIO 2: NSGA-II Flow Shop + Mix + CO")
    print("=" * 60)

    ordenes_base = generar_ordenes_c3()
    fifo = PlantaC3(ordenes_base, list(range(len(ordenes_base))), SIM_TIME_C3)
    res_fifo = fifo.ejecutar()
    imprimir_metricas_c3(res_fifo, "FIFO mix fijo (baseline)")
    co_fifo = coste_oportunidad_c3(res_fifo, ordenes_base)
    imprimir_coste_oportunidad(co_fifo, "CO - FIFO")

    pop, log, hof = ejecutar_nsga2_c3(n_gen=80, tam_pop=60)
    print(f"\n[GA C3] Frente de Pareto: {len(hof)} soluciones")

    res_ga, ordenes_ga, mix_ga = seleccionar_mejor_c3(hof)
    imprimir_metricas_c3(res_ga, "GA NSGA-II - mejor mix y secuencia")
    co_ga = coste_oportunidad_c3(res_ga, ordenes_ga)
    imprimir_coste_oportunidad(co_ga, "CO - GA")

    comparar_co_escenarios([co_fifo, co_ga], ["FIFO", "NSGA-II"])

    n_x = mix_ga.count('X'); n_y = mix_ga.count('Y')
    print(f"\n  Mix optimo GA: {n_x}X + {n_y}Y (ratio X={n_x/len(mix_ga):.0%})")
    print(f"  Mix recomendado T/CU: {co_ga['mix_recomendado']}")
    print(f"  Mejora Throughput : {res_ga['throughput']-res_fifo['throughput']:+.2f} EUR")
    print(f"  Reduccion CO      : {co_fifo['CO_total']-co_ga['CO_total']:+.2f} EUR\n")

    graficar_pareto_c3(hof)


if __name__ == "__main__":
    main()

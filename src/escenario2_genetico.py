"""
ESCENARIO 2 - Algoritmo Genetico con DEAP (multi-objetivo)
===========================================================
TOC aplicada:
  Las 3 metricas TOC (T, I, OE) se convierten en objetivos del GA:
    - Maximizar Throughput (T)
    - Minimizar Inventario en proceso (I / WIP)
    - OE es fijo → se usa como referencia para calcular Bo Neto

  El cromosoma es una permutacion de ordenes (secuencia de scheduling).
  El algoritmo evolutivo busca el frente de Pareto T vs WIP.

Ventaja sobre Escenario 1: maneja facilmente restricciones complejas
y multiples objetivos sin formulacion matematica explicita.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from planta_simulada import (
    generar_instancia, PlantaSimPy, imprimir_metricas,
    M_JOBS, SIM_TIME, OE_FIJO
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# Configuracion DEAP
# ──────────────────────────────────────────────

# Dos objetivos: maximizar T (+1.0) y minimizar WIP (-1.0)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individuo", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()


def crear_individuo(n: int) -> list:
    perm = list(range(n))
    random.shuffle(perm)
    return creator.Individuo(perm)


def evaluar(individuo, ordenes):
    """
    Funcion de fitness: simula la planta con la secuencia dada
    y devuelve (Throughput, -WIP) como tupla para minimizacion/maximizacion.
    """
    planta = PlantaSimPy(ordenes, list(individuo), SIM_TIME)
    res = planta.ejecutar()
    return (res["throughput"], res["inventario_wip"])   # DEAP aplica los pesos internamente


def cruce_ox(ind1, ind2):
    """Order Crossover (OX) - preserva el orden relativo de las ordenes."""
    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))

    hijo1 = [None] * size
    hijo2 = [None] * size

    hijo1[a:b] = ind1[a:b]
    hijo2[a:b] = ind2[a:b]

    def rellenar(hijo, padre):
        restantes = [x for x in padre if x not in hijo]
        pos = 0
        for i in range(size):
            if hijo[i] is None:
                hijo[i] = restantes[pos]
                pos += 1

    rellenar(hijo1, ind2)
    rellenar(hijo2, ind1)

    ind1[:] = hijo1
    ind2[:] = hijo2
    return ind1, ind2


def mutar_swap(individuo, indpb: float = 0.2):
    """Intercambia aleatoriamente pares de posiciones."""
    for i in range(len(individuo)):
        if random.random() < indpb:
            j = random.randint(0, len(individuo) - 1)
            individuo[i], individuo[j] = individuo[j], individuo[i]
    return (individuo,)


# ──────────────────────────────────────────────
# Registro de operadores
# ──────────────────────────────────────────────
def configurar_toolbox(ordenes):
    n = len(ordenes)
    toolbox.register("individuo", crear_individuo, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individuo)
    toolbox.register("evaluate", evaluar, ordenes=ordenes)
    toolbox.register("mate", cruce_ox)
    toolbox.register("mutate", mutar_swap, indpb=0.15)
    toolbox.register("select", tools.selNSGA2)   # NSGA-II para Pareto


# ──────────────────────────────────────────────
# Algoritmo principal NSGA-II
# ──────────────────────────────────────────────
def ejecutar_nsga2(ordenes,
                   n_generaciones: int = 80,
                   tam_poblacion: int = 60,
                   cx_prob: float = 0.8,
                   mut_prob: float = 0.2):

    configurar_toolbox(ordenes)

    pop = toolbox.population(n=tam_poblacion)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("T_max",  lambda vals: max(v[0] for v in vals))
    stats.register("WIP_min", lambda vals: min(v[1] for v in vals))

    print(f"\n[GA] Ejecutando NSGA-II: {n_generaciones} generaciones x {tam_poblacion} individuos")

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=tam_poblacion,
        lambda_=tam_poblacion,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=n_generaciones,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    return pop, log, hof


# ──────────────────────────────────────────────
# Seleccion de mejor individuo TOC
# ──────────────────────────────────────────────
def seleccionar_mejor_toc(hof, ordenes) -> dict:
    """
    Del frente de Pareto, elige el individuo que maximiza
    el Beneficio Neto TOC = T - OE (con WIP como desempate).
    """
    mejor = None
    mejor_bn = -float("inf")

    for ind in hof:
        t, wip = ind.fitness.values
        bn = t - OE_FIJO - wip * 5   # penalizacion leve por WIP
        if bn > mejor_bn:
            mejor_bn = bn
            mejor = ind

    planta = PlantaSimPy(ordenes, list(mejor), SIM_TIME)
    return planta.ejecutar()


# ──────────────────────────────────────────────
# Visualizacion frente de Pareto
# ──────────────────────────────────────────────
def graficar_pareto(hof, titulo: str = "Frente de Pareto: Throughput vs WIP"):
    puntos = [ind.fitness.values for ind in hof]
    t_vals  = [p[0] for p in puntos]
    wip_vals = [p[1] for p in puntos]

    plt.figure(figsize=(8, 5))
    plt.scatter(t_vals, wip_vals, c="steelblue", edgecolors="black", s=80, zorder=3)
    plt.xlabel("Throughput T (EUR)", fontsize=12)
    plt.ylabel("WIP - Inventario en proceso (ordenes)", fontsize=12)
    plt.title(titulo, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_ga.png", dpi=150)
    plt.show()
    print("[GA] Grafico guardado en pareto_ga.png")


# ──────────────────────────────────────────────
# Comparativa con FIFO
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ESCENARIO 2: Algoritmo Genetico NSGA-II + TOC")
    print("=" * 60)

    ordenes = generar_instancia()

    # Baseline FIFO
    from planta_simulada import PlantaSimPy
    fifo = PlantaSimPy(ordenes, list(range(len(ordenes))), SIM_TIME)
    resultado_fifo = fifo.ejecutar()
    imprimir_metricas(resultado_fifo, "FIFO (sin optimizacion)")

    # NSGA-II
    pop, log, hof = ejecutar_nsga2(ordenes, n_generaciones=80, tam_poblacion=60)

    print(f"\n[GA] Frente de Pareto: {len(hof)} soluciones no dominadas")

    # Mejor solucion segun criterio TOC
    resultado_ga = seleccionar_mejor_toc(hof, ordenes)
    imprimir_metricas(resultado_ga, "GA NSGA-II - mejor segun TOC (T-OE)")

    # Mejoras
    mejora_t  = resultado_ga["throughput"] - resultado_fifo["throughput"]
    mejora_bn = resultado_ga["beneficio_neto"] - resultado_fifo["beneficio_neto"]
    print(f"\n  Mejora en Throughput : {mejora_t:+.2f} EUR")
    print(f"  Mejora en Bo Neto    : {mejora_bn:+.2f} EUR")
    print(f"  Reduccion WIP        : {resultado_fifo['inventario_wip'] - resultado_ga['inventario_wip']} ordenes\n")

    # Frente de Pareto
    graficar_pareto(hof)


if __name__ == "__main__":
    main()

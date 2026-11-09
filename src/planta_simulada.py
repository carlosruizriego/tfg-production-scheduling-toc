"""
Entorno de simulacion comun para los 3 escenarios del TFG.
Planta tipo Job Shop con metricas TOC: Throughput (T), Inventory (I), Operating Expense (OE).

Estructura:
- N_MACHINES maquinas
- M_JOBS ordenes de produccion, cada una con una secuencia de operaciones
- La maquina con mayor utilizacion acumulada es el cuello de botella (restriccion TOC)
"""

import simpy
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ──────────────────────────────────────────────
# Parametros de la planta
# ──────────────────────────────────────────────
N_MACHINES = 5
M_JOBS = 10
PRECIO_VENTA = 100      # EUR por unidad vendida  → contribuye a T
COSTE_MATERIAL = 30     # EUR por unidad          → se resta de T
OE_FIJO = 500           # EUR gastos operativos fijos por simulacion
SIM_TIME = 200          # unidades de tiempo de la simulacion
RANDOM_SEED = 42


@dataclass
class Operacion:
    maquina: int
    duracion: float


@dataclass
class Orden:
    id: int
    operaciones: List[Operacion]
    valor_venta: float = PRECIO_VENTA
    coste_material: float = COSTE_MATERIAL
    tiempo_inicio: float = 0.0
    tiempo_fin: float = 0.0
    completada: bool = False


def generar_instancia(n_jobs: int = M_JOBS,
                      n_maquinas: int = N_MACHINES,
                      seed: int = RANDOM_SEED) -> List[Orden]:
    """Genera una instancia Job Shop reproducible."""
    rng = random.Random(seed)
    ordenes = []
    for j in range(n_jobs):
        # Cada orden visita todas las maquinas en orden aleatorio
        maquinas = list(range(n_maquinas))
        rng.shuffle(maquinas)
        ops = [Operacion(m, rng.uniform(5, 25)) for m in maquinas]
        ordenes.append(Orden(id=j, operaciones=ops))
    return ordenes


# ──────────────────────────────────────────────
# Simulador SimPy
# ──────────────────────────────────────────────
class PlantaSimPy:
    """
    Simula la planta Job Shop con SimPy.
    Recibe una secuencia de ordenes (scheduling) y devuelve metricas TOC.
    """

    def __init__(self, ordenes: List[Orden], secuencia: List[int] = None, sim_time: float = SIM_TIME):
        self.ordenes_base = ordenes
        self.secuencia = secuencia if secuencia else list(range(len(ordenes)))
        self.sim_time = sim_time

        # Metricas
        self.throughput = 0.0
        self.inventario_wip = 0.0
        self.tiempos_maquina: Dict[int, float] = {m: 0.0 for m in range(N_MACHINES)}
        self.ordenes_completadas: List[Orden] = []

    def _proceso_orden(self, env: simpy.Environment,
                       orden: Orden,
                       recursos: Dict[int, simpy.Resource]):
        orden.tiempo_inicio = env.now
        for op in orden.operaciones:
            with recursos[op.maquina].request() as req:
                yield req
                self.tiempos_maquina[op.maquina] += op.duracion
                yield env.timeout(op.duracion)
        orden.tiempo_fin = env.now
        orden.completada = True
        self.ordenes_completadas.append(orden)

    def ejecutar(self) -> Dict:
        import copy
        ordenes = copy.deepcopy(self.ordenes_base)
        env = simpy.Environment()
        recursos = {m: simpy.Resource(env, capacity=1) for m in range(N_MACHINES)}

        for idx in self.secuencia:
            env.process(self._proceso_orden(env, ordenes[idx], recursos))

        env.run(until=self.sim_time)

        completadas = [o for o in ordenes if o.completada]
        self.throughput = sum(o.valor_venta - o.coste_material for o in completadas)
        self.inventario_wip = len([o for o in ordenes if not o.completada])

        # Identificar cuello de botella (TOC paso 1)
        total_tiempo = sum(self.tiempos_maquina.values()) or 1
        utilizaciones = {m: t / self.sim_time for m, t in self.tiempos_maquina.items()}
        cuello = max(utilizaciones, key=utilizaciones.get)

        return {
            "throughput": self.throughput,
            "inventario_wip": self.inventario_wip,
            "OE": OE_FIJO,
            "beneficio_neto": self.throughput - OE_FIJO,
            "ordenes_completadas": len(completadas),
            "utilizaciones": utilizaciones,
            "cuello_de_botella": cuello,
            "utilizacion_cuello": utilizaciones[cuello],
        }


# ──────────────────────────────────────────────
# Utilidad: imprimir resultados TOC
# ──────────────────────────────────────────────
def imprimir_metricas(resultado: Dict, titulo: str = "Resultados TOC"):
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    print(f"  Throughput (T)       : {resultado['throughput']:.2f} EUR")
    print(f"  Gastos operativos(OE): {resultado['OE']:.2f} EUR")
    print(f"  Beneficio neto (T-OE): {resultado['beneficio_neto']:.2f} EUR")
    print(f"  WIP (ordenes paradas): {resultado['inventario_wip']}")
    print(f"  Ordenes completadas  : {resultado['ordenes_completadas']}")
    print(f"  Cuello de botella    : Maquina {resultado['cuello_de_botella']}"
          f" ({resultado['utilizacion_cuello']:.1%} utilizacion)")
    print(f"  Utilizaciones        :", {f"M{m}": f"{u:.1%}" for m, u in resultado['utilizaciones'].items()})
    print(sep)

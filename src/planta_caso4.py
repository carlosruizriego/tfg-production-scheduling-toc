"""
CASO 4: Linea Flow Shop con TIEMPOS DE CAMBIO (setups) dependientes de la secuencia
===================================================================================
Flow Shop puro M0->M1->M2->M3->M4 con tres familias de producto (A, B, C).
La maquina M2 es el CUELLO DE BOTELLA (tiempos de proceso mayores) y, ademas,
requiere un TIEMPO DE CAMBIO (setup) cada vez que cambia la familia de producto
que procesa. El resto de maquinas no necesitan cambio.

Implicacion TOC: cada setup en el cuello roba capacidad a TODO el sistema
("una hora perdida en el cuello es una hora perdida para toda la planta"). Por
tanto, una buena secuencia AGRUPA ordenes de la misma familia para amortizar los
cambios y liberar capacidad del recurso restrictivo. La secuenciacion deja de ser
solo un problema de orden y pasa a ser tambien un problema de agrupacion.

Familias (tiempos por maquina M0..M4; M2 es el cuello):
  A: margen 80   B: margen 65   C: margen 95
"""

import simpy
import random
import copy
from dataclasses import dataclass
from typing import List, Dict

N_MACHINES_C4 = 5
SETUP_M       = 2          # maquina cuello donde ocurren los cambios
SIM_TIME_C4   = 320
OE_FIJO_C4    = 600
RANDOM_SEED   = 42

# Tiempos base por maquina para cada familia (M2 es el mayor -> cuello)
TIEMPOS_BASE = {
    'A': [10, 12, 26, 11,  9],
    'B': [12, 10, 24, 13, 10],
    'C': [ 9, 11, 28, 10, 12],
}

PRECIO   = {'A': 120, 'B': 100, 'C': 140}
MATERIAL = {'A':  40, 'B':  35, 'C':  45}     # margenes: A=80, B=65, C=95

# Matriz de tiempos de cambio en el cuello: SETUP[origen][destino] (mismo => 0)
SETUP = {
    'A': {'A': 0, 'B': 10, 'C': 14},
    'B': {'A': 10, 'B': 0, 'C': 12},
    'C': {'A': 14, 'B': 12, 'C': 0},
}

FAMILIAS = ['A', 'B', 'C']

# Pool por defecto: 5A + 5B + 5C = 15 ordenes candidatas
POOL_DEFAULT = ['A'] * 5 + ['B'] * 5 + ['C'] * 5


@dataclass
class OrdenC4:
    id: int
    tipo: str
    tiempos: List[float]
    precio: float
    material: float
    completada: bool = False
    tiempo_fin: float = 0.0

    @property
    def margen(self):
        return self.precio - self.material


def generar_ordenes_c4(pool: List[str] = None, seed: int = RANDOM_SEED) -> List[OrdenC4]:
    """Genera ordenes con variacion de tiempos (+/-10%); el orden del pool se baraja."""
    rng = random.Random(seed)
    if pool is None:
        pool = POOL_DEFAULT[:]
    pool = pool[:]
    rng.shuffle(pool)
    ordenes = []
    for i, tipo in enumerate(pool):
        tiempos = [t * rng.uniform(0.9, 1.1) for t in TIEMPOS_BASE[tipo]]
        ordenes.append(OrdenC4(
            id=i, tipo=tipo, tiempos=tiempos,
            precio=PRECIO[tipo], material=MATERIAL[tipo]
        ))
    return ordenes


class PlantaC4:
    """
    Simulador SimPy del Flow Shop con setups dependientes de la secuencia en el cuello.
    Recibe la secuencia de despacho y devuelve metricas TOC, incluido el tiempo total
    perdido en cambios sobre el cuello.
    """

    def __init__(self, ordenes: List[OrdenC4], secuencia: List[int] = None,
                 sim_time: float = SIM_TIME_C4):
        self.ordenes_base = ordenes
        self.secuencia = secuencia if secuencia is not None else list(range(len(ordenes)))
        self.sim_time = sim_time
        self.tiempos_maquina: Dict[int, float] = {m: 0.0 for m in range(N_MACHINES_C4)}
        self.setup_total = 0.0
        self._last_fam = None      # ultima familia procesada en el cuello

    def _proceso(self, env, orden: OrdenC4, recursos):
        for m in range(N_MACHINES_C4):
            dur = orden.tiempos[m]
            with recursos[m].request() as req:
                yield req
                # Tiempo de cambio en el cuello si cambia la familia
                if m == SETUP_M:
                    if self._last_fam is not None and self._last_fam != orden.tipo:
                        s = SETUP[self._last_fam][orden.tipo]
                        self.setup_total += s
                        self.tiempos_maquina[m] += s
                        yield env.timeout(s)
                    self._last_fam = orden.tipo
                self.tiempos_maquina[m] += dur
                yield env.timeout(dur)
        orden.completada = True
        orden.tiempo_fin = env.now

    def ejecutar(self) -> Dict:
        ordenes = copy.deepcopy(self.ordenes_base)
        env = simpy.Environment()
        recursos = {m: simpy.Resource(env, capacity=1) for m in range(N_MACHINES_C4)}
        self._last_fam = None
        for idx in self.secuencia:
            env.process(self._proceso(env, ordenes[idx], recursos))
        env.run(until=self.sim_time)

        completadas = [o for o in ordenes if o.completada]
        throughput = sum(o.margen for o in completadas)
        wip = len([o for o in ordenes if not o.completada])
        conteo = {t: sum(1 for o in completadas if o.tipo == t) for t in FAMILIAS}

        utilizaciones = {m: t / self.sim_time for m, t in self.tiempos_maquina.items()}
        cuello = max(utilizaciones, key=utilizaciones.get)

        return {
            "throughput": throughput,
            "inventario_wip": wip,
            "OE": OE_FIJO_C4,
            "beneficio_neto": throughput - OE_FIJO_C4,
            "ordenes_completadas": len(completadas),
            "conteo_por_tipo": conteo,
            "setup_total": self.setup_total,
            "utilizaciones": utilizaciones,
            "tiempos_maquina": dict(self.tiempos_maquina),
            "cuello_de_botella": cuello,
            "utilizacion_cuello": utilizaciones[cuello],
        }


def imprimir_metricas_c4(resultado: Dict, titulo: str = "Resultados Caso 4"):
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    print(f"  Throughput (T)       : {resultado['throughput']:.2f} EUR")
    print(f"  Completadas por tipo : {resultado['conteo_por_tipo']}")
    print(f"  Tiempo en cambios    : {resultado['setup_total']:.1f} u.t. (cuello M{SETUP_M})")
    print(f"  Cuello de botella    : M{resultado['cuello_de_botella']}"
          f" ({resultado['utilizacion_cuello']:.1%})")
    print(sep)

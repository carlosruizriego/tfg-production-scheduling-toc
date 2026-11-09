"""
CASO 3: Linea de Fabricacion Flow Shop - 2 Productos
=====================================================
Flow Shop puro: todas las ordenes siguen la secuencia M0->M1->M2->M3->M4.
Los dos productos difieren en tiempos de procesado y precio de venta.

Producto X: precio=130, material=35 (margen=95), tiempos mayores (proceso complejo)
Producto Y: precio= 90, material=25 (margen=65), tiempos menores (proceso simple)

La secuenciacion optima depende de:
  1. El mix X/Y (cuantas unidades de cada tipo)
  2. El orden de procesado (regla de Johnson extendida para 2 productos)

El cuello de botella migra segun el mix: mas X -> cuello en M1; mas Y -> cuello en M3.
"""

import simpy
import random
import copy
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

N_MACHINES_C3 = 5
SIM_TIME_C3   = 250
OE_FIJO_C3    = 550
RANDOM_SEED   = 42

# Tiempos base por maquina para cada producto (Flow Shop: secuencia fija)
TIEMPOS_BASE = {
    'X': [22, 28, 15, 20, 14],   # Proceso complejo: M1 y M3 son los mas lentos
    'Y': [12, 18, 24, 10, 20],   # Proceso simple: M2 y M4 son los mas lentos
}

PRECIO  = {'X': 130, 'Y': 90}
MATERIAL = {'X': 35,  'Y': 25}

# Mix por defecto: 5X + 6Y = 11 ordenes
MIX_DEFAULT = ['X'] * 5 + ['Y'] * 6


@dataclass
class OrdenC3:
    id: int
    tipo: str
    tiempos: List[float]       # tiempo en cada maquina [M0..M4]
    precio: float
    material: float
    completada: bool = False
    tiempo_fin: float = 0.0

    @property
    def margen(self):
        return self.precio - self.material


def generar_ordenes_c3(mix: List[str] = None, seed: int = RANDOM_SEED) -> List[OrdenC3]:
    """Genera ordenes con variacion de tiempos (+/-10%)."""
    rng = random.Random(seed)
    if mix is None:
        mix = MIX_DEFAULT[:]
        rng.shuffle(mix)
    ordenes = []
    for i, tipo in enumerate(mix):
        tiempos = [t * rng.uniform(0.9, 1.1) for t in TIEMPOS_BASE[tipo]]
        ordenes.append(OrdenC3(
            id=i, tipo=tipo, tiempos=tiempos,
            precio=PRECIO[tipo], material=MATERIAL[tipo]
        ))
    return ordenes


class PlantaC3:
    """Simulador SimPy para Flow Shop de 2 productos."""

    def __init__(self, ordenes: List[OrdenC3], secuencia: List[int] = None,
                 sim_time: float = SIM_TIME_C3):
        self.ordenes_base = ordenes
        self.secuencia = secuencia if secuencia is not None else list(range(len(ordenes)))
        self.sim_time = sim_time
        self.tiempos_maquina: Dict[int, float] = {m: 0.0 for m in range(N_MACHINES_C3)}

    def _proceso(self, env, orden: OrdenC3, recursos):
        for m in range(N_MACHINES_C3):
            dur = orden.tiempos[m]
            with recursos[m].request() as req:
                yield req
                self.tiempos_maquina[m] += dur
                yield env.timeout(dur)
        orden.completada = True
        orden.tiempo_fin = env.now

    def ejecutar(self) -> Dict:
        ordenes = copy.deepcopy(self.ordenes_base)
        env = simpy.Environment()
        recursos = {m: simpy.Resource(env, capacity=1) for m in range(N_MACHINES_C3)}
        for idx in self.secuencia:
            env.process(self._proceso(env, ordenes[idx], recursos))
        env.run(until=self.sim_time)

        completadas = [o for o in ordenes if o.completada]
        throughput = sum(o.margen for o in completadas)
        wip = len([o for o in ordenes if not o.completada])
        conteo = {t: sum(1 for o in completadas if o.tipo == t) for t in ['X', 'Y']}

        utilizaciones = {m: t / self.sim_time for m, t in self.tiempos_maquina.items()}
        cuello = max(utilizaciones, key=utilizaciones.get)

        return {
            "throughput": throughput,
            "inventario_wip": wip,
            "OE": OE_FIJO_C3,
            "beneficio_neto": throughput - OE_FIJO_C3,
            "ordenes_completadas": len(completadas),
            "conteo_por_tipo": conteo,
            "utilizaciones": utilizaciones,
            "tiempos_maquina": dict(self.tiempos_maquina),
            "cuello_de_botella": cuello,
            "utilizacion_cuello": utilizaciones[cuello],
        }


def calcular_tcu_c3(resultado: Dict, ordenes: List[OrdenC3]) -> Dict:
    """
    Calcula T/CU para X e Y en el cuello actual.
    Usa el tiempo medio real observado en el cuello para cada tipo.
    """
    cuello = resultado["cuello_de_botella"]
    tiempos_por_tipo = {'X': [], 'Y': []}
    for o in ordenes:
        tiempos_por_tipo[o.tipo].append(o.tiempos[cuello])

    tcu = {}
    for tipo in ['X', 'Y']:
        margen = PRECIO[tipo] - MATERIAL[tipo]
        t_med = np.mean(tiempos_por_tipo[tipo]) if tiempos_por_tipo[tipo] else 1
        tcu[tipo] = margen / t_med

    return tcu


def coste_oportunidad_c3(resultado: Dict, ordenes: List[OrdenC3]) -> Dict:
    """Calcula el coste de oportunidad del Caso 3."""
    tcu = calcular_tcu_c3(resultado, ordenes)
    rank = sorted(tcu.items(), key=lambda x: x[1], reverse=True)
    mejor_prod, mejor_tcu = rank[0]

    cuello = resultado["cuello_de_botella"]
    tiempo_cuello = resultado["tiempos_maquina"].get(cuello, 0)
    T_max = mejor_tcu * tiempo_cuello
    T_real = resultado["throughput"]
    CO = max(0.0, T_max - T_real)

    conteo = resultado.get("conteo_por_tipo", {})
    ratio_actual = (conteo.get('X', 0) /
                    max(1, conteo.get('X', 0) + conteo.get('Y', 0)))

    # Ratio optimo teorico: el que maximiza T/CU en el cuello
    # Si T/CU_X > T/CU_Y: optimo es maximizar X (ratio -> 1)
    # Si T/CU_Y > T/CU_X: optimo es maximizar Y (ratio -> 0)
    ratio_optimo = 1.0 if tcu.get('X', 0) >= tcu.get('Y', 0) else 0.0

    return {
        "CO_total": round(CO, 2),
        "T_obtenido": round(T_real, 2),
        "T_max_teorico": round(T_max, 2),
        "eficiencia_cuello": round(T_real / T_max * 100, 1) if T_max > 0 else 0,
        "ranking_tcu": rank,
        "tcu_por_producto": {k: round(v, 3) for k, v in tcu.items()},
        "ratio_X_actual": round(ratio_actual, 2),
        "ratio_X_optimo": ratio_optimo,
        "mix_recomendado": "Maximizar X" if ratio_optimo == 1.0 else "Maximizar Y",
    }


def imprimir_metricas_c3(resultado: Dict, titulo: str = "Resultados Caso 3"):
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    print(f"  Throughput (T)       : {resultado['throughput']:.2f} EUR")
    print(f"  Gastos operativos    : {resultado['OE']:.2f} EUR")
    print(f"  Beneficio neto       : {resultado['beneficio_neto']:.2f} EUR")
    print(f"  WIP                  : {resultado['inventario_wip']}")
    print(f"  Completadas por tipo : {resultado['conteo_por_tipo']}")
    print(f"  Cuello de botella    : M{resultado['cuello_de_botella']}"
          f" ({resultado['utilizacion_cuello']:.1%})")
    print(f"  Utilizaciones        : "
          f"{ {f'M{m}': f'{u:.1%}' for m,u in resultado['utilizaciones'].items()} }")
    print(sep)

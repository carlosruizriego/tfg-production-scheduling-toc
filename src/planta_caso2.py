"""
CASO 2: Planta Multi-Producto con Restricciones de Minimos
==========================================================
3 productos (A, B, C) procesados en 5 maquinas.
Maquinas compartidas: M0, M2, M4 (las 3 las usan todos los productos)
Maquinas exclusivas: M1 (solo A y B), M3 (solo B y C)

Productos y rutas:
  A: M0(15) -> M1(20) -> M2(18) -> M4(12)   precio=150, material=40
  B: M0(20) -> M1(15) -> M2(25) -> M3(18) -> M4(10)  precio=100, material=30
  C: M0(12) -> M2(20) -> M3(22) -> M4(15)   precio=80,  material=25

Restricciones de minimos: A>=2, B>=3, C>=2
Ordenes totales: 4A + 5B + 4C = 13 ordenes
"""

import simpy
import random
import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

N_MACHINES_C2 = 5
SIM_TIME_C2   = 300
RANDOM_SEED   = 42
OE_FIJO_C2    = 600

# Definicion de productos: (maquina, duracion_base)
PRODUCTOS = {
    'A': {'precio': 150, 'material': 40,
          'ruta': [(0, 15), (1, 20), (2, 18), (4, 12)]},
    'B': {'precio': 100, 'material': 30,
          'ruta': [(0, 20), (1, 15), (2, 25), (3, 18), (4, 10)]},
    'C': {'precio':  80, 'material': 25,
          'ruta': [(0, 12), (2, 20), (3, 22), (4, 15)]},
}

MINIMOS = {'A': 2, 'B': 3, 'C': 2}

# Mix de ordenes: 4A, 5B, 4C
MIX_ORDENES = ['A'] * 4 + ['B'] * 5 + ['C'] * 4


@dataclass
class OrdenC2:
    id: int
    tipo: str
    ruta: List[Tuple[int, float]]
    precio: float
    material: float
    completada: bool = False
    tiempo_fin: float = 0.0

    @property
    def margen(self):
        return self.precio - self.material


def generar_ordenes_c2(mix: List[str] = None, seed: int = RANDOM_SEED) -> List[OrdenC2]:
    """Genera lista de ordenes segun el mix de productos."""
    rng = random.Random(seed)
    if mix is None:
        mix = MIX_ORDENES[:]
        rng.shuffle(mix)
    ordenes = []
    for i, tipo in enumerate(mix):
        p = PRODUCTOS[tipo]
        # Pequena variacion de tiempos (+/-10%) para realismo
        ruta_var = [(m, t * rng.uniform(0.9, 1.1)) for m, t in p['ruta']]
        ordenes.append(OrdenC2(
            id=i, tipo=tipo, ruta=ruta_var,
            precio=p['precio'], material=p['material']
        ))
    return ordenes


class PlantaC2:
    """Simulador SimPy para planta multi-producto."""

    def __init__(self, ordenes: List[OrdenC2], secuencia: List[int] = None,
                 sim_time: float = SIM_TIME_C2):
        self.ordenes_base = ordenes
        self.secuencia = secuencia if secuencia is not None else list(range(len(ordenes)))
        self.sim_time = sim_time
        self.tiempos_maquina: Dict[int, float] = {m: 0.0 for m in range(N_MACHINES_C2)}

    def _proceso(self, env, orden: OrdenC2, recursos):
        for maq, dur in orden.ruta:
            with recursos[maq].request() as req:
                yield req
                self.tiempos_maquina[maq] += dur
                yield env.timeout(dur)
        orden.completada = True
        orden.tiempo_fin = env.now

    def ejecutar(self) -> Dict:
        ordenes = copy.deepcopy(self.ordenes_base)
        env = simpy.Environment()
        recursos = {m: simpy.Resource(env, capacity=1) for m in range(N_MACHINES_C2)}
        for idx in self.secuencia:
            env.process(self._proceso(env, ordenes[idx], recursos))
        env.run(until=self.sim_time)

        completadas = [o for o in ordenes if o.completada]
        throughput = sum(o.margen for o in completadas)
        wip = len([o for o in ordenes if not o.completada])

        conteo = {t: sum(1 for o in completadas if o.tipo == t) for t in PRODUCTOS}
        minimos_ok = all(conteo.get(t, 0) >= MINIMOS[t] for t in MINIMOS)
        penalizacion = sum(
            max(0, MINIMOS[t] - conteo.get(t, 0)) * PRODUCTOS[t]['precio'] * 0.5
            for t in MINIMOS
        ) if not minimos_ok else 0

        utilizaciones = {m: t / self.sim_time for m, t in self.tiempos_maquina.items()}
        cuello = max(utilizaciones, key=utilizaciones.get)

        return {
            "throughput": throughput,
            "throughput_neto": throughput - penalizacion,
            "inventario_wip": wip,
            "OE": OE_FIJO_C2,
            "beneficio_neto": throughput - penalizacion - OE_FIJO_C2,
            "ordenes_completadas": len(completadas),
            "conteo_por_tipo": conteo,
            "minimos_cumplidos": minimos_ok,
            "penalizacion_minimos": penalizacion,
            "utilizaciones": utilizaciones,
            "tiempos_maquina": dict(self.tiempos_maquina),
            "cuello_de_botella": cuello,
            "utilizacion_cuello": utilizaciones[cuello],
        }


def calcular_tcu_c2(cuello: int) -> Dict[str, float]:
    """T/CU para cada producto en el cuello de botella."""
    tcu = {}
    for nombre, p in PRODUCTOS.items():
        margen = p['precio'] - p['material']
        t_cuello = next((t for m, t in p['ruta'] if m == cuello), None)
        tcu[nombre] = margen / t_cuello if t_cuello else float('inf')
    return tcu


def coste_oportunidad_c2(resultado: Dict) -> Dict:
    """Calcula el coste de oportunidad del Caso 2."""
    cuello = resultado["cuello_de_botella"]
    tcu = calcular_tcu_c2(cuello)
    rank = sorted(tcu.items(), key=lambda x: x[1], reverse=True)
    mejor_prod, mejor_tcu = rank[0]

    tiempo_cuello = resultado["tiempos_maquina"].get(cuello, 0)
    T_max = mejor_tcu * tiempo_cuello
    T_real = resultado["throughput"]
    CO = max(0.0, T_max - T_real)

    return {
        "CO_total": round(CO, 2),
        "T_obtenido": round(T_real, 2),
        "T_max_teorico": round(T_max, 2),
        "eficiencia_cuello": round(T_real / T_max * 100, 1) if T_max > 0 else 0,
        "ranking_tcu": rank,
        "mejor_producto_tcu": mejor_prod,
        "tcu_por_producto": {k: round(v, 3) for k, v in tcu.items()},
    }


def imprimir_metricas_c2(resultado: Dict, titulo: str = "Resultados Caso 2"):
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    print(f"  Throughput bruto     : {resultado['throughput']:.2f} EUR")
    print(f"  Penalizacion minimos : {resultado['penalizacion_minimos']:.2f} EUR"
          f"  {'[OK]' if resultado['minimos_cumplidos'] else '[INCUMPLIDO]'}")
    print(f"  Throughput neto      : {resultado['throughput_neto']:.2f} EUR")
    print(f"  Gastos operativos    : {resultado['OE']:.2f} EUR")
    print(f"  Beneficio neto       : {resultado['beneficio_neto']:.2f} EUR")
    print(f"  WIP                  : {resultado['inventario_wip']}")
    print(f"  Completadas por tipo : {resultado['conteo_por_tipo']}")
    print(f"  Cuello de botella    : M{resultado['cuello_de_botella']}"
          f" ({resultado['utilizacion_cuello']:.1%})")
    print(f"  Utilizaciones        : "
          f"{ {f'M{m}': f'{u:.1%}' for m,u in resultado['utilizaciones'].items()} }")
    print(sep)

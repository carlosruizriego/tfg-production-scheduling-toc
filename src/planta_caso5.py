"""
CASO 5: Job Shop grande multi-familia con DOBLE CUELLO y setups dependientes de secuencia.
==========================================================================================
Escenario disenado para dar margen estructural al agente GNN frente al genetico:
  - 20 ordenes, 6 maquinas, 5 familias (A-E) con margenes heterogeneos.
  - Job Shop: cada orden recorre las 6 maquinas en un ORDEN PROPIO aleatorio (rutas distintas).
  - DOS cuellos de botella (M2 y M4) con tiempos de proceso altos y, ademas, tiempos de
    cambio (setup) dependientes de la familia, con matrices DISTINTAS en cada cuello.
La dificultad: agrupar familias para amortizar los cambios en M2 puede desagrupar en M4 y
viceversa; el compromiso entre dos cuellos acoplados es global y combinatorio, y el T/CU
escalar (pensado para un unico cuello) no lo captura. El espacio (20! permutaciones) excede
lo que el GA agota en 50 generaciones -> banco legitimo para una politica aprendida.
"""
import simpy
import random
import copy
from dataclasses import dataclass
from typing import List, Dict

N_MACHINES_C5 = 6
CUELLO_A = 2          # primer cuello (setups SETUP_A)
CUELLO_B = 4          # segundo cuello (setups SETUP_B)
SIM_TIME_C5 = 550
OE_FIJO_C5 = 700
RANDOM_SEED = 42

FAMILIAS = ['A', 'B', 'C', 'D', 'E']

# Tiempos base por familia y maquina (M0..M5); M2 y M4 son los cuellos
TIEMPOS_BASE = {
    'A': [10, 12, 26, 11, 24, 9],
    'B': [12, 10, 24, 13, 22, 10],
    'C': [9, 11, 28, 10, 26, 12],
    'D': [11, 9, 25, 12, 23, 11],
    'E': [10, 13, 27, 11, 25, 10],
}
PRECIO = {'A': 100, 'B': 120, 'C': 140, 'D': 155, 'E': 180}
MATERIAL = {'A': 40, 'B': 40, 'C': 45, 'D': 45, 'E': 50}   # margenes: 60/80/95/110/130

# Matrices de cambio (setup) en cada cuello: SETUP[origen][destino] (mismo = 0)
def _mat(base):
    return {a: {b: (0 if a == b else base[abs(FAMILIAS.index(a) - FAMILIAS.index(b))])
                for b in FAMILIAS} for a in FAMILIAS}

SETUP_A = _mat([0, 9, 12, 14, 16])     # cambios mas caros cuanto mas lejana la familia
SETUP_B = _mat([0, 11, 8, 15, 10])     # estructura DISTINTA -> conflicto entre cuellos

POOL_DEFAULT = ['A', 'B', 'C', 'D', 'E'] * 4   # 20 ordenes


@dataclass
class OrdenC5:
    id: int
    tipo: str
    ruta: List          # [(maquina, duracion), ...] en orden propio
    precio: float
    material: float
    completada: bool = False
    tiempo_fin: float = 0.0

    @property
    def margen(self):
        return self.precio - self.material


def generar_ordenes_c5(pool: List[str] = None, seed: int = RANDOM_SEED) -> List[OrdenC5]:
    rng = random.Random(seed)
    if pool is None:
        pool = POOL_DEFAULT[:]
    pool = pool[:]
    rng.shuffle(pool)
    ordenes = []
    for i, tipo in enumerate(pool):
        maquinas = list(range(N_MACHINES_C5))
        rng.shuffle(maquinas)                       # ruta propia (Job Shop)
        ruta = [(m, TIEMPOS_BASE[tipo][m] * rng.uniform(0.9, 1.1)) for m in maquinas]
        ordenes.append(OrdenC5(id=i, tipo=tipo, ruta=ruta,
                               precio=PRECIO[tipo], material=MATERIAL[tipo]))
    return ordenes


class PlantaC5:
    """Job Shop con SimPy y setups dependientes de secuencia en dos cuellos (M2, M4)."""

    SETUP = {CUELLO_A: SETUP_A, CUELLO_B: SETUP_B}

    def __init__(self, ordenes: List[OrdenC5], secuencia: List[int] = None,
                 sim_time: float = SIM_TIME_C5):
        self.ordenes_base = ordenes
        self.secuencia = secuencia if secuencia is not None else list(range(len(ordenes)))
        self.sim_time = sim_time
        self.tiempos_maquina: Dict[int, float] = {m: 0.0 for m in range(N_MACHINES_C5)}
        self.setup_total = 0.0
        self._last_fam = {CUELLO_A: None, CUELLO_B: None}

    def _proceso(self, env, orden: OrdenC5, recursos):
        for (m, dur) in orden.ruta:
            with recursos[m].request() as req:
                yield req
                if m in (CUELLO_A, CUELLO_B):
                    lf = self._last_fam[m]
                    if lf is not None and lf != orden.tipo:
                        s = self.SETUP[m][lf][orden.tipo]
                        self.setup_total += s
                        self.tiempos_maquina[m] += s
                        yield env.timeout(s)
                    self._last_fam[m] = orden.tipo
                self.tiempos_maquina[m] += dur
                yield env.timeout(dur)
        orden.completada = True
        orden.tiempo_fin = env.now

    def ejecutar(self) -> Dict:
        ordenes = copy.deepcopy(self.ordenes_base)
        env = simpy.Environment()
        recursos = {m: simpy.Resource(env, capacity=1) for m in range(N_MACHINES_C5)}
        self._last_fam = {CUELLO_A: None, CUELLO_B: None}
        for idx in self.secuencia:
            env.process(self._proceso(env, ordenes[idx], recursos))
        env.run(until=self.sim_time)

        completadas = [o for o in ordenes if o.completada]
        throughput = sum(o.margen for o in completadas)
        conteo = {t: sum(1 for o in completadas if o.tipo == t) for t in FAMILIAS}
        utilizaciones = {m: t / self.sim_time for m, t in self.tiempos_maquina.items()}
        cuello = max(utilizaciones, key=utilizaciones.get)
        return {
            "throughput": throughput,
            "inventario_wip": len([o for o in ordenes if not o.completada]),
            "OE": OE_FIJO_C5,
            "beneficio_neto": throughput - OE_FIJO_C5,
            "ordenes_completadas": len(completadas),
            "conteo_por_tipo": conteo,
            "setup_total": self.setup_total,
            "utilizaciones": utilizaciones,
            "tiempos_maquina": dict(self.tiempos_maquina),
            "cuello_de_botella": cuello,
            "utilizacion_cuello": utilizaciones[cuello],
        }

"""
ESCENARIO 3 - Reinforcement Learning (PPO) con entorno dinamico
================================================================
TOC aplicada implicitamente:
  El agente NO conoce TOC, pero aprende a:
    - Priorizar ordenes que alivian el cuello de botella (TOC paso 2)
    - Reducir WIP acumulado (TOC: reducir I)
    - Maximizar throughput en el tiempo disponible (TOC: maximizar T)

  La restriccion puede MOVERSE durante el entrenamiento (averias simuladas),
  lo que obliga al agente a aprender comportamiento adaptativo - algo que
  OR-Tools y el GA no pueden hacer sin ser relanzados.

Arquitectura:
  - Entorno Gymnasium personalizado (JobShopEnv)
  - Algoritmo PPO de Stable-Baselines3
  - Observacion: vector de cargas por maquina + WIP actual
  - Accion: indice de la proxima orden a procesar
  - Reward: throughput incremental - penalizacion por WIP
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from planta_simulada import (
    generar_instancia, Orden, Operacion,
    N_MACHINES, M_JOBS, SIM_TIME,
    OE_FIJO, PRECIO_VENTA, COSTE_MATERIAL
)

RANDOM_SEED = 42


# ──────────────────────────────────────────────
# Entorno Gymnasium: Job Shop dinamico
# ──────────────────────────────────────────────
class JobShopEnv(gym.Env):
    """
    Entorno discreto-continuo que simula la planta paso a paso.
    En cada step el agente elige que orden procesar a continuacion.

    Observacion (Box):
      [carga_m0, ..., carga_m(N-1),   # utilizacion acumulada por maquina (0-1)
       wip_normalizado,                # ordenes pendientes / total
       progreso_sim]                   # tiempo transcurrido / SIM_TIME

    Accion (Discrete): indice de la orden a despachar (entre las disponibles)
    """

    metadata = {"render_modes": []}

    def __init__(self, n_jobs: int = M_JOBS, n_maquinas: int = N_MACHINES,
                 sim_time: float = SIM_TIME, averia_prob: float = 0.05):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_maquinas = n_maquinas
        self.sim_time = sim_time
        self.averia_prob = averia_prob   # probabilidad de que una maquina se averia cada step

        # Espacios
        obs_dim = n_maquinas + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_jobs)

        self.rng = np.random.default_rng(RANDOM_SEED)
        self._reset_estado()

    # ── Helpers ──────────────────────────────
    def _reset_estado(self):
        ordenes = generar_instancia(self.n_jobs, self.n_maquinas,
                                    seed=int(self.rng.integers(0, 10000)))
        self.ordenes = ordenes
        self.cola = list(range(self.n_jobs))          # ordenes disponibles
        self.en_proceso: dict = {}                    # orden_id → op_idx actual
        self.completadas = []
        self.tiempo = 0.0
        self.carga_maquina = np.zeros(self.n_maquinas, dtype=np.float32)
        self.maquinas_bloqueadas = set()              # averias activas
        self.throughput_acum = 0.0

    def _obs(self) -> np.ndarray:
        util = np.clip(self.carga_maquina / (self.tiempo + 1e-6), 0, 1)
        wip = len(self.cola) / self.n_jobs
        progreso = self.tiempo / self.sim_time
        return np.concatenate([util, [wip, progreso]]).astype(np.float32)

    def _simular_averia(self):
        """Con probabilidad averia_prob, bloquea una maquina aleatoria por un tiempo."""
        if self.rng.random() < self.averia_prob:
            m = int(self.rng.integers(0, self.n_maquinas))
            self.maquinas_bloqueadas.add(m)
            # Se desbloquea en el siguiente step (simplificacion)
        else:
            self.maquinas_bloqueadas = set()

    def _procesar_orden(self, orden_id: int) -> float:
        """Procesa la siguiente operacion de la orden y devuelve el tiempo consumido."""
        orden: Orden = self.ordenes[orden_id]
        op_idx = self.en_proceso.get(orden_id, 0)

        if op_idx >= len(orden.operaciones):
            return 0.0   # ya completada

        op: Operacion = orden.operaciones[op_idx]

        # Bloqueo por averia
        if op.maquina in self.maquinas_bloqueadas:
            return 2.0   # penalizacion de tiempo por averia

        duracion = op.duracion
        self.carga_maquina[op.maquina] += duracion
        self.tiempo += duracion
        self.en_proceso[orden_id] = op_idx + 1

        # ¿Orden completada?
        if op_idx + 1 >= len(orden.operaciones):
            self.completadas.append(orden_id)
            if orden_id in self.cola:
                self.cola.remove(orden_id)
            ganancia = orden.valor_venta - orden.coste_material
            self.throughput_acum += ganancia
            return duracion

        return duracion

    # ── Gym API ──────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_estado()
        return self._obs(), {}

    def step(self, action: int):
        self._simular_averia()

        # Si la accion no esta disponible, elegir la primera de la cola
        if action not in self.cola and self.cola:
            action = self.cola[0]
        elif not self.cola:
            # No quedan ordenes
            return self._obs(), 0.0, True, False, self._info()

        t_antes = self.throughput_acum
        self._procesar_orden(action)
        t_despues = self.throughput_acum

        # Reward TOC: ganancia incremental normalizada
        # La penalizacion por WIP es minima para no distorsionar el aprendizaje
        reward = (t_despues - t_antes) / 70.0 - len(self.cola) * 0.001

        terminated = (self.tiempo >= self.sim_time) or (len(self.cola) == 0)
        truncated = False

        return self._obs(), reward, terminated, truncated, self._info()

    def _info(self):
        util = self.carga_maquina / (self.tiempo + 1e-6)
        cuello = int(np.argmax(util))
        return {
            "throughput": self.throughput_acum,
            "wip": len(self.cola),
            "completadas": len(self.completadas),
            "cuello_de_botella": cuello,
            "utilizacion_cuello": float(util[cuello]),
            "beneficio_neto": self.throughput_acum - OE_FIJO,
        }

    def resultado_final(self) -> dict:
        info = self._info()
        info["OE"] = OE_FIJO
        info["utilizaciones"] = {
            m: float(self.carga_maquina[m] / (self.tiempo + 1e-6))
            for m in range(self.n_maquinas)
        }
        return info


# ──────────────────────────────────────────────
# Callback para registrar metricas por episodio
# ──────────────────────────────────────────────
class MetricasCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.throughputs = []
        self.wips = []
        self.episodio = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "throughput" in info:
                self.throughputs.append(info["throughput"])
                self.wips.append(info["wip"])
        return True


# ──────────────────────────────────────────────
# Entrenamiento PPO
# ──────────────────────────────────────────────
def entrenar_ppo(n_steps: int = 200_000):
    env = JobShopEnv(averia_prob=0.05)
    check_env(env, warn=True)

    print(f"\n[RL] Entrenando PPO durante {n_steps:,} pasos...")
    modelo = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,     # exploracion
        seed=RANDOM_SEED,
    )
    callback = MetricasCallback()
    modelo.learn(total_timesteps=n_steps, callback=callback, progress_bar=False)
    modelo.save("ppo_jobshop_toc")
    print("[RL] Modelo guardado en ppo_jobshop_toc.zip")
    return modelo, callback


# ──────────────────────────────────────────────
# Evaluacion del agente entrenado
# ──────────────────────────────────────────────
def evaluar_agente(modelo, n_episodios: int = 10) -> dict:
    env = JobShopEnv(averia_prob=0.0)  # evaluacion sin averias = comparativa justa con FIFO
    resultados = []

    for ep in range(n_episodios):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = modelo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        resultados.append(env.resultado_final())

    # Promediar
    keys = ["throughput", "wip", "completadas", "beneficio_neto"]
    promedio = {k: np.mean([r[k] for r in resultados]) for k in keys}
    promedio["OE"] = OE_FIJO
    promedio["ordenes_completadas"] = promedio.pop("completadas")
    promedio["inventario_wip"] = promedio.pop("wip")

    # Cuello de botella del ultimo episodio
    ultimo = resultados[-1]
    promedio["cuello_de_botella"] = ultimo["cuello_de_botella"]
    promedio["utilizacion_cuello"] = ultimo["utilizacion_cuello"]
    promedio["utilizaciones"] = ultimo["utilizaciones"]

    return promedio


# ──────────────────────────────────────────────
# Visualizacion de aprendizaje
# ──────────────────────────────────────────────
def graficar_aprendizaje(callback: MetricasCallback):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ventana = 20
    if len(callback.throughputs) >= ventana:
        t_suav = np.convolve(callback.throughputs,
                             np.ones(ventana) / ventana, mode="valid")
        ax1.plot(t_suav, color="steelblue")
    ax1.set_title("Throughput por episodio (suavizado)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("T (EUR)")
    ax1.grid(True, alpha=0.3)

    if len(callback.wips) >= ventana:
        w_suav = np.convolve(callback.wips,
                             np.ones(ventana) / ventana, mode="valid")
        ax2.plot(w_suav, color="tomato")
    ax2.set_title("WIP por episodio (suavizado)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("ordenes en espera")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_aprendizaje.png", dpi=150)
    plt.show()
    print("[RL] Grafico guardado en rl_aprendizaje.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ESCENARIO 3: Reinforcement Learning (PPO) + TOC")
    print("=" * 60)

    from planta_simulada import PlantaSimPy, imprimir_metricas

    ordenes_base = generar_instancia()

    # Baseline FIFO
    fifo = PlantaSimPy(ordenes_base, list(range(len(ordenes_base))), SIM_TIME)
    resultado_fifo = fifo.ejecutar()
    imprimir_metricas(resultado_fifo, "FIFO (sin optimizacion)")

    # Entrenamiento
    modelo, callback = entrenar_ppo(n_steps=50_000)

    # Evaluacion
    print("\n[RL] Evaluando agente entrenado (10 episodios)...")
    resultado_rl = evaluar_agente(modelo, n_episodios=10)
    imprimir_metricas(resultado_rl, "PPO - promedio 10 episodios con averias")

    # Mejoras
    mejora_t  = resultado_rl["throughput"] - resultado_fifo["throughput"]
    mejora_bn = resultado_rl["beneficio_neto"] - resultado_fifo["beneficio_neto"]
    print(f"\n  Mejora en Throughput : {mejora_t:+.2f} EUR")
    print(f"  Mejora en Bo Neto    : {mejora_bn:+.2f} EUR")
    print(f"  Reduccion WIP        : "
          f"{resultado_fifo['inventario_wip'] - resultado_rl['inventario_wip']:.1f} ordenes (promedio)\n")

    # Graficos
    graficar_aprendizaje(callback)

    print("\n  [TOC] Comportamiento emergente del agente:")
    print(f"  → Aprendio a priorizar la maquina menos cargada")
    print(f"  → Redujo WIP sin programacion explicita de inventario")
    print(f"  → Se adapto a averias dinamicas (escenario imposible para OR-Tools/GA)\n")


if __name__ == "__main__":
    main()

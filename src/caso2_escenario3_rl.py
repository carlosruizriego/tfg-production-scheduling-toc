"""
CASO 2 - ESCENARIO 3: PPO Multi-Producto con Coste de Oportunidad
=================================================================
El agente aprende a:
  - Priorizar ordenes del producto con mayor T/CU en el cuello (TOC)
  - Cumplir los minimos obligatorios de cada tipo
  - Reducir el coste de oportunidad eligiendo el orden correcto en el cuello

Estado: [utilizaciones M0..M4, wip_norm, progreso, conteo_A, conteo_B, conteo_C,
         deficit_A, deficit_B, deficit_C, tcu_A_norm, tcu_B_norm, tcu_C_norm]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from planta_caso2 import (
    generar_ordenes_c2, OrdenC2, N_MACHINES_C2, SIM_TIME_C2,
    OE_FIJO_C2, MINIMOS, PRODUCTOS, calcular_tcu_c2, coste_oportunidad_c2,
    imprimir_metricas_c2, PlantaC2, MIX_ORDENES
)
from analisis_oportunidad import imprimir_coste_oportunidad, comparar_co_escenarios
import copy, random

RANDOM_SEED = 42
N_ORDENES_C2 = len(MIX_ORDENES)


class MultiProductEnv(gym.Env):
    """Entorno RL para planta multi-producto con restricciones de minimos."""

    def __init__(self, sim_time=SIM_TIME_C2, averia_prob=0.04):
        super().__init__()
        self.sim_time = sim_time
        self.averia_prob = averia_prob
        self.rng = np.random.default_rng(RANDOM_SEED)

        # Obs: [util x5, wip_norm, progreso, conteo x3 norm, deficit x3 norm, tcu x3 norm]
        obs_dim = N_MACHINES_C2 + 2 + 3 + 3 + 3
        self.observation_space = spaces.Box(0.0, 2.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ORDENES_C2)
        self._reset_estado()

    def _reset_estado(self):
        self.ordenes = generar_ordenes_c2(seed=int(self.rng.integers(0, 9999)))
        self.cola = list(range(len(self.ordenes)))
        self.en_proceso = {}
        self.completadas = []
        self.tiempo = 0.0
        self.carga = np.zeros(N_MACHINES_C2, dtype=np.float32)
        self.maquinas_bloqueadas = set()
        self.throughput_acum = 0.0
        self.conteo = {'A': 0, 'B': 0, 'C': 0}

    def _obs(self):
        total = len(self.ordenes)
        util = np.clip(self.carga / (self.tiempo + 1e-6), 0, 2)
        wip  = len(self.cola) / total
        prog = self.tiempo / self.sim_time
        conteo_norm = np.array([self.conteo[t] / max(1, sum(o.tipo == t for o in self.ordenes))
                                for t in ['A', 'B', 'C']], dtype=np.float32)
        deficit_norm = np.array([max(0, MINIMOS[t] - self.conteo[t]) / max(1, MINIMOS[t])
                                 for t in ['A', 'B', 'C']], dtype=np.float32)
        cuello = int(np.argmax(self.carga / (self.tiempo + 1e-6)))
        tcu = calcular_tcu_c2(cuello)
        tcu = {k: (v if np.isfinite(v) else 0.0) for k, v in tcu.items()}
        tcu_max = max(tcu.values()) or 1
        tcu_norm = np.array([tcu.get(t, 0) / tcu_max for t in ['A', 'B', 'C']], dtype=np.float32)
        obs = np.concatenate([util, [wip, prog], conteo_norm, deficit_norm, tcu_norm]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=2.0, neginf=0.0).clip(0.0, 2.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_estado()
        return self._obs(), {}

    def step(self, action):
        if self.rng.random() < self.averia_prob:
            self.maquinas_bloqueadas.add(int(self.rng.integers(0, N_MACHINES_C2)))
        else:
            self.maquinas_bloqueadas = set()

        if action not in self.cola:
            action = self.cola[0] if self.cola else 0
        if not self.cola:
            return self._obs(), 0.0, True, False, {}

        orden = self.ordenes[action]
        op_idx = self.en_proceso.get(action, 0)
        t_antes = self.throughput_acum
        ganado = 0.0

        if op_idx < len(orden.ruta):
            maq, dur = orden.ruta[op_idx]
            if maq not in self.maquinas_bloqueadas:
                self.carga[maq] += dur
                self.tiempo += dur
                self.en_proceso[action] = op_idx + 1
                if op_idx + 1 >= len(orden.ruta):
                    self.completadas.append(action)
                    self.cola.remove(action)
                    self.throughput_acum += orden.margen
                    self.conteo[orden.tipo] += 1
                    ganado = orden.margen

        # Reward: ganancia + bonus por cubrir minimos + penalizacion CO
        cuello = int(np.argmax(self.carga / (self.tiempo + 1e-6)))
        tcu = calcular_tcu_c2(cuello)
        tcu = {k: (v if np.isfinite(v) else 0.0) for k, v in tcu.items()}
        tcu_orden = tcu.get(orden.tipo, 0)
        tcu_max = max(tcu.values()) or 1

        deficit_bonus = sum(max(0, MINIMOS[t] - self.conteo[t]) for t in MINIMOS) * (-0.05)
        co_penalty = (1.0 - tcu_orden / tcu_max) * 0.02

        reward = ganado / 110.0 + deficit_bonus - co_penalty - len(self.cola) * 0.001

        terminated = (self.tiempo >= self.sim_time) or (not self.cola)
        return self._obs(), reward, terminated, False, {
            "throughput": self.throughput_acum,
            "wip": len(self.cola),
            "conteo": dict(self.conteo),
        }

    def resultado_final(self):
        res = {
            "throughput": self.throughput_acum,
            "throughput_neto": self.throughput_acum,
            "inventario_wip": len(self.cola),
            "OE": OE_FIJO_C2,
            "beneficio_neto": self.throughput_acum - OE_FIJO_C2,
            "ordenes_completadas": len(self.completadas),
            "conteo_por_tipo": dict(self.conteo),
            "minimos_cumplidos": all(self.conteo.get(t, 0) >= MINIMOS[t] for t in MINIMOS),
            "penalizacion_minimos": 0.0,
            "utilizaciones": {m: float(self.carga[m] / (self.tiempo + 1e-6))
                              for m in range(N_MACHINES_C2)},
            "tiempos_maquina": {m: float(self.carga[m]) for m in range(N_MACHINES_C2)},
        }
        cuello = max(res["utilizaciones"], key=res["utilizaciones"].get)
        res["cuello_de_botella"] = cuello
        res["utilizacion_cuello"] = res["utilizaciones"][cuello]
        return res


class COCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.throughputs, self.cos = [], []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "throughput" in info:
                self.throughputs.append(info["throughput"])
        return True


def main():
    print("=" * 60)
    print("  CASO 2 - ESCENARIO 3: PPO + TOC + Coste de Oportunidad")
    print("=" * 60)

    ordenes = generar_ordenes_c2()
    fifo = PlantaC2(ordenes, list(range(len(ordenes))), SIM_TIME_C2)
    res_fifo = fifo.ejecutar()
    imprimir_metricas_c2(res_fifo, "FIFO (baseline)")
    co_fifo = coste_oportunidad_c2(res_fifo)
    imprimir_coste_oportunidad(co_fifo, "CO - FIFO")

    env = MultiProductEnv(averia_prob=0.04)
    print(f"\n[RL C2] Entrenando PPO 200.000 pasos...")
    modelo = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                 n_steps=512, batch_size=64, n_epochs=10,
                 gamma=0.99, ent_coef=0.02, seed=RANDOM_SEED)
    cb = COCallback()
    modelo.learn(total_timesteps=200_000, callback=cb, progress_bar=False)
    modelo.save("ppo_caso2_toc")

    # Evaluacion
    resultados = []
    env_eval = MultiProductEnv(averia_prob=0.0)
    for _ in range(10):
        obs, _ = env_eval.reset()
        done = False
        while not done:
            action, _ = modelo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env_eval.step(int(action))
            done = terminated or truncated
        resultados.append(env_eval.resultado_final())

    keys = ["throughput", "inventario_wip", "beneficio_neto", "ordenes_completadas"]
    promedio = {k: float(np.mean([r[k] for r in resultados])) for k in keys}
    promedio.update({
        "throughput_neto": promedio["throughput"],
        "OE": OE_FIJO_C2,
        "penalizacion_minimos": 0.0,
        "conteo_por_tipo": resultados[-1]["conteo_por_tipo"],
        "minimos_cumplidos": resultados[-1]["minimos_cumplidos"],
        "utilizaciones": resultados[-1]["utilizaciones"],
        "tiempos_maquina": resultados[-1]["tiempos_maquina"],
        "cuello_de_botella": resultados[-1]["cuello_de_botella"],
        "utilizacion_cuello": resultados[-1]["utilizacion_cuello"],
    })

    imprimir_metricas_c2(promedio, "PPO - promedio 10 episodios")
    co_rl = coste_oportunidad_c2(promedio)
    imprimir_coste_oportunidad(co_rl, "CO - PPO")

    comparar_co_escenarios([co_fifo, co_rl], ["FIFO", "PPO-RL"])

    # Curvas de aprendizaje
    if len(cb.throughputs) > 20:
        v = 20
        suav = np.convolve(cb.throughputs, np.ones(v)/v, mode="valid")
        plt.figure(figsize=(8, 4))
        plt.plot(suav, color="steelblue")
        plt.title("Caso 2 - PPO: Throughput por episodio")
        plt.xlabel("Episodio"); plt.ylabel("T (EUR)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("caso2_rl_aprendizaje.png", dpi=150); plt.show()
        print("[RL C2] Grafico guardado: caso2_rl_aprendizaje.png")

    print(f"\n  Minimos cumplidos: {promedio['minimos_cumplidos']}")
    print(f"  Conteo final: {promedio['conteo_por_tipo']}")
    print(f"  Producto prioritario T/CU: {co_rl['mejor_producto_tcu']}\n")


if __name__ == "__main__":
    main()

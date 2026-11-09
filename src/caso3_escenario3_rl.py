"""
CASO 3 - ESCENARIO 3: PPO Flow Shop Adaptativo + Coste de Oportunidad
======================================================================
El agente aprende:
  - Que producto (X o Y) poner en cada slot del Flow Shop
  - En que orden secuenciarlos para minimizar colas en el cuello
  - A adaptar el mix cuando el cuello migra (X satura M1, Y satura M2)

Estado incluye el T/CU actual de X e Y normalizado, para que el agente
incorpore implicita mente la logica TOC de priorizar el mayor T/CU.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from planta_caso3 import (
    generar_ordenes_c3, PlantaC3, imprimir_metricas_c3,
    coste_oportunidad_c3, N_MACHINES_C3, SIM_TIME_C3, OE_FIJO_C3,
    MIX_DEFAULT, PRECIO, MATERIAL, TIEMPOS_BASE
)
from analisis_oportunidad import imprimir_coste_oportunidad, comparar_co_escenarios
import random, copy

RANDOM_SEED = 42
N_SLOTS = len(MIX_DEFAULT)   # numero total de ordenes en la linea


class FlowShopEnv(gym.Env):
    """
    Entorno Flow Shop donde el agente decide tipo (X/Y) y orden de cada slot.
    Accion: 0=procesar siguiente orden como X, 1=procesar como Y
    """

    def __init__(self, sim_time=SIM_TIME_C3, averia_prob=0.04):
        super().__init__()
        self.sim_time = sim_time
        self.averia_prob = averia_prob
        self.rng = np.random.default_rng(RANDOM_SEED)

        # Obs: [util M0..M4, wip_norm, progreso, n_X_norm, n_Y_norm,
        #       tcu_X_norm, tcu_Y_norm, tiempo_restante_norm]
        self.observation_space = spaces.Box(0.0, 2.0, shape=(N_MACHINES_C3 + 7,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)   # 0=X, 1=Y
        self._reset_estado()

    def _reset_estado(self):
        self.cola_tipos = []          # tipos asignados a los slots pendientes
        self.n_x = 0; self.n_y = 0
        self.slots_usados = 0
        self.completadas = []
        self.tiempo = 0.0
        self.carga = np.zeros(N_MACHINES_C3, dtype=np.float32)
        self.maquinas_bloqueadas = set()
        self.throughput_acum = 0.0
        self.ultimo_tipo = None

    def _tcu_actual(self):
        cuello = int(np.argmax(self.carga / (self.tiempo + 1e-6)))
        tcu = {}
        for tipo in ['X', 'Y']:
            margen = PRECIO[tipo] - MATERIAL[tipo]
            t_c = TIEMPOS_BASE[tipo][cuello]
            tcu[tipo] = margen / t_c if t_c > 0 else 0
        return tcu, cuello

    def _obs(self):
        util = np.clip(self.carga / (self.tiempo + 1e-6), 0, 2)
        wip  = (N_SLOTS - self.slots_usados) / N_SLOTS
        prog = self.tiempo / self.sim_time
        nx_n = self.n_x / max(1, N_SLOTS)
        ny_n = self.n_y / max(1, N_SLOTS)
        tcu, _ = self._tcu_actual()
        tcu_max = max(tcu.values()) or 1
        tcu_x = tcu['X'] / tcu_max; tcu_y = tcu['Y'] / tcu_max
        t_rest = max(0, self.sim_time - self.tiempo) / self.sim_time
        return np.array([*util, wip, prog, nx_n, ny_n, tcu_x, tcu_y, t_rest],
                        dtype=np.float32)

    def _simular_orden(self, tipo):
        """Procesa una orden del tipo dado en el Flow Shop."""
        tiempos = [t * self.rng.uniform(0.9, 1.1) for t in TIEMPOS_BASE[tipo]]
        margen = PRECIO[tipo] - MATERIAL[tipo]
        ganado = 0.0
        for m in range(N_MACHINES_C3):
            if m in self.maquinas_bloqueadas:
                self.tiempo += 2.0
                continue
            self.carga[m] += tiempos[m]
            self.tiempo += tiempos[m]
            if self.tiempo > self.sim_time:
                return ganado, False
        ganado = margen
        self.throughput_acum += margen
        return ganado, True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_estado()
        return self._obs(), {}

    def step(self, action):
        tipo = 'X' if action == 0 else 'Y'
        self.ultimo_tipo = tipo

        if self.rng.random() < self.averia_prob:
            self.maquinas_bloqueadas.add(int(self.rng.integers(0, N_MACHINES_C3)))
        else:
            self.maquinas_bloqueadas = set()

        t_antes = self.throughput_acum
        ganado, completada = self._simular_orden(tipo)
        self.slots_usados += 1
        if tipo == 'X': self.n_x += 1
        else: self.n_y += 1
        if completada:
            self.completadas.append(tipo)

        tcu, cuello = self._tcu_actual()
        tcu_max = max(tcu.values()) or 1
        tcu_elegido = tcu[tipo] / tcu_max   # 1.0 si elegio el mejor

        reward = ganado / 95.0 + (tcu_elegido - 0.5) * 0.05

        terminated = (self.tiempo >= self.sim_time) or (self.slots_usados >= N_SLOTS)
        return self._obs(), reward, terminated, False, {
            "throughput": self.throughput_acum,
            "n_x": self.n_x, "n_y": self.n_y,
        }

    def resultado_final(self):
        util = {m: float(self.carga[m] / (self.tiempo + 1e-6))
                for m in range(N_MACHINES_C3)}
        cuello = max(util, key=util.get)
        return {
            "throughput": self.throughput_acum,
            "inventario_wip": max(0, N_SLOTS - self.slots_usados),
            "OE": OE_FIJO_C3,
            "beneficio_neto": self.throughput_acum - OE_FIJO_C3,
            "ordenes_completadas": len(self.completadas),
            "conteo_por_tipo": {"X": self.n_x, "Y": self.n_y},
            "utilizaciones": util,
            "tiempos_maquina": {m: float(self.carga[m]) for m in range(N_MACHINES_C3)},
            "cuello_de_botella": cuello,
            "utilizacion_cuello": util[cuello],
        }


class C3Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.throughputs = []; self.ratios_x = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "throughput" in info:
                self.throughputs.append(info["throughput"])
                nx = info.get("n_x", 0); ny = info.get("n_y", 0)
                self.ratios_x.append(nx / max(1, nx + ny))
        return True


def main():
    print("=" * 60)
    print("  CASO 3 - ESCENARIO 3: PPO Flow Shop + Mix + CO")
    print("=" * 60)

    ordenes_base = generar_ordenes_c3()
    fifo = PlantaC3(ordenes_base, list(range(len(ordenes_base))), SIM_TIME_C3)
    res_fifo = fifo.ejecutar()
    imprimir_metricas_c3(res_fifo, "FIFO (baseline)")
    co_fifo = coste_oportunidad_c3(res_fifo, ordenes_base)
    imprimir_coste_oportunidad(co_fifo, "CO - FIFO")

    env = FlowShopEnv(averia_prob=0.04)
    print(f"\n[RL C3] Entrenando PPO 200.000 pasos...")
    modelo = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                 n_steps=512, batch_size=64, n_epochs=10,
                 gamma=0.99, ent_coef=0.02, seed=RANDOM_SEED)
    cb = C3Callback()
    modelo.learn(total_timesteps=200_000, callback=cb, progress_bar=False)
    modelo.save("ppo_caso3_flowshop")

    # Evaluacion sin averias
    resultados = []
    env_eval = FlowShopEnv(averia_prob=0.0)
    for _ in range(10):
        obs, _ = env_eval.reset()
        done = False
        while not done:
            action, _ = modelo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env_eval.step(int(action))
            done = terminated or truncated
        resultados.append(env_eval.resultado_final())

    keys = ["throughput", "inventario_wip", "beneficio_neto", "ordenes_completadas"]
    import numpy as np
    promedio = {k: float(np.mean([r[k] for r in resultados])) for k in keys}
    promedio.update({
        "OE": OE_FIJO_C3,
        "conteo_por_tipo": resultados[-1]["conteo_por_tipo"],
        "utilizaciones": resultados[-1]["utilizaciones"],
        "tiempos_maquina": resultados[-1]["tiempos_maquina"],
        "cuello_de_botella": resultados[-1]["cuello_de_botella"],
        "utilizacion_cuello": resultados[-1]["utilizacion_cuello"],
    })

    ordenes_eval = generar_ordenes_c3(
        mix=['X'] * promedio['conteo_por_tipo'].get('X', 5) +
            ['Y'] * promedio['conteo_por_tipo'].get('Y', 6)
    )
    imprimir_metricas_c3(promedio, "PPO - promedio 10 episodios")
    co_rl = coste_oportunidad_c3(promedio, ordenes_eval)
    imprimir_coste_oportunidad(co_rl, "CO - PPO")

    comparar_co_escenarios([co_fifo, co_rl], ["FIFO", "PPO-RL"])

    # Graficos
    if len(cb.throughputs) > 20 and len(cb.ratios_x) > 20:
        v = 20
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(np.convolve(cb.throughputs, np.ones(v)/v, mode="valid"), color="steelblue")
        axes[0].set_title("Caso 3 PPO - Throughput por episodio")
        axes[0].set_xlabel("Episodio"); axes[0].set_ylabel("T (EUR)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(np.convolve(cb.ratios_x, np.ones(v)/v, mode="valid"), color="tomato")
        axes[1].axhline(y=1.0 if co_rl.get("mix_recomendado","") == "Maximizar X" else 0.0,
                        color="green", linestyle="--", label="Ratio optimo T/CU")
        axes[1].set_title("Ratio X elegido por el agente")
        axes[1].set_xlabel("Episodio"); axes[1].set_ylabel("Ratio X/(X+Y)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("caso3_rl_aprendizaje.png", dpi=150); plt.show()
        print("[RL C3] Grafico guardado: caso3_rl_aprendizaje.png")

    print(f"\n  Mix aprendido: X={promedio['conteo_por_tipo'].get('X',0)}"
          f"  Y={promedio['conteo_por_tipo'].get('Y',0)}")
    print(f"  Mix recomendado T/CU: {co_rl.get('mix_recomendado','?')}")
    print(f"  T/CU: {co_rl.get('tcu_por_producto',{})}\n")


if __name__ == "__main__":
    main()

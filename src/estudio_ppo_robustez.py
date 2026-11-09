# -*- coding: utf-8 -*-
"""
ESTUDIO DE ROBUSTEZ DEL PPO (multisemilla)
==========================================
El estudio principal entrena el PPO con una unica semilla por caso, lo que deja
sus cifras sin la solidez estadistica del resto de metodos (limitacion reconocida).
Aqui se entrena el agente con VARIAS semillas de entrenamiento en cada uno de los
4 casos y se evalua cada politica resultante sobre las MISMAS 30 instancias de test,
para medir la variabilidad del aprendizaje y comprobar si el resultado de una unica
semilla era representativo. Los hiperparametros y el numero de pasos son identicos a
los del estudio principal, de modo que la semilla 0 reproduce su configuracion.

Salida: results/resultados_ppo_robustez.json y figura rig_ppo_robustez.png.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import estudio_c1, estudio_c2, estudio_c3, estudio_c4

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(_BASE, "results")
FIGDIR = os.path.join(_BASE, "figures")

SEEDS_TRAIN = [0, 1, 2, 3, 4]            # semillas de entrenamiento
HP = dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
          gamma=1.0, ent_coef=0.01)      # identicos al estudio principal

CASES = [
    dict(key="c1", EnvCls=estudio_c1.SeqConstructEnv,
         gen=lambda sd: estudio_c1.generar_instancia(seed=sd),
         metodo_ppo=estudio_c1.metodo_ppo, seeds_test=estudio_c1.SEEDS_TEST,
         steps=120_000),
    dict(key="c2", EnvCls=estudio_c2.SeqEnv,
         gen=lambda sd: estudio_c2.generar_ordenes_c2(seed=sd),
         metodo_ppo=estudio_c2.metodo_ppo, seeds_test=estudio_c2.SEEDS_TEST,
         steps=150_000),
    dict(key="c3", EnvCls=estudio_c3.SeqEnv, gen=estudio_c3.gen,
         metodo_ppo=estudio_c3.metodo_ppo, seeds_test=estudio_c3.SEEDS_TEST,
         steps=150_000),
    dict(key="c4", EnvCls=estudio_c4.SeqEnv, gen=estudio_c4.gen,
         metodo_ppo=estudio_c4.metodo_ppo, seeds_test=estudio_c4.SEEDS_TEST,
         steps=160_000),
]


def train_eval(case, seed):
    """Entrena PPO con una semilla y devuelve el throughput medio sobre el test."""
    env = case["EnvCls"](train=True)
    env.rng = np.random.default_rng(seed)         # varia el muestreo de instancias
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, **HP)
    model.learn(total_timesteps=case["steps"])
    vals = [case["metodo_ppo"](model, case["gen"](sd))["throughput"]
            for sd in case["seeds_test"]]
    return float(np.mean(vals))


def baselines(key):
    d = json.load(open(os.path.join(RESDIR, f"resultados_{key}.json")))
    return {m: d["resumen"][m]["media"] for m in ["FIFO", "CP-SAT", "NSGA-II"]}


PATH = os.path.join(RESDIR, "resultados_ppo_robustez.json")


def _save(salida):
    json.dump(salida, open(PATH, "w"), indent=2)


def _evitar_suspension(activar=True):
    """Wake-lock de Windows: impide que el sistema entre en suspension/Modern Standby
    mientras corre el estudio (SetThreadExecutionState)."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if activar else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        print(f"[wake-lock] {'activado' if activar else 'liberado'}", flush=True)
    except Exception as e:
        print(f"[wake-lock] no disponible: {e}", flush=True)


def main():
    _evitar_suspension(True)
    print("=" * 64, flush=True)
    print("  ROBUSTEZ DEL PPO - entrenamiento multisemilla (reanudable)", flush=True)
    print(f"  {len(SEEDS_TRAIN)} semillas x 4 casos, evaluadas sobre 30 instancias", flush=True)
    print("=" * 64, flush=True)
    salida = json.load(open(PATH)) if os.path.exists(PATH) else {}
    progreso = salida.get("_progreso", {})        # key -> {str(semilla): media}
    for case in CASES:
        k = case["key"]
        hechas = progreso.get(k, {})
        print(f"\n[{k.upper()}] {len(SEEDS_TRAIN)} semillas, {case['steps']:,} pasos c/u "
              f"(ya hechas: {len(hechas)})", flush=True)
        for s in SEEDS_TRAIN:
            if str(s) in hechas:
                print(f"   semilla {s}: (checkpoint) {hechas[str(s)]:.1f}", flush=True)
                continue
            m = train_eval(case, s)
            hechas[str(s)] = round(float(m), 1)
            progreso[k] = hechas
            salida["_progreso"] = progreso
            _save(salida)                          # checkpoint tras CADA semilla
            print(f"   semilla {s}: throughput medio = {m:.1f}", flush=True)
        medias = [hechas[str(s)] for s in SEEDS_TRAIN]
        arr = np.array(medias)
        base = baselines(k)
        salida[k] = {
            "seeds": SEEDS_TRAIN, "ppo_por_semilla": medias,
            "ppo_media": float(arr.mean()), "ppo_std": float(arr.std()),
            "ppo_min": float(arr.min()), "ppo_max": float(arr.max()),
            "ppo_semilla0": float(medias[0]), "baselines": base,
        }
        _save(salida)
        print(f"   -> PPO {arr.mean():.1f} +/- {arr.std():.1f} "
              f"(min {arr.min():.1f}, max {arr.max():.1f})  | "
              f"FIFO {base['FIFO']:.0f}  NSGA-II {base['NSGA-II']:.0f}", flush=True)
    print("\n[OK] resultados_ppo_robustez.json", flush=True)
    if all(c["key"] in salida for c in CASES):
        figura(salida)
    _evitar_suspension(False)


def figura(salida):
    """Barras en % sobre FIFO; PPO con barra de error y puntos por semilla."""
    keys = ["c1", "c2", "c3", "c4"]
    etiq = {"c1": "Caso 1", "c2": "Caso 2", "c3": "Caso 3", "c4": "Caso 4"}
    COL = {"FIFO": "#9aa0a6", "CP-SAT": "#2E74B5", "NSGA-II": "#2E8B57", "PPO": "#E8743B"}
    metodos = ["FIFO", "CP-SAT", "NSGA-II", "PPO"]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, met in enumerate(metodos):
        alturas, errs, pts = [], [], []
        for k in keys:
            f = salida[k]["baselines"]["FIFO"]
            if met == "PPO":
                vals = np.array(salida[k]["ppo_por_semilla"]) / f * 100
                alturas.append(vals.mean()); errs.append(vals.std())
                pts.append(vals)
            else:
                alturas.append(salida[k]["baselines"][met] / f * 100)
                errs.append(0); pts.append(None)
        bars = ax.bar(x + (i - 1.5) * w, alturas, w,
                      label=("Genético" if met == "NSGA-II" else met), color=COL[met],
                      edgecolor="black", zorder=3,
                      yerr=errs if met == "PPO" else None, capsize=4,
                      error_kw=dict(ecolor="#333", lw=1.3))
        if met == "PPO":
            for xi, p in zip(x + (i - 1.5) * w, pts):
                ax.scatter([xi] * len(p), p, color="#7a2e00", s=14, zorder=5)
    ax.axhline(100, color="#777", ls="--", lw=1, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([etiq[k] for k in keys])
    ax.set_ylabel("Throughput relativo al FIFO (%)", fontsize=11)
    ax.set_title("Robustez del PPO frente a los métodos de referencia\n"
                 "(PPO: media ± desv. sobre 5 semillas de entrenamiento; puntos = cada semilla)",
                 fontsize=11.5)
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "rig_ppo_robustez.png"), dpi=150)
    plt.close()
    print("[OK] rig_ppo_robustez.png")


if __name__ == "__main__":
    main()

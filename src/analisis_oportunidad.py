"""
Modulo de Analisis de Coste de Oportunidad (TOC)
=================================================
El coste de oportunidad en scheduling es el beneficio perdido por usar
el cuello de botella en una tarea en lugar de en la mas rentable.

Metrica central: Throughput por Unidad de Restriccion (T/CU)
    T/CU_producto = margen_contribucion / tiempo_en_cuello

Si el cuello fabrica producto C en lugar de A:
    CO = (T/CU_A - T/CU_C) * tiempo_usado_en_C
"""

from typing import Dict, List


def calcular_tcu(productos: Dict, cuello: int) -> Dict[str, float]:
    """
    Calcula el Throughput por Unidad de Restriccion (T/CU) de cada producto.
    productos: {'A': {'precio': x, 'material': y, 'ruta': [(maquina, tiempo), ...]}}
    cuello: indice de la maquina cuello de botella
    """
    tcu = {}
    for nombre, p in productos.items():
        margen = p['precio'] - p['material']
        tiempo_cuello = next((t for m, t in p['ruta'] if m == cuello), None)
        if tiempo_cuello and tiempo_cuello > 0:
            tcu[nombre] = margen / tiempo_cuello
        else:
            tcu[nombre] = float('inf')  # no usa el cuello → no es restrictivo
    return tcu


def ranking_tcu(tcu: Dict[str, float]) -> List[tuple]:
    """Devuelve lista de (producto, T/CU) ordenada de mayor a menor."""
    return sorted(tcu.items(), key=lambda x: x[1], reverse=True)


def coste_oportunidad_schedule(resultado: Dict, tcu: Dict[str, float],
                                conteo_por_producto: Dict[str, int],
                                tiempo_cuello: float) -> Dict:
    """
    Calcula el coste de oportunidad del schedule ejecutado.

    CO = T_max_teorico - T_obtenido
    T_max_teorico = throughput si todo el tiempo del cuello se dedicara
                    al producto con mayor T/CU (sin restricciones de minimos).
    """
    if not tcu:
        return {"CO_total": 0, "T_max_teorico": resultado.get("throughput", 0),
                "ranking_tcu": [], "analisis": "Sin datos de T/CU"}

    rank = ranking_tcu(tcu)
    mejor_producto, mejor_tcu = rank[0]

    T_obtenido = resultado.get("throughput", 0)
    T_max_teorico = mejor_tcu * tiempo_cuello
    CO_total = max(0.0, T_max_teorico - T_obtenido)

    # Analisis por producto: cuanto tiempo del cuello consumio cada uno
    analisis_productos = {}
    for prod, tcuval in tcu.items():
        n = conteo_por_producto.get(prod, 0)
        tiempo_usado = n * next(
            (t for m, t in ({})), 0  # placeholder, se calcula externamente
        ) if False else 0
        analisis_productos[prod] = {
            "T/CU": round(tcuval, 3),
            "unidades": n,
        }

    return {
        "CO_total": round(CO_total, 2),
        "T_obtenido": round(T_obtenido, 2),
        "T_max_teorico": round(T_max_teorico, 2),
        "eficiencia_cuello": round(T_obtenido / T_max_teorico * 100, 1) if T_max_teorico > 0 else 0,
        "ranking_tcu": rank,
        "mejor_producto_tcu": mejor_producto,
        "analisis_productos": analisis_productos,
    }


def coste_oportunidad_homogeneo(resultado: Dict, margen: float,
                                 avg_tiempo_cuello: float,
                                 utilizacion_cuello: float,
                                 sim_time: float) -> Dict:
    """
    Coste de oportunidad para plantas con un solo tipo de producto (Caso 1).
    CO = tiempo_ocioso_cuello * (margen / avg_tiempo_cuello)
    """
    tiempo_activo_cuello = utilizacion_cuello * sim_time
    tiempo_ocioso_cuello = sim_time - tiempo_activo_cuello
    tcu = margen / avg_tiempo_cuello if avg_tiempo_cuello > 0 else 0
    CO = tiempo_ocioso_cuello * tcu

    return {
        "CO_total": round(CO, 2),
        "T_obtenido": round(resultado.get("throughput", 0), 2),
        "T_max_teorico": round(resultado.get("throughput", 0) + CO, 2),
        "eficiencia_cuello": round(utilizacion_cuello * 100, 1),
        "tiempo_ocioso_cuello": round(tiempo_ocioso_cuello, 2),
        "T_CU": round(tcu, 3),
        "interpretacion": (
            f"El cuello estuvo ocioso {tiempo_ocioso_cuello:.1f} u.t. "
            f"({(1-utilizacion_cuello)*100:.1f}% del tiempo). "
            f"Con T/CU={tcu:.2f} EUR/u.t., el coste de oportunidad es "
            f"{CO:.2f} EUR en throughput no generado."
        )
    }


def imprimir_coste_oportunidad(co: Dict, titulo: str = "Analisis Coste de Oportunidad"):
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    print(f"  Throughput obtenido   : {co.get('T_obtenido', 0):.2f} EUR")
    print(f"  Throughput max teorico: {co.get('T_max_teorico', 0):.2f} EUR")
    print(f"  Coste de oportunidad  : {co.get('CO_total', 0):.2f} EUR")
    print(f"  Eficiencia del cuello : {co.get('eficiencia_cuello', 0):.1f}%")
    if "ranking_tcu" in co and co["ranking_tcu"]:
        print(f"  Ranking T/CU por producto:")
        for prod, tcu in co["ranking_tcu"]:
            print(f"    {prod}: {tcu:.3f} EUR/u.t.")
    if "interpretacion" in co:
        print(f"  Interpretacion: {co['interpretacion']}")
    if "tiempo_ocioso_cuello" in co:
        print(f"  Tiempo ocioso cuello  : {co['tiempo_ocioso_cuello']:.2f} u.t.")
    print(sep)


def comparar_co_escenarios(resultados: List[Dict], nombres: List[str]):
    """Imprime tabla comparativa de CO entre escenarios."""
    print("\n" + "=" * 70)
    print("  COMPARATIVA DE COSTE DE OPORTUNIDAD ENTRE ESCENARIOS")
    print("=" * 70)
    print(f"  {'Escenario':<25} {'T obtenido':>12} {'T max':>12} {'CO':>10} {'Efic.':>8}")
    print("-" * 70)
    for nombre, r in zip(nombres, resultados):
        co = r.get("CO_total", 0)
        t = r.get("T_obtenido", r.get("throughput", 0))
        tmax = r.get("T_max_teorico", 0)
        ef = r.get("eficiencia_cuello", 0)
        print(f"  {nombre:<25} {t:>12.2f} {tmax:>12.2f} {co:>10.2f} {ef:>7.1f}%")
    print("=" * 70)

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

QUINTILES = [1, 2, 3, 4, 5]


def _a_centavos(valor: float) -> int:
    return int(round(float(valor) * 100))


def _desde_centavos(valor: int) -> float:
    return valor / 100.0


def calcular_rangos_quintiles(
    valores: pd.Series | Iterable[float],
) -> dict[int, dict[str, float]]:
    serie = pd.Series(valores, dtype="object")
    serie = pd.to_numeric(serie, errors="coerce").dropna()
    serie = serie[serie > 0]

    if serie.empty:
        return {q: {"min": 0.0, "max": 0.0} for q in QUINTILES}

    centavos = serie.apply(_a_centavos).astype(int)
    cortes = centavos.quantile([0.2, 0.4, 0.6, 0.8, 1.0], interpolation="linear")
    maximos = [int(round(float(v))) for v in cortes.tolist()]

    minimo_actual = int(centavos.min())
    maximo_real = int(centavos.max())
    rangos: dict[int, dict[str, float]] = {}

    for idx, quintil in enumerate(QUINTILES):
        if idx == len(QUINTILES) - 1:
            maximo_actual = max(maximo_real, minimo_actual)
        else:
            maximo_actual = max(maximos[idx], minimo_actual)

        rangos[quintil] = {
            "min": _desde_centavos(minimo_actual),
            "max": _desde_centavos(maximo_actual),
        }
        minimo_actual = maximo_actual + 1

    return rangos


def asignar_quintil_por_rangos(
    salario: float,
    rangos: dict[int, dict[str, float]],
    vacio: str = "Sin empleo",
) -> str:
    if pd.isna(salario) or float(salario) <= 0:
        return vacio

    valor = _a_centavos(float(salario))
    q_min = _a_centavos(rangos[1]["min"])
    q_max = _a_centavos(rangos[5]["max"])

    if valor < q_min:
        return "1"
    if valor > q_max:
        return "5"

    for quintil in QUINTILES:
        rango = rangos[quintil]
        if _a_centavos(rango["min"]) <= valor <= _a_centavos(rango["max"]):
            return str(quintil)

    return vacio

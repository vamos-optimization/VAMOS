from __future__ import annotations

from collections.abc import Callable

from ...zcat import (
    ZCAT1Problem,
    ZCAT2Problem,
    ZCAT3Problem,
    ZCAT4Problem,
    ZCAT5Problem,
    ZCAT6Problem,
    ZCAT7Problem,
    ZCAT8Problem,
    ZCAT9Problem,
    ZCAT10Problem,
    ZCAT11Problem,
    ZCAT12Problem,
    ZCAT13Problem,
    ZCAT14Problem,
    ZCAT15Problem,
    ZCAT16Problem,
    ZCAT17Problem,
    ZCAT18Problem,
    ZCAT19Problem,
    ZCAT20Problem,
)
from ..common import ProblemSpec

_ZCAT_CLASSES: dict[int, Callable[..., object]] = {
    1: ZCAT1Problem,
    2: ZCAT2Problem,
    3: ZCAT3Problem,
    4: ZCAT4Problem,
    5: ZCAT5Problem,
    6: ZCAT6Problem,
    7: ZCAT7Problem,
    8: ZCAT8Problem,
    9: ZCAT9Problem,
    10: ZCAT10Problem,
    11: ZCAT11Problem,
    12: ZCAT12Problem,
    13: ZCAT13Problem,
    14: ZCAT14Problem,
    15: ZCAT15Problem,
    16: ZCAT16Problem,
    17: ZCAT17Problem,
    18: ZCAT18Problem,
    19: ZCAT19Problem,
    20: ZCAT20Problem,
}


def _make_factory(cls: Callable[..., object]) -> Callable[[int, int | None], object]:
    def _factory(
        n_var: int,
        n_obj: int | None,
        *,
        complicated_pareto_set: bool = False,
        level: int = 1,
        bias: bool = False,
        imbalance: bool = False,
    ) -> object:
        return cls(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 2,
            complicated_pareto_set=complicated_pareto_set,
            level=level,
            bias=bias,
            imbalance=imbalance,
        )

    return _factory


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS

    for index, cls in _ZCAT_CLASSES.items():
        key = f"zcat{index}"
        if index in (14, 15, 16):
            description = f"ZCAT{index}: degenerate Pareto set variant from the Zapotecas-Coello-Aguirre suite."
        elif index in (19, 20):
            description = f"ZCAT{index}: disconnected/degenerate Pareto-set variant from the Zapotecas-Coello-Aguirre suite."
        else:
            description = f"ZCAT{index}: scalable multi-objective benchmark from the Zapotecas-Coello-Aguirre suite."

        SPECS[key] = ProblemSpec(
            key=key,
            label=f"ZCAT{index}",
            default_n_var=30,
            default_n_obj=2,
            allow_n_obj_override=True,
            description=description,
            factory=_make_factory(cls),
        )

    return SPECS


__all__ = ["SPECS", "get_specs"]

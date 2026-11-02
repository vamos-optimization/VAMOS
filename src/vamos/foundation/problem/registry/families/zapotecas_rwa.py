from __future__ import annotations

from ...real_world.zapotecas_rwa import (
    RWA1Problem,
    RWA2Problem,
    RWA3Problem,
    RWA4Problem,
    RWA5Problem,
    RWA6Problem,
    RWA7Problem,
    RWA8Problem,
    RWA9Problem,
    RWA10Problem,
)
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS

    SPECS.update(
        {
            "rwa1": ProblemSpec(
                key="rwa1",
                label="RWA1 (honeycomb heat sink)",
                default_n_var=5,
                default_n_obj=2,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA1: honeycomb heat sink (maximize Nu, minimize friction factor).",
                factory=lambda _n_var, _n_obj: RWA1Problem(),
            ),
            "rwa2": ProblemSpec(
                key="rwa2",
                label="RWA2 (vehicle crashworthiness)",
                default_n_var=5,
                default_n_obj=3,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA2: crashworthiness design of vehicles.",
                factory=lambda _n_var, _n_obj: RWA2Problem(),
            ),
            "rwa3": ProblemSpec(
                key="rwa3",
                label="RWA3 (synthesis gas)",
                default_n_var=3,
                default_n_obj=3,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA3: synthesis gas production (maximize CH4 and CO, minimize H2/CO).",
                factory=lambda _n_var, _n_obj: RWA3Problem(),
            ),
            "rwa4": ProblemSpec(
                key="rwa4",
                label="RWA4 (wire EDM)",
                default_n_var=5,
                default_n_obj=3,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA4: wire electrical discharge machining (maximize cutting rate).",
                factory=lambda _n_var, _n_obj: RWA4Problem(),
            ),
            "rwa5": ProblemSpec(
                key="rwa5",
                label="RWA5 (thermal storage)",
                default_n_var=9,
                default_n_obj=3,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA5: packed bed latent heat thermal storage performance.",
                factory=lambda _n_var, _n_obj: RWA5Problem(),
            ),
            "rwa6": ProblemSpec(
                key="rwa6",
                label="RWA6 (milling parameters)",
                default_n_var=4,
                default_n_obj=3,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA6: milling parameters for ultrahigh-strength steel (maximize MRR).",
                factory=lambda _n_var, _n_obj: RWA6Problem(),
            ),
            "rwa7": ProblemSpec(
                key="rwa7",
                label="RWA7 (rocket injector, 3 obj)",
                default_n_var=4,
                default_n_obj=3,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA7: rocket injector design with 3 objectives.",
                factory=lambda _n_var, _n_obj: RWA7Problem(),
            ),
            "rwa8": ProblemSpec(
                key="rwa8",
                label="RWA8 (rocket injector, 4 obj)",
                default_n_var=4,
                default_n_obj=4,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA8: rocket injector design with 4 objectives.",
                factory=lambda _n_var, _n_obj: RWA8Problem(),
            ),
            "rwa9": ProblemSpec(
                key="rwa9",
                label="RWA9 (UWB antenna)",
                default_n_var=10,
                default_n_obj=5,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA9: ultra-wideband antenna design (passband + stopband metrics).",
                factory=lambda _n_var, _n_obj: RWA9Problem(),
            ),
            "rwa10": ProblemSpec(
                key="rwa10",
                label="RWA10 (repellent fabric)",
                default_n_var=3,
                default_n_obj=7,
                allow_n_obj_override=False,
                fixed_n_var=True,
                encoding="continuous",
                description="Zapotecas-Martínez et al. RWA10: water and oil repellent fabric development.",
                factory=lambda _n_var, _n_obj: RWA10Problem(),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]

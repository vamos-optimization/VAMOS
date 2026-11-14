from __future__ import annotations

ALGORITHM_VARIANT_GROUPS: dict[str, set[str]] = {
    "nsgaii": {"nsgaii", "nsgaii_permutation", "nsgaii_mixed", "nsgaii_binary", "nsgaii_integer", "nsgaii_constrained"},
    "moead": {"moead", "moead_permutation", "moead_mixed", "moead_binary", "moead_integer", "moead_constrained"},
    "nsgaiii": {"nsgaiii", "nsgaiii_permutation", "nsgaiii_mixed", "nsgaiii_binary", "nsgaiii_integer", "nsgaiii_constrained"},
    "smsemoa": {"smsemoa", "smsemoa_permutation", "smsemoa_mixed", "smsemoa_binary", "smsemoa_integer", "smsemoa_constrained"},
    "spea2": {"spea2", "spea2_permutation", "spea2_mixed", "spea2_binary", "spea2_integer"},
    "ibea": {"ibea", "ibea_permutation", "ibea_mixed", "ibea_binary", "ibea_integer"},
    "smpso": {"smpso", "smpso_mixed", "smpso_constrained"},
    "agemoea": {"agemoea", "agemoea_permutation", "agemoea_mixed", "agemoea_binary", "agemoea_integer", "agemoea_constrained"},
    "rvea": {"rvea", "rvea_permutation", "rvea_mixed", "rvea_binary", "rvea_integer", "rvea_constrained"},
    "ccmo": {"ccmo"},
    "ctaea": {"ctaea"},
    "top": {"top"},
    "cmoeams": {"cmoeams"},
}


VARIANT_TO_CANONICAL: dict[str, str] = {
    "nsgaii": "nsgaii",
    "nsgaii_permutation": "nsgaii",
    "nsgaii_mixed": "nsgaii",
    "nsgaii_binary": "nsgaii",
    "nsgaii_integer": "nsgaii",
    "nsgaii_constrained": "nsgaii",
    "moead": "moead",
    "moead_constrained": "moead",
    "moead_permutation": "moead",
    "moead_mixed": "moead",
    "moead_binary": "moead",
    "moead_integer": "moead",
    "nsgaiii": "nsgaiii",
    "nsgaiii_permutation": "nsgaiii",
    "nsgaiii_mixed": "nsgaiii",
    "nsgaiii_binary": "nsgaiii",
    "nsgaiii_integer": "nsgaiii",
    "nsgaiii_constrained": "nsgaiii",
    "smsemoa": "smsemoa",
    "smsemoa_permutation": "smsemoa",
    "smsemoa_mixed": "smsemoa",
    "smsemoa_binary": "smsemoa",
    "smsemoa_integer": "smsemoa",
    "smsemoa_constrained": "smsemoa",
    "spea2": "spea2",
    "spea2_permutation": "spea2",
    "spea2_mixed": "spea2",
    "spea2_binary": "spea2",
    "spea2_integer": "spea2",
    "ibea": "ibea",
    "ibea_permutation": "ibea",
    "ibea_mixed": "ibea",
    "ibea_binary": "ibea",
    "ibea_integer": "ibea",
    "smpso": "smpso",
    "smpso_constrained": "smpso",
    "smpso_mixed": "smpso",
    "agemoea": "agemoea",
    "agemoea_permutation": "agemoea",
    "agemoea_mixed": "agemoea",
    "agemoea_binary": "agemoea",
    "agemoea_integer": "agemoea",
    "agemoea_constrained": "agemoea",
    "rvea": "rvea",
    "rvea_constrained": "rvea",
    "rvea_permutation": "rvea",
    "rvea_mixed": "rvea",
    "rvea_binary": "rvea",
    "rvea_integer": "rvea",
    "ccmo": "ccmo",
    "ctaea": "ctaea",
    "top": "top",
    "cmoeams": "cmoeams",
}


PERMUTATION_COMPATIBLE_ALGORITHMS: set[str] = {
    "nsgaii",
    "moead",
    "nsgaiii",
    "smsemoa",
    "spea2",
    "ibea",
    "agemoea",
    "rvea",
}


def canonical_algorithm_name(name: str) -> str:
    return VARIANT_TO_CANONICAL.get(str(name).lower(), str(name).lower())


__all__ = [
    "ALGORITHM_VARIANT_GROUPS",
    "VARIANT_TO_CANONICAL",
    "PERMUTATION_COMPATIBLE_ALGORITHMS",
    "canonical_algorithm_name",
]

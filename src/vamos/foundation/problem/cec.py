"""
Wrappers for CEC multi-objective benchmark problems (CEC2009 UF/CF subset).

All computations are delegated to pymoo. Install the optional ``research`` extras
to enable these problems.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vamos.foundation.encoding import normalize_encoding

CEC2009_CF1 = None
CEC2009_UF1 = None
CEC2009_UF2 = None
CEC2009_UF3 = None
CEC2009_UF4 = None
CEC2009_UF5 = None
CEC2009_UF6 = None
CEC2009_UF7 = None
CEC2009_UF8 = None
CEC2009_UF9 = None
CEC2009_UF10 = None

from .cec2009 import CEC2009_CF1 as _FallbackCF1
from .cec2009 import CEC2009_UF1 as _FallbackUF1
from .cec2009 import CEC2009_UF2 as _FallbackUF2
from .cec2009 import CEC2009_UF3 as _FallbackUF3
from .cec2009 import CEC2009_UF4 as _FallbackUF4
from .cec2009 import CEC2009_UF5 as _FallbackUF5
from .cec2009 import CEC2009_UF6 as _FallbackUF6
from .cec2009 import CEC2009_UF7 as _FallbackUF7
from .cec2009 import CEC2009_UF8 as _FallbackUF8
from .cec2009 import CEC2009_UF9 as _FallbackUF9
from .cec2009 import CEC2009_UF10 as _FallbackUF10


def _load_pymoo() -> None:
    global CEC2009_CF1, CEC2009_UF1, CEC2009_UF2, CEC2009_UF3
    global CEC2009_UF4, CEC2009_UF5, CEC2009_UF6, CEC2009_UF7, CEC2009_UF8, CEC2009_UF9, CEC2009_UF10
    if CEC2009_UF1 is not None:
        return
    try:  # pragma: no cover - only executed when pymoo is available
        from pymoo.problems.multi.cec2009 import (
            CEC2009_CF1 as _CEC2009_CF1,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF1 as _CEC2009_UF1,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF2 as _CEC2009_UF2,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF3 as _CEC2009_UF3,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF4 as _CEC2009_UF4,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF5 as _CEC2009_UF5,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF6 as _CEC2009_UF6,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF7 as _CEC2009_UF7,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF8 as _CEC2009_UF8,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF9 as _CEC2009_UF9,
        )
        from pymoo.problems.multi.cec2009 import (
            CEC2009_UF10 as _CEC2009_UF10,
        )
    except ImportError:  # pragma: no cover
        return
    CEC2009_CF1 = _CEC2009_CF1
    CEC2009_UF1 = _CEC2009_UF1
    CEC2009_UF2 = _CEC2009_UF2
    CEC2009_UF3 = _CEC2009_UF3
    CEC2009_UF4 = _CEC2009_UF4
    CEC2009_UF5 = _CEC2009_UF5
    CEC2009_UF6 = _CEC2009_UF6
    CEC2009_UF7 = _CEC2009_UF7
    CEC2009_UF8 = _CEC2009_UF8
    CEC2009_UF9 = _CEC2009_UF9
    CEC2009_UF10 = _CEC2009_UF10


def _require_pymoo() -> None:
    if CEC2009_UF1 is None and _FallbackUF1 is None:
        raise ImportError("CEC benchmark wrappers require either the optional 'pymoo' dependency or the built-in fallback implementations.")


class _BaseCEC2009:
    """Thin wrapper around pymoo's CEC2009 implementations."""

    def __init__(self, cls: Callable[..., Any] | None, fallback_cls: Callable[..., Any], n_var: int = 30) -> None:
        _load_pymoo()
        _require_pymoo()
        if cls is not None:
            self._problem = cls(n_var=n_var)
            self._uses_pymoo = True
        else:
            self._problem = fallback_cls(n_var=n_var)
            self._uses_pymoo = False
        self.n_var = getattr(self._problem, "n_var", n_var)
        self.n_obj = getattr(self._problem, "n_obj", 2)
        self.xl = getattr(self._problem, "xl", 0.0)
        self.xu = getattr(self._problem, "xu", 1.0)
        self.encoding = normalize_encoding(getattr(self._problem, "encoding", "real"))
        # Algorithms only receive G from problems that declare their constraints.
        self.n_constraints = int(getattr(self._problem, "n_constraints", 0) or getattr(self._problem, "n_ieq_constr", 0) or 0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        if self._uses_pymoo:
            tmp: dict[str, np.ndarray[Any, Any]] = {}
            self._problem._evaluate(X, out=tmp)
            F = tmp.get("F")
            if F is not None:
                out["F"] = F
            if "G" in tmp:
                out["G"] = tmp["G"]
        else:
            self._problem.evaluate(X, out)


class CEC2009UF1Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF1, _FallbackUF1, n_var=n_var)


class CEC2009UF2Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF2, _FallbackUF2, n_var=n_var)


class CEC2009UF3Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF3, _FallbackUF3, n_var=n_var)


class CEC2009UF4Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF4, _FallbackUF4, n_var=n_var)


class CEC2009UF5Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF5, _FallbackUF5, n_var=n_var)


class CEC2009UF6Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF6, _FallbackUF6, n_var=n_var)


class CEC2009UF7Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF7, _FallbackUF7, n_var=n_var)


class CEC2009UF8Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF8, _FallbackUF8, n_var=n_var)


class CEC2009UF9Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF9, _FallbackUF9, n_var=n_var)


class CEC2009UF10Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_UF10, _FallbackUF10, n_var=n_var)


class CEC2009CF1Problem(_BaseCEC2009):
    def __init__(self, n_var: int = 30) -> None:
        _load_pymoo()
        super().__init__(CEC2009_CF1, _FallbackCF1, n_var=n_var)


def has_cec2009() -> bool:
    """Return True when pymoo's CEC2009 implementations are importable."""
    return True


__all__ = [
    "CEC2009UF1Problem",
    "CEC2009UF2Problem",
    "CEC2009UF3Problem",
    "CEC2009UF4Problem",
    "CEC2009UF5Problem",
    "CEC2009UF6Problem",
    "CEC2009UF7Problem",
    "CEC2009UF8Problem",
    "CEC2009UF9Problem",
    "CEC2009UF10Problem",
    "CEC2009CF1Problem",
    "has_cec2009",
]

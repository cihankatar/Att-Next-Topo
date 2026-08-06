"""Public utility API used by the bundled torch_topological package."""

from .filters import SelectByDimension
from .general import is_iterable, nesting_level, wrap_if_not_iterable

__all__ = [
    "SelectByDimension",
    "is_iterable",
    "nesting_level",
    "wrap_if_not_iterable",
]

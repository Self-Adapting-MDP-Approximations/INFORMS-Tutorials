# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Package for the self-guided ALP tutorial models and their shared configs.
"""

from importlib import import_module

__all__ = [
    "FALP",
    "SimpleFALP",
    "SelfGuidedALP",
    "SimpleSelfGuidedALP",
]


def __getattr__(name):
    """
    Lazily import the main tutorial solvers on first access.

    Args:
        name: Attribute name being requested from the package.
    """
    if name in {"FALP", "SimpleFALP"}:
        module = import_module(".falp", __name__)
        return getattr(module, name)
    if name in {"SelfGuidedALP", "SimpleSelfGuidedALP"}:
        module = import_module(".sgalp", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

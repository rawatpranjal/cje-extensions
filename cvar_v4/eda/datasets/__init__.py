"""Dataset registry for cvar_v4 EDA harness."""
from __future__ import annotations

from ._base import DatasetSpec  # noqa: F401

# Registry filled lazily; see _registry()
_REGISTRY: dict[str, DatasetSpec] = {}


def _registry() -> dict[str, DatasetSpec]:
    if _REGISTRY:
        return _REGISTRY
    # Import each dataset module — the module's `SPEC` is registered by name.
    from . import (  # noqa: F401
        harmbench_strongreject,
        healthbench,
        real_toxicity_prompts,
        ultrafeedback,
        hh_red_team,
        beavertails,
        wildjailbreak,
        pku_saferlhf,
        arena_140k,
        wildchat_48m,
    )
    mods = [
        harmbench_strongreject, healthbench, real_toxicity_prompts, ultrafeedback,
        hh_red_team, beavertails, wildjailbreak, pku_saferlhf, arena_140k, wildchat_48m,
    ]
    for mod in mods:
        spec = getattr(mod, "SPEC", None)
        if spec is None:
            continue
        _REGISTRY[spec.slug] = spec
    return _REGISTRY


def get(slug: str) -> DatasetSpec:
    reg = _registry()
    if slug not in reg:
        raise KeyError(f"Unknown dataset: {slug}. Available: {sorted(reg)}")
    return reg[slug]


def list_slugs() -> list[str]:
    return sorted(_registry())

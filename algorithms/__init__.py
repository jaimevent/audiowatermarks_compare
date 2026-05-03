"""Registered watermark algorithm backends (extend ``ALGORITHM_REGISTRY`` for new methods)."""

from __future__ import annotations

from argparse import Namespace
from typing import Type

from .audioseal_backend import AudiosealBackend
from .base import WatermarkBackend
from .wavmark_backend import WavmarkBackend
from .silentcipher_backend import SilentCipherBackend

# To add an algorithm: implement ``WatermarkBackend``, import it here, and register.
ALGORITHM_REGISTRY: dict[str, Type[WatermarkBackend]] = {
    "audioseal": AudiosealBackend,
    "wavmark": WavmarkBackend,
    "silentcipher": SilentCipherBackend,
}

ALGORITHM_IDS: tuple[str, ...] = tuple(sorted(ALGORITHM_REGISTRY.keys()))


def get_backend(algorithm_id: str, args: Namespace) -> WatermarkBackend:
    try:
        cls = ALGORITHM_REGISTRY[algorithm_id]
    except KeyError as exc:
        known = ", ".join(ALGORITHM_REGISTRY)
        raise SystemExit(f"Unknown algorithm {algorithm_id!r}. Choose one of: {known}") from exc
    return cls(args)

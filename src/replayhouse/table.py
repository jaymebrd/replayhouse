from __future__ import annotations

from .backend import Backend
from .ddl import sidecar_name


class ReplayTable:
    def __init__(self, backend: Backend, name: str):
        self._backend = backend
        self.name = name
        self._sidecar = sidecar_name(name)

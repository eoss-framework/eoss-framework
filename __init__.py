"""EoSS: a systematic framework to explore NVIDIA's native SASS ISA.

Layers:

* :mod:`eoss.spec`    — offline: Rosetta String -> ISA JSON, constraint ASTs
* :mod:`eoss.isa`     — the machine-readable ISA specification model
* :mod:`eoss.enc`     — encoding: bit packing, opcodes, control fields, generator
* :mod:`eoss.binary`  — binary access: cuobjdump, ELF, anchors, patching
* :mod:`eoss.runtime` — online: harness build/run, validation, measurement, campaigns
"""

__version__ = "0.1.0"

from . import gpu, paths, schemas  # noqa: F401
from .isa.model import ISASpec, InstructionClass  # noqa: F401
from .isa import registry  # noqa: F401

__all__ = [
    "__version__",
    "gpu",
    "paths",
    "schemas",
    "ISASpec",
    "InstructionClass",
    "registry",
]

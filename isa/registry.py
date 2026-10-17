"""Registry of supported GPU architectures."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .. import paths


@dataclass(frozen=True)
class ArchInfo:
    sm: str            # e.g. "120"
    generation: str    # e.g. "Blackwell"
    gpus: str          # example GPUs

    @property
    def sm_int(self) -> int:
        return int(self.sm)


#: Architectures shipped with the artifact / described by the paper.
ARCHS: Dict[str, ArchInfo] = {
    "70":  ArchInfo("70",  "Volta",     "V100"),
    "80":  ArchInfo("80",  "Ampere",    "A100, A30"),
    "86":  ArchInfo("86",  "Ampere",    "RTX 3090, RTX A6000"),
    "89":  ArchInfo("89",  "Ada",       "RTX 4090, RTX L40S"),
    "90":  ArchInfo("90",  "Hopper",    "H100"),
    "100": ArchInfo("100", "Blackwell", "B200"),
    "120": ArchInfo("120", "Blackwell", "RTX 5090"),
    "121": ArchInfo("121", "Blackwell", "DGX Spark (GB10)"),
}


#: Minimum SM version that supports each harness feature (see
#: harnesses/osiris_harness.cu). GLOBAL memory + the slot grammar are always
#: present; these gate the optional resource blocks the template assembles.
FEATURE_MIN_SM: Dict[str, int] = {
    "SHARED": 70,     # __shared__ region (LDS/STS/LDL/STL)
    "TEXTURE": 70,    # bound texture object (TEX/TLD/TXQ)
    "SURFACE": 70,    # bound surface object (SULD/SUST)
    "TMA": 90,        # CUtensorMap + mbarrier (UTMALDG/STG/PF/REDG) — Hopper+
    "CLUSTER": 90,    # thread-block cluster (UCGABAR arrive/wait) — Hopper+
}

#: Named harness "flavors": feature-sets compiled from the one template. Each
#: is a specialized measurement harness; the union validates across the ones a
#: generation can build. ``super`` is the primary combinable campaign harness.
HARNESS_FLAVORS: Dict[str, tuple] = {
    "super": ("SHARED", "TEXTURE", "SURFACE"),
    "tma": ("SHARED", "TMA"),
    "cga": ("CLUSTER",),
}


def normalize_sm(sm) -> str:
    """Normalize an SM version to the canonical string form ("120")."""
    s = str(sm).lower().removeprefix("sm_").removeprefix("sm")
    return s


def gen_features(sm) -> set:
    """Harness features the given SM version supports."""
    n = int(normalize_sm(sm))
    return {feat for feat, min_sm in FEATURE_MIN_SM.items() if n >= min_sm}


def harness_flavors(sm) -> Dict[str, tuple]:
    """Buildable harness flavors for this SM — those whose every feature is
    supported. Feeds the per-generation set of union validation stages."""
    feats = gen_features(sm)
    return {name: fs for name, fs in HARNESS_FLAVORS.items()
            if set(fs) <= feats}


def arch_info(sm) -> Optional[ArchInfo]:
    return ARCHS.get(normalize_sm(sm))


def supported_sms() -> List[str]:
    """SM versions known to the registry (whether or not ISA JSON is present)."""
    return sorted(ARCHS.keys(), key=int)


def available_sms() -> List[str]:
    """SM versions with a pre-built ISA JSON available locally."""
    return paths.available_isa_sms()

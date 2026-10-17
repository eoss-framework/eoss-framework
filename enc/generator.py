#!/usr/bin/env python3
"""Constraint-aware SASS instruction encoding generator.

Creates valid instruction encodings from enriched ISA JSON definitions
with support for:
- Condition tree validation
- SIZE-based register allocation
- Latency-aware instruction scheduling
- Pipeline-based instruction generation
- Complete modifier/suffix combinations
- Configurable register values

Absorbed from OSIRIS/gpu_osiris_generator.py; bit packing comes from
``eoss.enc.bits``, scheduling enums from ``eoss.spec.data``, and the
cuobjdump invocation from ``eoss.binary.cuobjdump``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .bits import parse_int_literal, write_bit_segments
from ..binary.cuobjdump import cuobjdump_path
from ..spec import data as spec_data

IMM_PATTERN = re.compile(r"^(?P<bits>\d+)\s*/\s*(?P<value>0x[0-9a-fA-F]+|0b[01]+|\d+)$")


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration for instruction generation."""
    json_path: Path
    output_path: Path
    binary_output: Optional[Path] = None

    # Generation modes
    generate_all_variants: bool = False  # Generate all modifier combinations vs random
    random_class: bool = False  # Randomly choose one class per opcode
    encodings_per_class: int = 1  # Emit up to k distinct valid operand fills per class

    # Register configuration
    register_values: Dict[str, int] = field(default_factory=dict)  # e.g., {"R0": 0, "R1": 42}
    use_zero_registers: bool = True  # Prefer RZ/URZ when safe
    use_predicate_true: bool = True  # Prefer PT for predicates
    safe_dest_registers: Optional[Tuple[int, int]] = None  # Safe range for Rd (e.g., (24, 31) or None)
    safe_source_registers: Optional[Tuple[int, int]] = None  # Safe range for Ra, Rb, Rc (e.g., (0, 7) or None)
    all_rz: bool = False  # Force all GPRs to RZ (255) for zero side effects
    rz_gpr_only: bool = False  # Force GPRs to RZ but allow proper UR allocation (0-62)
    zero_immediates: bool = False  # Force all immediate values and non-register/predicate args to 0
    source_gpr: Optional[int] = None  # Specific GPR register for all source GPRs (e.g., 0 for R0 which holds ~1.0f)
    source_ugpr: Optional[int] = None  # Specific UGPR register for all source UGPRs
    immediate_value: Optional[int] = None  # Specific value for all immediates (overrides zero_immediates)

    # Latency and scheduling
    use_latency_info: bool = False  # Use real latencies from sm_xx_latencies.txt
    schedule_by_pipeline: bool = False  # Group by pipeline for realistic scheduling
    insert_nops: bool = False  # Insert NOPs for latency satisfaction

    # Validation and filtering
    validate_conditions: bool = True  # Only generate instructions that pass condition trees
    validate_register_sizes: bool = True  # Respect SIZE predicates for register allocation
    fail_on_invalid: bool = False  # Fail if any instruction is invalid
    filter_pipelines: Optional[Set[str]] = None  # Only generate instructions from these pipelines

    # Output options
    verbose: bool = False
    seed: Optional[int] = None
    max_instructions: Optional[int] = None  # Limit total instructions

    # Override options
    overrides: Dict[str, str] = field(default_factory=dict)
    dtype_overrides: Dict[str, str] = field(default_factory=dict)

    # Harness-aware operand policy: resource register map extracted from the
    # target harness (see eoss.runtime.regmap.RegisterMap). When set, texture/
    # surface operands target the handle/coordinate registers the harness
    # actually keeps live at the measure slot instead of hardcoded UR4/R4/R0.
    register_map: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class InstructionResult:
    """Result of generating a single instruction."""
    opcode_key: str
    instruction_name: str
    class_name: Optional[str]
    variant_id: Optional[str]  # e.g., "__RRR", "__RRI"

    argument_values: Dict[str, int]
    bytes_le: bytes

    # Validation results
    failed_conditions: List[str]
    is_valid: bool

    # Metadata
    pipeline: Optional[str] = None
    latency_cycles: Optional[int] = None
    delay: int = 0  # Stall count encoded in opex bits 105-108 (0-15 cycles)
    occupancy: Optional[int] = None
    register_allocations: Dict[str, List[int]] = field(default_factory=dict)
    static_valid: Optional[bool] = None  # None = not tested, True/False = cuobjdump result
    disasm_text: Optional[str] = None  # Disassembled mnemonic from cuobjdump


class ConditionEvaluationError(RuntimeError):
    """Raised when a condition tree cannot be evaluated."""


# ============================================================================
# Helper Functions
# ============================================================================

def requires_64bit_alignment(instruction_name: str) -> bool:
    """
    Determine if an instruction requires 64-bit (even-aligned) registers.

    64-bit operations include:
    - Double-precision FP: dadd, dfma, dmul, dmnmx, dsetp, dset
    - 64-bit integer/uniform: uiadd3_64, usel_64, umov_imm64, etc.
    - Wide multiply-add: uimad_wide, imad_wide (produces 64-bit result)
    - LEA instructions: uclea, ulepc (64-bit address computation)
    - Any instruction with "_64" or "imm64" in its name

    Returns True if the instruction uses 64-bit register pairs.
    """
    # Double-precision floating point operations (prefix 'd')
    fp64_prefixes = ['dadd', 'dfma', 'dmul', 'dmnmx', 'dsetp', 'dset', 'drcp', 'dsqrt']

    instr_lower = instruction_name.lower()
    for prefix in fp64_prefixes:
        if instr_lower.startswith(prefix):
            return True

    # 64-bit operations identified by name patterns
    if '_64' in instr_lower or 'imm64' in instr_lower:
        return True

    # Wide multiply-add (produces 64-bit result from 32-bit inputs)
    if 'imad_wide' in instr_lower:
        return True

    # Uniform LEA instructions (64-bit address computation)
    if instr_lower.startswith('uclea') or instr_lower.startswith('ulepc'):
        return True

    return False


def is_data_immediate(arg_name: str, alias: str) -> bool:
    """Check if an argument is a data operand immediate (not a control/modifier field).

    Returns True for source operand immediates (Sb, Sc, imm8, offsets, etc.)
    that represent data values the instruction operates on. Returns False for
    control fields (pm_pred, req_bit_set, e, sz, ftz, cop, etc.) that configure
    instruction behavior.
    """
    name_lower = arg_name.lower()
    if 'imm' in name_lower:
        return True
    if 'offset' in name_lower:
        return True
    if arg_name in ('Sb', 'Sc', 'Sa', 'Sb_absolute'):
        return True
    if 'constbank' in name_lower:
        return True
    if 'SCALE' in arg_name:
        return True
    return False


# ============================================================================
# Format Field Default Value Extraction
# ============================================================================

# Pattern to match /DEFAULT_SET("value"):fieldname in TEXT strings
DEFAULT_PATTERN_WITH_VALUE = re.compile(r'/([A-Za-z0-9_]+)\("([^"]*)"\):([a-zA-Z_][a-zA-Z0-9_.]*)')
# Pattern to match /DEFAULT_SET:fieldname (no default value specified)
DEFAULT_PATTERN_NO_VALUE = re.compile(r'/([A-Za-z0-9_]+):([a-zA-Z_][a-zA-Z0-9_.]*)')


def extract_field_defaults(
    text_field: str,
    defaults: Mapping[str, Any]
) -> Dict[str, int]:
    """Extract default values for format fields from TEXT format string.

    Parses patterns like:
    - /Round1("RN"):rnd - with explicit default value
    - /Float64:srcfmt - without explicit default (use first valid from DEFAULTS)
    - /DSTFMT_U8_S8_U16_S16_U32_S32("S32"):dstfmt

    Args:
        text_field: The TEXT format string from instruction definition
        defaults: The DEFAULTS section from ISA JSON

    Returns:
        Dict mapping field names to their default values
    """
    field_defaults: Dict[str, int] = {}

    # First, find matches with explicit default values: /DEFAULT_SET("value"):fieldname
    for match in DEFAULT_PATTERN_WITH_VALUE.finditer(text_field):
        default_set_name = match.group(1)  # e.g., "Round1"
        default_value_name = match.group(2)  # e.g., "RN"
        field_name = match.group(3)  # e.g., "rnd"

        # Look up the default set in DEFAULTS
        if default_set_name in defaults:
            default_set = defaults[default_set_name]
            if isinstance(default_set, dict):
                # Handle negated values (e.g., "noftz" -> look for "noftz" or "FTZ")
                if default_value_name in default_set:
                    value = default_set[default_value_name]
                    if value is not None:
                        field_defaults[field_name] = value
                # Try without "no" prefix
                elif default_value_name.startswith("no"):
                    # For "noftz", "nontz", etc., the value is typically 0 (disabled)
                    field_defaults[field_name] = 0

    # Second, find matches without default values: /DEFAULT_SET:fieldname
    # Use first valid (non-INVALID) value from the DEFAULT set
    for match in DEFAULT_PATTERN_NO_VALUE.finditer(text_field):
        default_set_name = match.group(1)
        field_name = match.group(2)

        # Skip if already found (with explicit value takes precedence)
        if field_name in field_defaults:
            continue

        # Skip if this pattern actually has a value (already processed by first pattern)
        # Check character immediately before the DEFAULT_SET name - should be '/'
        # If there's a ')' right before the ':', it's a pattern with value
        match_end = match.end()
        colon_pos = text_field.rfind(':', match.start(), match_end)
        if colon_pos > 0:
            char_before_colon = text_field[colon_pos - 1]
            if char_before_colon == ')':
                continue  # This is /SET("value"):field, skip it

        # Look up and use first valid value
        if default_set_name in defaults:
            default_set = defaults[default_set_name]
            if isinstance(default_set, dict):
                for name, value in default_set.items():
                    if "INVALID" not in name.upper() and value is not None:
                        field_defaults[field_name] = value
                        break

    return field_defaults


def get_first_valid_default(defaults: Mapping[str, Any], default_set_name: str) -> Optional[int]:
    """Get the first valid (non-INVALID) value from a DEFAULT set.

    Args:
        defaults: The DEFAULTS section from ISA JSON
        default_set_name: Name of the DEFAULT set to look up

    Returns:
        First valid integer value, or None if not found
    """
    if default_set_name not in defaults:
        return None

    default_set = defaults[default_set_name]
    if not isinstance(default_set, dict):
        return None

    # Prefer non-INVALID values
    for name, value in default_set.items():
        if "INVALID" not in name.upper() and value is not None:
            return value

    return None


# ============================================================================
# Enhanced Register Allocation
# ============================================================================

class RegisterAllocator:
    """Allocates registers based on SIZE predicates."""

    def __init__(self, config: GeneratorConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.used_registers: Set[int] = set()
        self.used_uniform_registers: Set[int] = set()

    def allocate_register(
        self,
        reg_type: str,  # "GPR" or "UGPR"
        size_bits: int,
        prefer_zero: bool = False
    ) -> int:
        """Allocate a register respecting size requirements.

        Args:
            reg_type: "GPR" for normal registers, "UGPR" for uniform
            size_bits: Required size in bits (32 = 1 reg, 64 = 2 consecutive regs)
            prefer_zero: Whether to prefer RZ/URZ

        Returns:
            Register number
        """
        # Calculate required register count
        reg_count = (size_bits + 31) // 32

        # Handle zero register preference
        if prefer_zero and self.config.use_zero_registers:
            if reg_type == "GPR":
                return 255  # RZ
            else:
                return 255  # URZ (uniform zero register, 8-bit encoding)

        # Get max register based on type
        if reg_type == "GPR":
            max_reg = 254
            used = self.used_registers
        else:
            max_reg = 79
            used = self.used_uniform_registers

        # For multi-register allocations, need consecutive registers
        if reg_count > 1:
            # IMPORTANT: 64-bit operations require even-numbered base registers!
            # SASS constraint: 64-bit pairs must be (R0,R1), (R2,R3), etc.
            # Try to find consecutive unused registers with even base
            for base in range(0, max_reg - reg_count + 1):
                # Force even alignment for 64-bit operations
                if base % 2 != 0:
                    continue
                if all(base + i not in used for i in range(reg_count)):
                    # Mark all as used
                    for i in range(reg_count):
                        used.add(base + i)
                    return base
            # Fallback: use any valid even-numbered base
            max_even_base = ((max_reg - reg_count + 1) // 2) * 2
            base = self.rng.choice(range(0, max_even_base + 1, 2))  # Only even numbers
            for i in range(reg_count):
                used.add(base + i)
            return base
        else:
            # Single register allocation
            if self.config.register_values:
                # Check if user specified a value
                reg_name = f"R{len(used)}" if reg_type == "GPR" else f"UR{len(used)}"
                if reg_name in self.config.register_values:
                    reg_num = list(self.config.register_values.keys()).index(reg_name)
                    used.add(reg_num)
                    return reg_num

            # Random allocation
            reg = self.rng.randint(0, max_reg)
            used.add(reg)
            return reg

    def allocate_predicate(self, prefer_true: bool = False) -> int:
        """Allocate a predicate register (P0-P6, PT=7)."""
        if self.config.use_predicate_true:
            return 7  # PT - Always execute
        return self.rng.randint(0, 7)


# ============================================================================
# Condition Tree Evaluation
# ============================================================================

def to_number(value: Any) -> int:
    """Convert any value to an integer."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return parse_int_literal(value)
        except ValueError:
            return 1 if value else 0
    return int(value)


def to_boolean(value: Any) -> bool:
    """Convert any value to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def evaluate_condition_tree(
    node: Mapping[str, Any],
    values: Dict[str, int],
    treat_implication: bool = False
) -> int:
    """Evaluate a condition tree node.

    Args:
        node: Condition tree node
        values: Current argument values
        treat_implication: Whether to treat && as implication

    Returns:
        Integer result of evaluation
    """
    node_type = node.get("type")

    if node_type == "literal":
        return to_number(node.get("value"))

    if node_type in ("constant", "parameter", "enum"):
        return to_number(node.get("value"))

    if node_type == "argument":
        name = node.get("name")
        if name in values:
            return values[name]
        raise ConditionEvaluationError(f"Unknown argument '{name}'")

    if node_type == "identifier":
        # Try to resolve from values
        name = str(node.get("name", ""))
        if name in values:
            return values[name]
        # Check if this is an unresolved enum (contains @)
        # We can't evaluate conditions with unresolved enums - skip them
        if "@" in name:
            raise ConditionEvaluationError(
                f"Cannot evaluate unresolved enum identifier: {name}"
            )
        # Otherwise use the value if already resolved
        return to_number(node.get("value", 0))

    if node_type == "defined":
        # DEFINED/!DEFINED table lookup conditions cannot be evaluated statically.
        # These check if a combination of argument values exists in a lookup table.
        # Since we don't have the table data at generation time, skip evaluation.
        raise ConditionEvaluationError(
            f"Cannot evaluate DEFINED table lookup: {node.get('table_name', 'unknown')}"
        )

    if node_type == "binary":
        op = node.get("op")
        left = evaluate_condition_tree(node.get("left", {}), values, treat_implication)
        right = evaluate_condition_tree(node.get("right", {}), values, treat_implication)

        if op == "&&":
            if treat_implication:
                return int((not to_boolean(left)) or to_boolean(right))
            return int(to_boolean(left) and to_boolean(right))
        if op == "||":
            return int(to_boolean(left) or to_boolean(right))
        if op == "==":
            return int(to_number(left) == to_number(right))
        if op == "!=":
            return int(to_number(left) != to_number(right))
        if op == "<=":
            return int(to_number(left) <= to_number(right))
        if op == ">=":
            return int(to_number(left) >= to_number(right))
        if op == "<":
            return int(to_number(left) < to_number(right))
        if op == ">":
            return int(to_number(left) > to_number(right))
        if op == "+":
            return to_number(left) + to_number(right)
        if op == "-":
            return to_number(left) - to_number(right)
        if op == "*":
            return to_number(left) * to_number(right)
        if op == "%":
            right_num = to_number(right)
            return to_number(left) % right_num if right_num else 0
        if op == "&":
            return to_number(left) & to_number(right)
        if op == "|":
            return to_number(left) | to_number(right)
        if op == "<<":
            return to_number(left) << to_number(right)
        if op == ">>":
            return to_number(left) >> to_number(right)
        raise ConditionEvaluationError(f"Unsupported operator '{op}'")

    if node_type == "unary":
        op = node.get("op")
        operand = evaluate_condition_tree(node.get("operand", {}), values, treat_implication)
        if op == "!":
            return int(not to_boolean(operand))
        if op == "-":
            return -to_number(operand)
        raise ConditionEvaluationError(f"Unsupported unary operator '{op}'")

    raise ConditionEvaluationError(f"Unsupported node type '{node_type}'")


def evaluate_top_level_implication(
    tree: Mapping[str, Any],
    values: Dict[str, int]
) -> int:
    """Evaluate a tree where the top-level && represents an implication (->).

    For expressions like: (A && B && C) -> D
    which get parsed as: (A && B && C) && D
    We need to treat ONLY the top-level && as implication (NOT left OR right),
    while nested && in the antecedent are evaluated as normal AND.
    """
    if tree.get("type") != "binary" or tree.get("op") != "&&":
        # Not a top-level && - shouldn't happen but evaluate normally
        return evaluate_condition_tree(tree, values, treat_implication=False)

    # Evaluate left (antecedent) and right (consequent) separately
    # The left side should be evaluated normally (nested && are real ANDs)
    left = evaluate_condition_tree(tree.get("left", {}), values, treat_implication=False)
    right = evaluate_condition_tree(tree.get("right", {}), values, treat_implication=False)

    # Apply implication semantics: NOT(left) OR right
    return int((not to_boolean(left)) or to_boolean(right))


def validate_conditions(
    conditions: Sequence[Mapping[str, Any]],
    values: Dict[str, int]
) -> List[str]:
    """Validate all conditions for an instruction.

    Returns:
        List of failed condition names
    """
    failed: List[str] = []

    for condition in conditions:
        if not isinstance(condition, Mapping):
            continue

        # Skip conditions with parsing errors
        if condition.get("error") or not condition.get("tree"):
            continue

        # Skip malformed conditions where FORMAT line was parsed as a condition name
        cond_name = str(condition.get("name", ""))
        if cond_name.startswith("FORMAT "):
            continue

        expression = str(condition.get("expression", ""))
        has_implication = "->" in expression

        try:
            if has_implication:
                # Expression has implication - the parser converts "->" to "&&"
                # so the top-level "&&" represents the implication operator
                # Use special evaluation that treats only top-level && as implication
                result = evaluate_top_level_implication(
                    condition["tree"],
                    values
                )
            else:
                # No implication - evaluate normally
                result = evaluate_condition_tree(
                    condition["tree"],
                    values,
                    treat_implication=False
                )
            if not to_boolean(result):
                name = condition.get("name") or "UNKNOWN_CONDITION"
                failed.append(str(name))
        except ConditionEvaluationError:
            # Can't evaluate - skip
            continue

    return failed


def analyze_and_fix_conditions(
    conditions: Sequence[Mapping[str, Any]],
    values: Dict[str, int],
    failed_names: List[str]
) -> Dict[str, int]:
    """Analyze failed conditions and try to fix register/field values.

    Returns:
        Dictionary of suggested fixes {arg_name: new_value}
    """
    fixes: Dict[str, int] = {}

    for condition in conditions:
        if not isinstance(condition, Mapping):
            continue

        name = condition.get("name", "")
        if name not in failed_names:
            continue

        expression = str(condition.get("expression", ""))
        tree = condition.get("tree", {})

        # Handle MISALIGNED_REG_ERROR: need even-aligned registers
        if name == "MISALIGNED_REG_ERROR":
            for reg in ["Rd", "Ra", "Rb", "Rc", "Re", "URd", "URa", "URb", "URc", "URe"]:
                if reg in expression and reg in values:
                    current = values[reg]
                    if current != 255 and current % 2 != 0:
                        fixes[reg] = current - 1 if current > 0 else 0
                    if ("96" in expression or "128" in expression) and current != 255:
                        if current % 4 != 0:
                            fixes[reg] = (current // 4) * 4

        # Handle OOR_REG_ERROR: register out of valid range
        elif name == "OOR_REG_ERROR":
            for reg in ["Rd", "Ra", "Rb", "Rc", "Re"]:
                if reg in expression and reg in values:
                    current = values[reg]
                    if current != 255 and current > 253:
                        fixes[reg] = 252
            for ureg in ["URa", "URb", "URc", "URd", "URe"]:
                if ureg in expression and ureg in values:
                    current = values[ureg]
                    if current != 255 and current > 62:
                        fixes[ureg] = 31

        # Handle ILLEGAL_INSTR_ENCODING_ERROR and ILLEGAL_INSTR_ENCODING_SASS_ONLY_ERROR
        elif name in ("ILLEGAL_INSTR_ENCODING_ERROR", "ILLEGAL_INSTR_ENCODING_SASS_ONLY_ERROR"):
            import re

            # Check for "field != Register@RZ" - register must NOT be RZ
            rz_neq = r"\((\w+)\)!=`(?:Register|ZeroRegister)@RZ\)"
            for match in re.finditer(rz_neq, expression):
                field_name = match.group(1)
                if field_name in values and values[field_name] == 255:
                    fixes[field_name] = 0  # Use R0 instead of RZ

            # Check for "field != UniformRegister@URZ" - register must NOT be URZ
            urz_neq = r"\((\w+)\)!=`(?:UniformRegister|ZeroUniformRegister)@URZ\)"
            for match in re.finditer(urz_neq, expression):
                field_name = match.group(1)
                if field_name in values and values[field_name] == 255:
                    fixes[field_name] = 0  # Use UR0 instead of URZ

            # Note: "((field)==Register@RZ)" in an ILLEGAL condition means
            # the expression is TRUE when encoding is valid — i.e., field MUST
            # be RZ. Do NOT change RZ to R0 here; only the "!=" pattern above
            # means "RZ is forbidden."

            # Check for simple equality patterns: (field == 0xN) or (field == N)
            eq_hex = r"\((\w+)\s*==\s*0x([0-9a-fA-F]+)\)"
            for match in re.finditer(eq_hex, expression):
                field_name = match.group(1)
                req_val = int(match.group(2), 16)
                if field_name in values and values[field_name] != req_val:
                    fixes[field_name] = req_val

            # Check for standalone decimal equality: (field == N)
            # Only apply for expressions that are JUST this equality (not part of ||/-> chains)
            eq_standalone = re.fullmatch(r"\s*\(+(\w+)\s*==\s*(\d+)\)+\s*", expression)
            if eq_standalone:
                field_name = eq_standalone.group(1)
                req_val = int(eq_standalone.group(2))
                if field_name in values and values[field_name] != req_val:
                    fixes[field_name] = req_val

            # Check for implication patterns: (condition) -> (requirement)
            # Example: (((depth == `DEPTH_cctl@SHALLOW))) -> (((dst_wr_sb == 7)))
            # Handle specific known patterns
            if "depth ==" in expression and "dst_wr_sb ==" in expression and "->" in expression:
                # Pattern: if depth is 0 (SHALLOW), dst_wr_sb must be 7
                if "depth" in values and values["depth"] == 0:
                    if "dst_wr_sb" in values and values["dst_wr_sb"] != 7:
                        fixes["dst_wr_sb"] = 7

            # Generic implication: (field == val) -> (other_field == req_val)
            impl_match = re.search(r"\(+(\w+)\s*==.*?(\d+)\)+\s*->\s*\(+(\w+)\s*==\s*(\d+)\)+", expression)
            if impl_match:
                cond_field = impl_match.group(1)
                req_field = impl_match.group(3)
                req_val = int(impl_match.group(4))
                # If condition field is 0 (common default), apply requirement
                if cond_field in values and values.get(cond_field, -1) == 0:
                    if req_field in values and values[req_field] != req_val:
                        fixes[req_field] = req_val

            # Implication with RZ/URZ consequent:
            # Pattern: (word_mask==N||...) -> ((field)==Register@RZ) or UniformRegister@URZ
            # When the antecedent matches (field values are in the set), require RZ/URZ
            rz_impl = re.search(
                r"->\s*\(+\((\w+)\)==`(?:Register|ZeroRegister)@RZ\)+",
                expression
            )
            if rz_impl:
                reg_field = rz_impl.group(1)
                # Check if any antecedent value matches
                ante_values = set(int(m.group(1)) for m in re.finditer(r"==\s*(\d+)", expression[:expression.index("->")]))
                ante_field_m = re.search(r"\((\w+)==", expression)
                if ante_field_m:
                    ante_field = ante_field_m.group(1)
                    if ante_field in values and values[ante_field] in ante_values:
                        if reg_field in values and values[reg_field] != 255:
                            fixes[reg_field] = 255  # Set to RZ

            urz_impl = re.search(
                r"->\s*\(+\((\w+)\)==`(?:UniformRegister|ZeroUniformRegister)@URZ\)+",
                expression
            )
            if urz_impl:
                reg_field = urz_impl.group(1)
                ante_values = set(int(m.group(1)) for m in re.finditer(r"==\s*(\d+)", expression[:expression.index("->")]))
                ante_field_m = re.search(r"\((\w+)==", expression)
                if ante_field_m:
                    ante_field = ante_field_m.group(1)
                    if ante_field in values and values[ante_field] in ante_values:
                        if reg_field in values and values[reg_field] != 255:
                            fixes[reg_field] = 255  # Set to URZ

            # Check for invalid enum values: (field != `ENUM@INVALID_X)
            # Collect ALL invalid values for each field first, then choose a valid one
            invalid_vals_per_field: Dict[str, set] = {}
            invalid_pattern = r"\((\w+)\s*!=\s*`\w+@INVALID(\d+)\)"
            for match in re.finditer(invalid_pattern, expression):
                field_name = match.group(1)
                invalid_val = int(match.group(2))
                if field_name not in invalid_vals_per_field:
                    invalid_vals_per_field[field_name] = set()
                invalid_vals_per_field[field_name].add(invalid_val)

            for field_name, invalid_set in invalid_vals_per_field.items():
                if field_name in values and values[field_name] in invalid_set:
                    # Find first valid value (not in invalid set)
                    for candidate in range(256):
                        if candidate not in invalid_set:
                            fixes[field_name] = candidate
                            break

            # Check for simple != patterns: (field != N)
            # Also collect all forbidden values first
            forbidden_per_field: Dict[str, set] = {}
            simple_neq = r"\((\w+)\s*!=\s*(\d+)\)"
            for match in re.finditer(simple_neq, expression):
                field_name = match.group(1)
                forbidden_val = int(match.group(2))
                if field_name not in forbidden_per_field:
                    forbidden_per_field[field_name] = set()
                forbidden_per_field[field_name].add(forbidden_val)

            for field_name, forbidden_set in forbidden_per_field.items():
                if field_name in invalid_vals_per_field:
                    continue  # Already handled by INVALID pattern
                if field_name in values and values[field_name] in forbidden_set:
                    for candidate in range(256):
                        if candidate not in forbidden_set:
                            fixes[field_name] = candidate
                            break

    return fixes


# ============================================================================
# Instruction Assembly
# ============================================================================

def is_memory_instruction(instruction_data: Mapping[str, Any]) -> bool:
    """Check if an instruction is a memory instruction that needs valid addresses.

    Memory instructions are identified by MEM_SCBD_TYPE = "MEM_INST" in the
    PROPERTIES section. Note: many non-memory instructions (FRND, F2I, etc.)
    share the mio_pipe pipeline, so pipeline alone is not a reliable indicator.

    Returns:
        True if this is a memory instruction requiring valid address registers
    """
    properties = instruction_data.get("PROPERTIES", {})
    if isinstance(properties, Mapping):
        mem_scbd_type = properties.get("MEM_SCBD_TYPE", "")
        if mem_scbd_type == "MEM_INST":
            return True

    return False


def is_ldcu_instruction(instruction_data: Mapping[str, Any]) -> bool:
    """Check if instruction is LDCU (Load Constant Uniform).

    LDCU instructions cannot use URZ (63) as destination - they need
    a valid uniform register (UR0-UR31) to write to.

    Returns:
        True if this is an LDCU instruction
    """
    instruction_name = instruction_data.get("INSTRUCTION", "")
    if isinstance(instruction_name, str):
        name_lower = instruction_name.lower()
        if "ldcu" in name_lower or "uldc" in name_lower:
            return True

    # Also check pipeline - OP_LDCU is dedicated pipeline
    latency_info = instruction_data.get("LATENCY_INFO", {})
    if isinstance(latency_info, Mapping):
        pipeline = latency_info.get("pipeline", "")
        if pipeline == "OP_LDCU":
            return True

    return False


def is_udp_instruction(instruction_data: Mapping[str, Any]) -> bool:
    """Check if instruction is in UDP (Uniform Datapath) pipeline.

    UDP instructions work on uniform registers and need valid UR0-UR31,
    not URZ (63) for destinations.

    Returns:
        True if this is a UDP pipeline instruction
    """
    latency_info = instruction_data.get("LATENCY_INFO", {})
    if isinstance(latency_info, Mapping):
        pipeline = latency_info.get("pipeline", "")
        if pipeline == "udp_pipe":
            return True
    return False


def is_texture_instruction(instruction_data: Mapping[str, Any]) -> bool:
    """Check if instruction is a texture instruction (TEX, TLD4, TXQ).

    Texture instructions require:
    - A valid texture object handle in URe (uniform register)
    - Coordinates in Ra/Rb (GPR)
    - Destination registers in Rd/Ra_URc (GPR)

    Returns:
        True if this is a texture instruction
    """
    instruction_name = instruction_data.get("INSTRUCTION", "")
    if isinstance(instruction_name, str):
        name_lower = instruction_name.lower()
        if any(x in name_lower for x in ["tex_", "tld4_", "tld_", "txq_", "txd_", "tmml_"]):
            return True
    return False


def is_surface_instruction(instruction_data: Mapping[str, Any]) -> bool:
    """Check if instruction is a surface instruction (SULD, SUST, SURED, SUQ).

    Surface instructions require:
    - A valid surface object handle in URe (uniform register)
    - Coordinates in Ra/Rb (GPR)
    - Data registers in Rd/Ra_URc (GPR)

    Returns:
        True if this is a surface instruction
    """
    instruction_name = instruction_data.get("INSTRUCTION", "")
    if isinstance(instruction_name, str):
        name_lower = instruction_name.lower()
        if any(x in name_lower for x in ["suld_", "sust_", "sured_", "suq_", "suatom_", "suquery_", "footprint_"]):
            return True
    return False


def is_tma_instruction(instruction_data: Mapping[str, Any]) -> bool:
    """Check if instruction is a Tensor Memory Accelerator descriptor op
    (UTMALDG/UTMASTG/UTMAPF/UTMAREDG): a global<->shared bulk copy driven by a
    CUtensorMap descriptor in a uniform register, with the shared destination
    in another uniform register (see the tma flavor of harnesses/osiris_harness.cu).
    """
    name = str(instruction_data.get("INSTRUCTION", "")).lower()
    return name.startswith(("utmaldg", "utmastg", "utmapf", "utmaredg"))


def get_address_register_args(instruction_data: Mapping[str, Any]) -> Set[str]:
    """Get argument names that should use address registers (R2 for 64-bit addresses).

    Looks at PREDICATES section for ISRC_A_SIZE = 64 and similar patterns.
    Handles both literal 64 values and expressions that can evaluate to 64.

    Returns:
        Set of argument names that need 64-bit address registers
    """
    address_args: Set[str] = set()
    predicates = instruction_data.get("PREDICATES", {})

    if not isinstance(predicates, Mapping):
        return address_args

    # Check for 64-bit address size indicators
    # Pattern: ISRC_A_SIZE = 64 means the 'a' register (Ra) needs 64-bit address
    # Also match expressions like "32 + ((e==`E@E))*32" which can be 64
    for key, value in predicates.items():
        is_64bit = False

        if isinstance(value, int) and value == 64:
            is_64bit = True
        elif isinstance(value, str):
            # Check for expressions that can be 64-bit
            # e.g., "32 + ((e==`E@E))*32" can be 64 when e is set
            # For safety, treat any SIZE expression containing "+32" or "64" as potentially 64-bit
            if "64" in value or ("+32" in value or "* 32" in value or "*32" in value):
                is_64bit = True

        if is_64bit:
            # Extract the register letter from the key
            # e.g., ISRC_A_SIZE -> 'Ra', ILABEL_Ra_SIZE -> 'Ra'
            if "ISRC_A" in key or "ILABEL_Ra" in key:
                address_args.add("Ra")
            if "ISRC_B" in key or "ILABEL_Rb" in key:
                address_args.add("Rb")
            if "ISRC_C" in key or "ILABEL_Rc" in key:
                address_args.add("Rc")

    return address_args


def assemble_instruction(
    opcode_key: str,
    instruction_data: Mapping[str, Any],
    config: GeneratorConfig,
    allocator: RegisterAllocator,
    class_name: Optional[str] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    tables: Optional[Mapping[str, Any]] = None
) -> InstructionResult:
    """Assemble a single instruction variant.

    Args:
        opcode_key: Opcode identifier
        instruction_data: Instruction metadata from JSON
        defaults: The DEFAULTS section from ISA JSON for format field lookup
        config: Generator configuration
        allocator: Register allocator
        class_name: Optional instruction class/variant name

    Returns:
        InstructionResult with assembled bytes and metadata
    """
    # Detect 64-bit instructions that require even-aligned registers
    instruction_name = class_name or instruction_data.get("INSTRUCTION", opcode_key)
    if isinstance(instruction_name, dict):
        # Extract first variant name from dict
        instruction_name = next(iter(instruction_name.keys()), opcode_key)
    # Check both class name and full instruction name for 64-bit detection
    full_instr_name = str(instruction_data.get("INSTRUCTION", ""))
    is_64bit_instr = (requires_64bit_alignment(str(instruction_name)) or
                      requires_64bit_alignment(full_instr_name))

    # Detect memory instructions that need valid address registers
    is_mem_instr = is_memory_instruction(instruction_data)
    address_reg_args = get_address_register_args(instruction_data) if is_mem_instr else set()
    # Global atomics (ATOMG/REDG and generic ATOM/RED with a 64-bit address)
    # need an *aligned* base — R2 (the kernel's live LDG address) may not be
    # 8/16-byte aligned. Target the harness's aligned atomic scratch instead.
    _atom_fam = (full_instr_name or "").lower().split("_")[0]
    is_atomic_instr = (is_mem_instr and bool(address_reg_args)
                       and _atom_fam in ("atomg", "redg", "atom", "red", "atomcas"))

    # Detect LDCU and UDP instructions - these need valid uniform register destinations
    # URZ (63) cannot be a destination for these instructions
    is_ldcu = is_ldcu_instruction(instruction_data)
    is_udp = is_udp_instruction(instruction_data)
    needs_valid_ur_dest = is_ldcu or is_udp

    # Detect texture/surface instructions - these need valid texture/surface object handles
    # in uniform registers (URe) and proper coordinate/destination registers
    is_tex_instr = is_texture_instruction(instruction_data)
    is_surf_instr = is_surface_instruction(instruction_data)
    # Detect TMA descriptor ops — need the CUtensorMap descriptor UR and a
    # shared-destination UR the harness keeps live (config.register_map).
    is_tma_instr = is_tma_instruction(instruction_data)

    arguments = instruction_data.get("ARGUMENTS", {})
    predicates = instruction_data.get("PREDICATES", {})
    conditions = instruction_data.get("CONDITIONS", [])
    latency_info = instruction_data.get("LATENCY_INFO", {})

    # Extract default values for format fields from TEXT field
    # TEXT can be at top level or inside FORMAT object
    field_defaults: Dict[str, int] = {}
    if defaults:
        text_field = instruction_data.get("TEXT", "")
        if not text_field:
            format_data = instruction_data.get("FORMAT", {})
            text_field = format_data.get("TEXT", "")
        if text_field:
            field_defaults = extract_field_defaults(text_field, defaults)

    # Also apply register defaults from FORMAT.DEFAULTS (e.g., Ra: "RZ", Rb: "RZ")
    format_data = instruction_data.get("FORMAT", {})
    fmt_defaults = format_data.get("DEFAULTS", {})
    for fld_name, fld_default in fmt_defaults.items():
        if fld_name in field_defaults:
            continue  # Already set by TEXT pattern extraction
        if isinstance(fld_default, str):
            # Translate register names to numeric values
            if fld_default == "RZ":
                field_defaults[fld_name] = 255  # Register@RZ
            elif fld_default == "URZ":
                field_defaults[fld_name] = 255  # UniformRegister@URZ
            elif fld_default == "PT":
                field_defaults[fld_name] = 7  # Predicate True
            elif fld_default == "UPT":
                field_defaults[fld_name] = 7  # Uniform Predicate True
            elif fld_default == "noe":
                # E/noe flag: default "noe"=0, but most memory instructions need E=1
                # to function correctly. Override to 1 for safety.
                field_defaults[fld_name] = 1
            elif fld_name in ("usched_info", "batch_t"):
                # Scheduling field: store symbolic name for opex table lookup.
                # Values like '"W1"/PRINT' → extract "W1", plain "DRAIN" → "DRAIN"
                val = fld_default.strip('"').split('"')[0].split("/")[0].strip()
                field_defaults[fld_name] = val
            elif defaults:
                # Try resolving via DEFAULTS enum lookup using FORMAT.FIELDS
                # e.g., pm_pred: "PMN" → look up PM_PRED enum → PMN=0
                for ff in format_data.get("FIELDS", []):
                    if ff.get("target") == fld_name:
                        enum_name = ff.get("source", "").lstrip("/")
                        enum_set = defaults.get(enum_name, {})
                        if isinstance(enum_set, dict) and fld_default in enum_set:
                            resolved = enum_set[fld_default]
                            if isinstance(resolved, int):
                                field_defaults[fld_name] = resolved
                        break
        elif isinstance(fld_default, int):
            field_defaults[fld_name] = fld_default

    # Build enum resolution map from FORMAT.FIELDS
    # For fields like /BarSYNCALL:barmode, resolve the enum value
    # so that *barmode aliases can find the correct value
    enums: Dict[str, int] = {}
    fmt_fields = format_data.get("FIELDS", [])
    if isinstance(fmt_fields, list):
        for fmt_field in fmt_fields:
            if not isinstance(fmt_field, dict):
                continue
            source = fmt_field.get("source", "")
            target = fmt_field.get("target", "")
            if source.startswith("/") and target:
                # This is an enum field like /BarSYNCALL:barmode
                enum_name = source[1:]  # Remove leading /
                # Look up the enum value from the DEFAULTS section
                if target in fmt_defaults:
                    fld_val = fmt_defaults[target]
                    if isinstance(fld_val, str):
                        # Resolve string enum value from global defaults
                        if fld_val in defaults:
                            resolved = defaults[fld_val]
                            if isinstance(resolved, int):
                                enums[target] = resolved
                    elif isinstance(fld_val, int):
                        enums[target] = fld_val
                elif defaults:
                    # No explicit default in FORMAT — resolve from enum definition
                    # in DEFAULTS. The enum is a dict like {'SYNC': 0, ...}
                    enum_def = defaults.get(enum_name)
                    if isinstance(enum_def, dict) and len(enum_def) >= 1:
                        # Use the first (or only) enum value
                        first_val = next(iter(enum_def.values()))
                        if isinstance(first_val, int):
                            # With --all-rz on UDP instructions, certain enum
                            # values crash with zero operands. Override to safe
                            # defaults for UF2FP 8-bit conversion variants:
                            # - UNPACK/PACK+MERGE merge modes → 0 (passthrough)
                            # - Non-F32 source formats → 2 (F32)
                            # Only for URURUR/URIUR variants (URURI has different
                            # encoding constraints where merge=0 is invalid).
                            safe_val = first_val
                            is_ururi = "URURI" in opcode_key and "URURUR" not in opcode_key
                            if config.all_rz and is_udp and not is_ururi:
                                eu = enum_name.upper()
                                if len(enum_def) == 1 and ("UNPACK" in eu or "PACK_AB_MERGE" in eu):
                                    safe_val = 0
                                if target == "srcfmt" and first_val != 2:
                                    if "SRCFMT" in eu or "F16ONLY" in eu:
                                        safe_val = 2
                            enums[target] = safe_val
                            field_defaults[target] = safe_val
                    elif isinstance(enum_def, int):
                        enums[target] = enum_def
                        field_defaults[target] = enum_def

    # Get opcode value
    if class_name and "OPCODES" in instruction_data:
        opcodes = instruction_data["OPCODES"]
        if class_name in opcodes:
            opcode_val = parse_int_literal(str(opcodes[class_name]))
        else:
            opcode_val = parse_int_literal(instruction_data.get("OPCODE", opcode_key))
    else:
        opcode_val = parse_int_literal(instruction_data.get("OPCODE", opcode_key))

    # Compute instruction delay (stall count) from latency data
    # Prefer gpr_self_latency (self-pipeline diagonal from GPR TABLE_TRUE matrix)
    # which matches nak's self-RAW computation. Fall back to TABLE_TRUE[0] (first
    # column = ALU reader latency), then max(TABLE_TRUE), then default=1.
    computed_delay = 1  # Default: minimum safe stall (1 cycle)
    if isinstance(latency_info, Mapping):
        gpr_self = latency_info.get("gpr_self_latency")
        if gpr_self is not None:
            computed_delay = min(int(gpr_self), 15)  # Cap at 15 (4-bit field)
        else:
            table_true = latency_info.get("latencies", {}).get("TABLE_TRUE", [])
            if table_true:
                computed_delay = min(table_true[0], 15)  # First column as fallback

    # Prepare 16-byte instruction buffer (128-bit)
    buffer = bytearray(16)
    values: Dict[str, int] = {}
    register_allocs: Dict[str, List[int]] = {}

    # Get SIZE correlations for register allocation
    size_correlations = predicates.get("_SIZE_CORRELATIONS", {})

    # Process each argument
    for arg_name, arg_info in arguments.items():
        if not isinstance(arg_info, Mapping):
            continue

        bit_spec = arg_info.get("bits")
        width = arg_info.get("width")
        alias = str(arg_info.get("alias", ""))

        # Determine value for this argument
        value: int

        # Priority 1: Hardcoded values (alias starts with * followed by a number)
        # e.g., *255 means RZ, *0 means zero, *7 means PT
        # These MUST be checked first, before any register allocation logic
        alias_stripped = alias.lstrip("*") if alias.startswith("*") else ""
        if alias.startswith("*") and alias_stripped.replace("-","").isdigit():
            value = int(alias_stripped)
        elif alias.replace("-","").isdigit() and not alias.startswith("TABLES"):
            # Plain numeric alias without * (e.g., alias='0' for ftz=0, alias='7' for dst_wr_sb=7)
            value = int(alias)
        elif arg_name == "opcode":
            # Always use the opcode value
            value = opcode_val
        elif arg_name.endswith("_not"):
            # Predicate negation flags should be 0 (not negated) for always-execute
            value = 0
        elif alias.startswith("TABLES_opex"):
            # Encode scheduling via TABLES_opex_N(batch_t, usched_info) lookup.
            #
            # The opex field (8 bits) is split across bits 124:122 (hi) and 109:105 (lo).
            # It encodes stall cycles, yield hints, and batch scheduling.
            #
            # The value is looked up from the opex table using:
            #   batch_t: batch scheduling mode (default "NOP"=0)
            #   usched_info: stall/wait configuration (default varies per instruction)
            #
            # USCHED_INFO values: DRAIN=0, W1EG=1..W15EG=15, W1=17..W11=27
            # BATCH_T values: NOP=0, BATCH_START=1, BATCH_START_TILE=2, BATCH_END=4, BARRIER_EXEMPT=5
            #
            # For standalone execution safety, we default to:
            #   batch_t=NOP(0), usched_info=DRAIN(0) → opex=0 (minimum stall, no batch)
            # Combined with dst_wr_sb=7, src_rel_sb=7 (no barriers) this avoids
            # the scoreboard deadlock bug discovered in TTU experiments (Phase 22).

            # Extract table name to find instruction's default usched_info
            table_name = alias.split("(")[0].strip()

            # Resolve defaults from FORMAT fields
            usched_info_default = "DRAIN"
            batch_t_default = "NOP"
            if field_defaults:
                usched_info_default = field_defaults.get("usched_info", "DRAIN")
                batch_t_default = field_defaults.get("batch_t", "NOP")

            # Map symbolic names to numeric values (canonical tables)
            usched_val = spec_data.usched_info().get(usched_info_default, 0)
            batch_val = spec_data.batch_t().get(batch_t_default, 0)

            # Compute opex value from batch_t and usched_info
            # The table formula is: opex = batch_val * 32 + usched_val
            # (This matches TABLES_opex_N for all valid (batch_t, usched_info) pairs)
            value = (batch_val * 32) + usched_val
        elif alias.startswith("TABLES_"):
            # General table lookup (TABLES_mem_N, TABLES_Pnz_0, etc.)
            # Resolve input arguments from field_defaults, look up in table
            table_name = alias.split("(")[0].strip()
            table_data = tables.get(table_name, []) if tables else []
            # Extract argument names from alias: TABLES_X(a,b,c) -> [a, b, c]
            args_match = alias.split("(", 1)
            if len(args_match) > 1:
                arg_names = [a.strip().split("@")[0] for a in args_match[1].rstrip(")").split(",")]
            else:
                arg_names = []
            # Resolve each argument to its default numeric value
            input_vals = []
            for aname in arg_names:
                if aname in field_defaults and isinstance(field_defaults[aname], int):
                    input_vals.append(field_defaults[aname])
                elif aname in enums:
                    input_vals.append(enums[aname])
                elif aname in values:
                    input_vals.append(values[aname])
                else:
                    input_vals.append(0)
            # Look up in table: find entry where inputs match
            value = 0
            input_strs = [str(v) for v in input_vals]
            for entry in table_data:
                if isinstance(entry, dict) and entry.get("inputs") == input_strs:
                    value = entry["output"]
                    break
        elif is_tma_instr and config.register_map:
            # TMA descriptor op: descriptor address in Ra (UR), shared dest in
            # Rb (UR) — the registers the tma harness keeps live at the slot.
            rmap = config.register_map
            _desc = rmap.get("tma_desc_ur")
            _shared = rmap.get("tma_shared_ur")
            alias_clean = alias.split("(")[0].split("@")[0].strip().lstrip("*") if alias else ""
            if (arg_name == "Ra" or alias_clean == "Ra") and _desc is not None:
                value = _desc
            elif (arg_name == "Rb" or alias_clean == "Rb") and _shared is not None:
                value = _shared
            elif arg_name == "Ra_URc" and _shared is not None:
                value = _shared
            else:
                alias_field = alias.lstrip("*").split("(")[0].split("@")[0].strip()
                if arg_name in field_defaults:
                    value = field_defaults[arg_name]
                elif alias_field and alias_field in field_defaults:
                    value = field_defaults[alias_field]
                elif alias.startswith("*") and alias_field in enums:
                    value = enums[alias_field]
                elif arg_name.startswith("P") or "pred" in alias.lower():
                    value = 7  # (U)PT
                else:
                    value = 0
        elif is_tex_instr or is_surf_instr:
            # TEXTURE/SURFACE INSTRUCTION: Requires special register setup
            # Must check BEFORE is_mem_instr because TEX/SURF are in mio_pipe
            alias_clean = alias.split("(")[0].split("@")[0].strip().lstrip("*") if alias else ""

            # Harness-aware register map (eoss.runtime.regmap): target the
            # handle/coordinate registers the harness parks live at the measure
            # slot. Falls back to the historical hardcoded values when absent.
            rmap = config.register_map or {}
            _hkey = "surf_handle_ur" if is_surf_instr else "tex_handle_ur"
            _ckey = "surf_coord_regs" if is_surf_instr else "tex_coord_regs"
            _handle = rmap.get(_hkey)
            _handle = _handle if _handle is not None else 4
            _coords = rmap.get(_ckey) or []
            _coord = _coords[0] if _coords else None
            # Pre-Hopper bindless texture: the handle is a GPR (Rb) and coords a
            # GPR (Ra), with the SCR flag set — not a uniform-register handle.
            _gpr_handle = rmap.get("tex_handle_reg")
            _gpr_coord = rmap.get("tex_coord_reg")
            _gpr_mode = (not is_surf_instr) and _gpr_handle is not None

            if arg_name == "URe" or alias_clean == "URe":
                # Texture/Surface object handle (UR)
                value = _handle
            elif arg_name == "Rd" or alias_clean == "Rd":
                # Destination — RZ (discard) when a map is set, matching the
                # harness's own probe (TEX RZ, ...); else legacy R8.
                value = 255 if config.register_map else 8
            elif arg_name == "Ra_URc" or alias == "Rd2":
                # Additional destination - use R12 (consecutive quad)
                value = 12
            elif arg_name == "scr" and _gpr_mode:
                value = 1  # SCR: bindless texture handle in a register
            elif arg_name == "Ra" or alias_clean == "Ra":
                # Coordinate — the harness's live coord reg (GPR-mode uses the
                # pre-Hopper coord GPR); else R4.
                value = (_gpr_coord if _gpr_mode and _gpr_coord is not None
                         else _coord if _coord is not None else 4)
            elif arg_name == "Rb" or alias_clean == "Rb":
                # Coordinate Y on Hopper+; the GPR texture *handle* pre-Hopper.
                value = (_gpr_handle if _gpr_mode
                         else _coord if _coord is not None else 0)
            elif arg_name == "paramA" or alias_clean == "paramA":
                # Texture dimension: 1=2D
                value = 1
            elif arg_name == "wmsk" or alias_clean == "wmsk":
                # Write mask: 0xF = all 4 channels
                value = 0xF
            elif arg_name == "lodlc":
                value = 0  # No LOD control
            elif arg_name == "aoffi":
                # Use UAOFFI (2) to satisfy conditions that require URe != URZ
                # when aoffi is noaoffi(0) or AOFFI(1)
                value = 2  # UAOFFI mode
            elif arg_name == "uai":
                value = 0  # No uniform address indexing
            elif arg_name == "mode" or alias_clean == "dc":
                value = 0  # No depth compare
            elif arg_name == "cop":
                value = 0  # Default cache op
            elif arg_name == "op":
                # For TLD4, op alias is "toff" - need UAOFFI (3) to allow valid URe
                # For TEX, op alias is "div" - use 0
                if alias_clean == "toff":
                    value = 3  # UAOFFI mode for TLD4
                else:
                    value = 0  # Default for TEX (div)
            elif arg_name == "memdesc" or alias_clean == "nodep":
                value = 0
            elif arg_name == "Pu" or alias_clean == "Pu":
                value = 7  # PT
            elif arg_name == "scr":
                value = 0
            elif arg_name == "rm16":
                value = 0
            else:
                alias_field = alias.lstrip("*").split("(")[0].split("@")[0].strip()
                if arg_name in field_defaults:
                    value = field_defaults[arg_name]
                elif alias_field and alias_field in field_defaults:
                    value = field_defaults[alias_field]
                elif alias.startswith("*") and alias_field in enums:
                    value = enums[alias_field]
                elif arg_name.startswith("P") or "pred" in alias.lower():
                    value = 7  # PT
                else:
                    value = 0
        elif is_mem_instr:
            # MEMORY INSTRUCTION: Check if this argument needs a valid address register
            # Memory instructions need valid address in Ra (or Rb, Rc for some)
            alias_clean = alias.split("(")[0].split("@")[0].strip().lstrip("*")
            # Check FORMAT DEFAULTS first - if argument is fixed to RZ/URZ, respect that
            # (e.g., __RaRZ variants where Ra must be RZ)
            if arg_name in field_defaults and field_defaults[arg_name] == 255:
                # FORMAT says this register must be RZ/URZ - don't override with address
                value = 255
            elif alias_clean in address_reg_args and arg_name not in field_defaults:
                # Address operand. Atomics need an aligned base: use the harness's
                # aligned atomic scratch register when the register map provides
                # it; otherwise R2 (the kernel's live LDG/STG address).
                _atom_reg = (config.register_map or {}).get("atom_addr_reg")
                if is_atomic_instr and _atom_reg is not None:
                    value = _atom_reg
                else:
                    value = 2
            elif alias_clean == "Rd" or alias_clean == "URd":
                # Destination register - use RZ (255) for safety
                if alias_clean.startswith("UR"):
                    # LDCU/UDP instructions cannot use URZ as destination
                    if needs_valid_ur_dest:
                        value = allocator.rng.randint(0, 31)  # Valid UR destination
                    else:
                        value = 255  # URZ (8-bit encoding)
                else:
                    value = 255  # RZ
            elif "_UR" in arg_name.upper():
                # Uniform register embedded in name (Ra_URc, Rb_URc, etc.)
                # These are uniform registers in 8-bit encoding (URZ = 255)
                # Check this BEFORE generic R* check to avoid misclassification
                value = config.source_ugpr if config.source_ugpr is not None else 255
            elif alias_clean.startswith("R") and len(alias_clean) <= 3:
                # Other GPR source registers
                value = config.source_gpr if config.source_gpr is not None else 255
            elif alias_clean.startswith("UR") and len(alias_clean) <= 4:
                # Other UGPR source registers
                value = config.source_ugpr if config.source_ugpr is not None else 255
            else:
                # Other arguments (immediates, flags, enum aliases)
                alias_field = alias.lstrip("*").split("(")[0].split("@")[0].strip()
                if arg_name in field_defaults:
                    value = field_defaults[arg_name]
                elif alias_field and alias_field in field_defaults:
                    value = field_defaults[alias_field]
                elif alias.startswith("*") and alias_field.isdigit():
                    value = int(alias_field)
                elif alias.startswith("*") and alias_field in enums:
                    value = enums[alias_field]
                elif arg_name.startswith("P") or "pred" in alias.lower():
                    value = 7  # PT
                elif "VarLatOperandEnc" in alias:
                    value = 7
                elif alias.isdigit():
                    value = int(alias)
                elif config.immediate_value is not None and is_data_immediate(arg_name, alias):
                    value = config.immediate_value
                else:
                    value = 0
        elif is_udp and not is_mem_instr:
            # UDP (Uniform Datapath) instructions operate on uniform registers
            # All register arguments (Rd, Ra, Rb, Rc, etc.) are uniform registers (0-63)
            # Note: UDP memory instructions (like utmaldg) are handled in is_mem_instr branch
            alias_clean = alias.split("(")[0].split("@")[0].strip().lstrip("*") if alias else ""

            # Check if this is a register argument based on name pattern
            # Exclude modifier sub-fields (alias contains @, e.g. "URa@negate") —
            # these are 1-bit flags on registers, not registers themselves
            is_modifier = "@" in alias
            is_reg_arg = (not is_modifier and
                         (arg_name in ["Rd", "Ra", "Rb", "Rc", "Re"] or
                         arg_name.startswith("UR") or
                         "_UR" in arg_name.upper() or
                         (alias_clean.startswith("R") and len(alias_clean) <= 3) or
                         (alias_clean.startswith("UR") and len(alias_clean) <= 4)))

            is_dest = arg_name == "Rd" or arg_name == "URd" or alias_clean == "Rd" or alias_clean == "URd"

            if is_reg_arg:
                if is_dest:
                    value = allocator.rng.randint(0, 31)
                else:
                    value = config.source_ugpr if config.source_ugpr is not None else 255
            else:
                udp_alias_field = alias_clean.lstrip("*")
                if arg_name in field_defaults:
                    value = field_defaults[arg_name]
                elif udp_alias_field and udp_alias_field in field_defaults:
                    value = field_defaults[udp_alias_field]
                elif alias.startswith("*") and udp_alias_field in enums:
                    value = enums[udp_alias_field]
                elif arg_name.startswith("P") or "pred" in alias.lower():
                    value = 7  # PT
                elif config.immediate_value is not None and is_data_immediate(arg_name, alias):
                    value = config.immediate_value
                else:
                    value = 0
        elif arg_name in field_defaults and field_defaults[arg_name] == 255:
            # FORMAT DEFAULTS specify this register as RZ/URZ (255)
            # This takes priority over general allocation (e.g., __RaRZ variants)
            value = 255
        elif config.validate_register_sizes and "SIZE" in predicates:
            # Use SIZE-aware register allocation
            # Check if this argument has a size correlation
            size_info = None
            for size_key, size_data in size_correlations.items():
                for arg in size_data.get("arguments", []):
                    if arg.get("name") == arg_name:
                        size_info = size_data
                        break
                if size_info:
                    break

            if size_info:
                # Allocate register based on size
                size_bits = size_info["size_bits"]
                reg_count = size_info["register_count"]

                # Determine register type from argument name or alias
                # Check: arg starts with "UR", contains "_UR" (like Ra_URc), alias contains "uniform", or alias starts with "UR"
                alias_upper = alias.upper()
                has_ur_suffix = "_UR" in arg_name.upper()  # Detect Ra_URc, Rb_URc, etc.
                if arg_name.startswith("UR") or has_ur_suffix or "uniform" in alias.lower() or alias_upper.startswith("UR"):
                    reg_type = "UGPR"
                else:
                    reg_type = "GPR"

                # Check if this should prefer zero
                prefer_zero = "RZ" in alias or "URZ" in alias

                # Determine if this is destination or source by checking ALIAS
                # Registers have single-letter aliases like Ra, Rb, Rc, Rd (GPR) or URa, URb, URc, URd (UGPR)
                # Immediates have aliases like Sb, Sc, Imm, ImmU, etc.
                alias_clean = alias.split("(")[0].split("@")[0].strip()  # Remove (PT), @not, etc.
                # Exclude modifier sub-fields (alias contains @, e.g. "Ra@negate")
                is_modifier = "@" in alias
                # Check for both GPR (Ra, Rb, etc.) and UGPR (URa, URb, etc.)
                is_gpr_register = (not is_modifier and
                              alias_clean.startswith("R") and len(alias_clean) <= 3 and
                              not any(x in alias_clean for x in ["RZ", "Rpc"]))
                is_ugpr_register = (not is_modifier and
                              alias_clean.startswith("UR") and len(alias_clean) <= 4 and
                              "URZ" not in alias_clean)
                is_register = is_gpr_register or is_ugpr_register
                is_dest = alias_clean == "Rd" or alias_clean == "URd"
                is_source = is_register and not is_dest

                # Check if this is a memory address operand (for memory instructions)
                # Memory instructions need valid address registers, not RZ
                # Strip leading '*' from alias for comparison (e.g., "*Ra" -> "Ra")
                alias_for_addr_check = alias_clean.lstrip("*")
                is_address_operand = is_mem_instr and alias_for_addr_check in address_reg_args

                # Apply register allocation strategy
                # MEMORY ADDRESS OPERANDS: Use R2 for 64-bit addresses (valid memory pointer)
                if is_address_operand and is_register and reg_type == "GPR":
                    # R2/R3 holds the memory buffer address in our mem-aware kernel
                    # For 64-bit addresses, use R2 (which pairs with R3)
                    value = 2  # R2 - loaded with valid memory address in kernel
                elif (config.all_rz or config.rz_gpr_only) and is_register and reg_type == "GPR":
                    if reg_count > 1:
                        max_64bit_base = 254 - reg_count + 1
                        even_bases = list(range(0, max_64bit_base + 1, 2))
                        value = allocator.rng.choice(even_bases) if even_bases else 0
                    elif is_source and config.source_gpr is not None:
                        value = config.source_gpr
                    else:
                        value = 255
                elif config.all_rz and is_register and reg_type == "UGPR":
                    if reg_count > 1:
                        max_64bit_base = 62 - reg_count + 1
                        even_bases = list(range(0, max_64bit_base + 1, 2))
                        value = allocator.rng.choice(even_bases) if even_bases else 0
                    elif is_dest and needs_valid_ur_dest:
                        value = allocator.rng.randint(0, 31)
                    elif is_source and config.source_ugpr is not None:
                        value = config.source_ugpr
                    else:
                        value = 255
                elif config.rz_gpr_only and is_register and reg_type == "UGPR":
                    # rz_gpr_only: Allow proper UR allocation (0-62) instead of URZ
                    # This helps UR instructions actually execute with real values
                    if reg_count > 1:
                        # 64-bit UGPR operation: use even-aligned register pair
                        max_64bit_base = 62 - reg_count + 1
                        even_bases = list(range(0, max_64bit_base + 1, 2))
                        value = allocator.rng.choice(even_bases) if even_bases else 0
                    elif is_dest and needs_valid_ur_dest:
                        value = allocator.rng.randint(0, 31)  # Valid UR destination
                    elif is_dest:
                        value = allocator.rng.randint(0, 62)  # Non-URZ destination
                    else:
                        value = allocator.rng.randint(0, 62)  # Non-URZ source
                elif is_dest and config.safe_dest_registers and is_register and reg_type == "GPR":
                    # Force destination to safe register range
                    min_reg, max_reg = config.safe_dest_registers
                    # For 64-bit operations, ensure even alignment and proper max
                    if reg_count > 1:
                        max_reg = min(max_reg, 254 - reg_count + 1)  # Cap for 64-bit
                        # Generate even-numbered registers in range
                        min_even = (min_reg + 1) // 2 * 2  # Round up to even
                        max_even = max_reg // 2 * 2  # Round down to even
                        if min_even <= max_even:
                            even_regs = list(range(min_even, max_even + 1, 2))
                            value = allocator.rng.choice(even_regs)
                        else:
                            value = 0  # Fallback
                    else:
                        value = allocator.rng.randint(min_reg, max_reg)
                elif is_source and config.safe_source_registers and is_register and reg_type == "GPR":
                    # Force source to safe register range
                    min_reg, max_reg = config.safe_source_registers
                    # For 64-bit operations, ensure even alignment and proper max
                    if reg_count > 1:
                        max_reg = min(max_reg, 254 - reg_count + 1)  # Cap for 64-bit
                        # Generate even-numbered registers in range
                        min_even = (min_reg + 1) // 2 * 2  # Round up to even
                        max_even = max_reg // 2 * 2  # Round down to even
                        if min_even <= max_even:
                            even_regs = list(range(min_even, max_even + 1, 2))
                            value = allocator.rng.choice(even_regs)
                        else:
                            value = 0  # Fallback
                    else:
                        value = allocator.rng.randint(min_reg, max_reg)
                elif not is_register and config.immediate_value is not None and is_data_immediate(arg_name, alias):
                    value = config.immediate_value
                elif not is_register and config.zero_immediates:
                    value = 0
                else:
                    value = allocator.allocate_register(reg_type, size_bits, prefer_zero)
                    # Force even alignment for 64-bit instructions (in case size_bits wasn't detected as 64)
                    if is_64bit_instr and is_register and value % 2 != 0:
                        # Round down to nearest even number
                        value = (value // 2) * 2
                register_allocs[arg_name] = list(range(value, value + reg_count))
            else:
                alias_field = alias.lstrip("*").split("(")[0].split("@")[0].strip()
                if arg_name in field_defaults:
                    value = field_defaults[arg_name]
                elif alias_field and alias_field in field_defaults:
                    value = field_defaults[alias_field]
                elif alias.startswith("*") and alias_field.isdigit():
                    value = int(alias_field)
                elif alias.startswith("*") and alias_field in enums:
                    value = enums[alias_field]
                elif arg_name.startswith("P") or "pred" in alias.lower():
                    prefer_true = "PT" in alias
                    value = allocator.allocate_predicate(prefer_true)
                elif "VarLatOperandEnc" in alias:
                    value = 7
                elif config.immediate_value is not None and is_data_immediate(arg_name, alias):
                    value = config.immediate_value
                else:
                    value = 0
        else:
            # Simple default allocation
            # Check for uniform register via _UR suffix in arg name (Ra_URc, Rb_URc, etc.)
            has_ur_suffix = "_UR" in arg_name.upper()

            if has_ur_suffix:
                if config.all_rz:
                    value = config.source_ugpr if config.source_ugpr is not None else 255
                elif config.rz_gpr_only:
                    value = allocator.rng.randint(0, 62)
                else:
                    value = allocator.rng.randint(0, 63)
            elif arg_name.startswith("R") and not arg_name.startswith("Rpc"):
                # Check if this is actually a register by examining alias
                alias_clean = alias.split("(")[0].split("@")[0].strip()
                alias_upper = alias_clean.upper()

                # Check if alias indicates uniform register (URa, URb, etc.)
                is_uniform_via_alias = alias_upper.startswith("UR") and len(alias_clean) <= 4

                is_gpr_register = (alias_clean.startswith("R") and len(alias_clean) <= 3 and
                              not any(x in alias_clean for x in ["RZ", "Rpc"]))
                is_ugpr_register = is_uniform_via_alias and "URZ" not in alias_upper
                is_register = is_gpr_register or is_ugpr_register
                is_dest = alias_clean == "Rd" or alias_upper == "URD"
                is_source = is_register and not is_dest

                # Check if this is a memory address operand (for memory instructions)
                # Strip leading '*' from alias for comparison (e.g., "*Ra" -> "Ra")
                alias_for_addr_check = alias_clean.lstrip("*")
                is_address_operand = is_mem_instr and alias_for_addr_check in address_reg_args

                # Apply register allocation strategy
                # MEMORY ADDRESS OPERANDS: Use R2 for 64-bit addresses (valid memory pointer)
                if is_address_operand and is_register and not is_ugpr_register:
                    # R2/R3 holds the memory buffer address in our mem-aware kernel
                    value = 2  # R2 - loaded with valid memory address in kernel
                elif config.all_rz and is_register and is_ugpr_register:
                    if is_64bit_instr:
                        even_bases = list(range(0, 62, 2))
                        value = allocator.rng.choice(even_bases) if even_bases else 0
                    elif is_dest and needs_valid_ur_dest:
                        value = allocator.rng.randint(0, 31)
                    elif is_source and config.source_ugpr is not None:
                        value = config.source_ugpr
                    else:
                        value = 255
                elif config.rz_gpr_only and is_register and is_ugpr_register:
                    # rz_gpr_only: Allow proper UR allocation (0-62) instead of URZ
                    if is_64bit_instr:
                        # 64-bit: use even-aligned register pair
                        even_bases = list(range(0, 62, 2))
                        value = allocator.rng.choice(even_bases) if even_bases else 0
                    elif is_dest and needs_valid_ur_dest:
                        value = allocator.rng.randint(0, 31)  # Valid UR destination
                    elif is_dest:
                        value = allocator.rng.randint(0, 62)  # Non-URZ destination
                    else:
                        value = allocator.rng.randint(0, 62)  # Non-URZ source
                elif (config.all_rz or config.rz_gpr_only) and is_register and not is_ugpr_register:
                    if is_source and config.source_gpr is not None:
                        value = config.source_gpr
                    else:
                        value = 255
                elif is_dest and config.safe_dest_registers and is_register:
                    # Force destination to safe register range
                    min_reg, max_reg = config.safe_dest_registers
                    # Conservative: without size info, generate even-aligned regs for safety
                    if min_reg <= max_reg:
                        value = allocator.rng.randint(min_reg, max_reg)
                    else:
                        value = 0
                elif is_source and config.safe_source_registers and is_register:
                    # Force source to safe register range
                    min_reg, max_reg = config.safe_source_registers
                    # Conservative: without size info, generate even-aligned regs for safety
                    if min_reg <= max_reg:
                        value = allocator.rng.randint(min_reg, max_reg)
                    else:
                        value = 0
                elif not is_register and config.immediate_value is not None and is_data_immediate(arg_name, alias):
                    value = config.immediate_value
                elif not is_register and config.zero_immediates:
                    value = 0
                else:
                    if is_64bit_instr and is_register:
                        # 64-bit instruction: force even-aligned register (0, 2, 4, ..., 14)
                        value = allocator.rng.choice(range(0, 16, 2))
                    else:
                        value = allocator.rng.randint(0, 15)
            elif arg_name.startswith("UR") or "_UR" in arg_name.upper():
                # Uniform register (UR*, or Ra_URc, Rb_URc, etc.)
                # Determine if this is a destination based on alias
                alias_clean = alias.split("(")[0].split("@")[0].strip()
                alias_upper = alias_clean.upper()
                ur_is_dest = alias_upper == "URD" or alias_clean == "Rd"

                if config.all_rz:
                    if is_64bit_instr:
                        even_bases = list(range(0, 62, 2))
                        value = allocator.rng.choice(even_bases) if even_bases else 0
                    elif ur_is_dest and needs_valid_ur_dest:
                        value = allocator.rng.randint(0, 31)
                    elif not ur_is_dest and config.source_ugpr is not None:
                        value = config.source_ugpr
                    else:
                        value = 255
                elif is_64bit_instr:
                    even_bases = list(range(0, 62, 2))
                    value = allocator.rng.choice(even_bases) if even_bases else 0
                else:
                    value = allocator.rng.randint(0, 63)  # Valid uniform register range
            else:
                alias_field = alias.lstrip("*").split("(")[0].split("@")[0].strip()
                if arg_name in field_defaults:
                    value = field_defaults[arg_name]
                elif alias_field and alias_field in field_defaults:
                    value = field_defaults[alias_field]
                elif alias.startswith("*") and alias_field.isdigit():
                    value = int(alias_field)
                elif alias.startswith("*") and alias_field in enums:
                    value = enums[alias_field]
                elif arg_name.startswith("P"):
                    value = allocator.allocate_predicate(True)
                elif "VarLatOperandEnc" in alias:
                    value = 7
                elif config.immediate_value is not None and is_data_immediate(arg_name, alias):
                    value = config.immediate_value
                elif config.zero_immediates:
                    value = 0
                else:
                    value = 0

        # Apply width mask
        if width is not None:
            value &= (1 << int(width)) - 1

        # Store value and write to buffer
        values[arg_name] = value
        write_bit_segments(buffer, bit_spec, value, width)

    # Control section (bytes 8-15) is now handled by ARGUMENTS:
    # - opex (bits 124-122, 109-105): scheduling/delay info from TABLE lookup
    # - dst_wr_sb (bits 112-110): destination write scoreboard, defaults to 7 (none)
    # - src_rel_sb (bits 115-113): source release scoreboard, defaults to 7 (none)
    # - pm_pred (bits 103-102): PM predicate
    # - req_bit_set (bits 121-116): barrier wait mask
    #
    # DO NOT overwrite these bits with hardcoded values - the ARGUMENTS encode them correctly.
    # The previous code here was corrupting the ARGUMENT-defined control fields.

    # Build alias -> argument name mapping for condition evaluation
    # Conditions reference fields by alias names (e.g., "scale") but values use
    # argument names (e.g., "cop"). We need both directions.
    alias_to_arg: Dict[str, str] = {}
    for arg_name, arg_info in arguments.items():
        if not isinstance(arg_info, Mapping):
            continue
        alias = str(arg_info.get("alias", ""))
        if alias:
            # Skip modifier aliases like "Ra@negate", "Rb@absolute" - these are
            # sub-field references, not the main field alias
            if "@" in alias:
                continue
            # Clean alias: remove function call syntax, leading *
            alias_clean = alias.split("(")[0].strip().lstrip("*")
            if alias_clean and alias_clean != arg_name and not alias_clean.isdigit():
                alias_to_arg[alias_clean] = arg_name
                # Add alias entry to values so the condition evaluator can find it
                if arg_name in values and alias_clean not in values:
                    values[alias_clean] = values[arg_name]

    # ------------------------------------------------------------------
    # User field overrides (GeneratorConfig.overrides): explicit
    # field-or-alias name -> integer value. Applied after the default
    # value cascade and never auto-fixed away, enabling the programmable
    # Encoder API ("any instruction, any modifier").
    # ------------------------------------------------------------------
    overridden_fields: Set[str] = set()
    if config.overrides:
        for field_name, raw_value in config.overrides.items():
            value = resolve_field_value(raw_value)
            real_arg = alias_to_arg.get(field_name, field_name)
            if real_arg not in arguments:
                raise KeyError(
                    f"Unknown field '{field_name}' for {instruction_name}; "
                    f"available: {sorted(arguments.keys())}")
            values[real_arg] = value
            if field_name != real_arg:
                values[field_name] = value
            arg_info = arguments[real_arg]
            bit_spec = arg_info.get("bits", "")
            width = arg_info.get("width")
            if width is not None:
                value &= (1 << int(width)) - 1
            write_bit_segments(buffer, bit_spec, value, width)
            overridden_fields.add(real_arg)

    # Validate conditions if enabled, with reactive fixing
    failed_conditions: List[str] = []
    max_fix_attempts = 3  # Try to fix up to 3 times

    # For texture/surface instructions, add alias mappings needed for condition evaluation
    # TLD4 uses "toff" alias for "op" in conditions
    if is_tex_instr or is_surf_instr:
        if "op" in values:
            values["toff"] = values["op"]  # TLD4 condition uses toff
        if "mode" in values:
            values["dc"] = values["mode"]  # TEX condition uses dc

    if config.validate_conditions:
        failed_conditions = validate_conditions(conditions, values)

        # Reactive error fixing: if conditions fail, try to fix and retry
        fix_attempt = 0
        while failed_conditions and fix_attempt < max_fix_attempts:
            fixes = analyze_and_fix_conditions(conditions, values, failed_conditions)
            # Never auto-fix fields the user explicitly overrode
            if overridden_fields:
                fixes = {k: v for k, v in fixes.items()
                         if alias_to_arg.get(k, k) not in overridden_fields}
            if not fixes:
                break  # No fixes found, give up

            # Apply fixes to values and re-write to buffer
            for fix_name, new_value in fixes.items():
                values[fix_name] = new_value
                # Resolve alias to real argument name for buffer writing
                real_arg = alias_to_arg.get(fix_name, fix_name)
                if real_arg != fix_name:
                    values[real_arg] = new_value  # Keep real arg in sync
                # Find the bit spec for this argument and re-encode
                if real_arg in arguments:
                    arg_info = arguments[real_arg]
                    bit_spec = arg_info.get("bits", "")
                    width = arg_info.get("width")
                    if width is not None:
                        new_value &= (1 << int(width)) - 1
                    write_bit_segments(buffer, bit_spec, new_value, width)

            # Re-validate
            failed_conditions = validate_conditions(conditions, values)
            fix_attempt += 1

    is_valid = len(failed_conditions) == 0

    # Extract latency information
    pipeline = latency_info.get("pipeline")
    latency_cycles = None
    occupancy = None

    if config.use_latency_info and latency_info:
        # Use TABLE_TRUE for input latency (first value)
        latencies = latency_info.get("latencies", {})
        if "TABLE_TRUE" in latencies and latencies["TABLE_TRUE"]:
            latency_cycles = latencies["TABLE_TRUE"][0]

        # Get occupancy
        pipeline_res = latency_info.get("pipeline_resource", {})
        if pipeline_res:
            occupancy = pipeline_res.get("occupancy")

    variant_id = class_name if class_name else None

    return InstructionResult(
        opcode_key=opcode_key,
        instruction_name=instruction_data.get("INSTRUCTION", opcode_key),
        class_name=class_name,
        variant_id=variant_id,
        argument_values=values,
        bytes_le=bytes(buffer),
        failed_conditions=failed_conditions,
        is_valid=is_valid,
        pipeline=pipeline,
        latency_cycles=latency_cycles,
        delay=computed_delay,
        occupancy=occupancy,
        register_allocations=register_allocs
    )


# ============================================================================
# Main Generation Logic
# ============================================================================

def assemble_encodings(
    opcode_key: str,
    instruction_data: Mapping[str, Any],
    config: GeneratorConfig,
    allocator: RegisterAllocator,
    class_name: Optional[str] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    tables: Optional[Mapping[str, Any]] = None,
) -> List[InstructionResult]:
    """Assemble one representative encoding for a class, or up to
    ``config.encodings_per_class`` distinct constraint-satisfying encodings.

    Free operand fields (registers, immediates) are drawn from the allocator's
    RNG, so re-assembling the same class yields a different fill. When ``k > 1``
    we keep the first ``k`` fills that pass the constraint AST and produce
    distinct bytes. The RNG advances deterministically, so a fixed seed
    reproduces the same set; a class with no free fields collapses to a single
    encoding. If no valid fill is found we fall back to the representative
    encoding (matching the ``k == 1`` behaviour) so coverage is never reduced.
    """
    k = config.encodings_per_class
    if k <= 1:
        return [assemble_instruction(opcode_key, instruction_data, config,
                                     allocator, class_name, defaults, tables)]

    seen: Set[bytes] = set()
    encodings: List[InstructionResult] = []
    representative: Optional[InstructionResult] = None
    # Bounded so classes with a small satisfying operand space terminate; we
    # return however many distinct encodings we found (possibly fewer than k).
    for _ in range(k * 8 + 8):
        result = assemble_instruction(opcode_key, instruction_data, config,
                                      allocator, class_name, defaults, tables)
        if representative is None:
            representative = result
        if not result.is_valid or result.bytes_le in seen:
            continue
        seen.add(result.bytes_le)
        encodings.append(result)
        if len(encodings) >= k:
            break
    return encodings if encodings else [representative]


def generate_all_instructions(
    data: Mapping[str, Any],
    config: GeneratorConfig
) -> List[InstructionResult]:
    """Generate instruction examples for all opcodes.

    Args:
        data: ISA JSON data with INSTRUCTIONS, DEFAULTS, etc.
        config: Generator configuration

    Returns:
        List of generated instruction results
    """
    rng = random.Random(config.seed)
    allocator = RegisterAllocator(config, rng)

    instructions = data.get("INSTRUCTIONS", {})
    if not isinstance(instructions, Mapping):
        raise ValueError("INSTRUCTIONS section missing or malformed")

    # Get DEFAULTS for format field lookup
    defaults = data.get("DEFAULTS", {})
    tables = data.get("TABLES", {})

    results: List[InstructionResult] = []

    for opcode_key, inst_data in instructions.items():
        if not isinstance(inst_data, Mapping):
            continue

        # Apply pipeline filter if specified
        if config.filter_pipelines:
            inst_pipeline = inst_data.get("LATENCY_INFO", {}).get("pipeline")
            if inst_pipeline not in config.filter_pipelines:
                continue  # Skip this instruction, not in allowed pipelines

        # Get all possible variants/classes for this opcode
        opcodes = inst_data.get("OPCODES")

        if config.generate_all_variants and isinstance(opcodes, Mapping) and opcodes:
            # Generate all variants (each class emits 1 or up to k encodings)
            for class_name in opcodes.keys():
                results.extend(assemble_encodings(
                    opcode_key,
                    inst_data,
                    config,
                    allocator,
                    class_name,
                    defaults,
                    tables
                ))
        elif config.random_class and isinstance(opcodes, Mapping) and opcodes:
            # Pick one random variant
            class_name = rng.choice(list(opcodes.keys()))
            results.extend(assemble_encodings(
                opcode_key,
                inst_data,
                config,
                allocator,
                class_name,
                defaults,
                tables
            ))
        else:
            # Generate single example
            results.extend(assemble_encodings(
                opcode_key,
                inst_data,
                config,
                allocator,
                None,
                defaults,
                tables
            ))

        # Check instruction limit
        if config.max_instructions and len(results) >= config.max_instructions:
            break

    return results


# ============================================================================
# Output Generation
# ============================================================================

def write_outputs(
    results: Sequence[InstructionResult],
    config: GeneratorConfig
) -> None:
    """Write results to JSON and optional binary output.

    Args:
        results: Generated instruction results
        config: Generator configuration
    """
    # Prepare JSON payload
    payload = []
    for item in results:
        payload.append({
            "opcode_key": item.opcode_key,
            "instruction": item.instruction_name,
            "class": item.class_name,
            "variant": item.variant_id,
            "argument_values": dict(item.argument_values),
            "bytes_le": item.bytes_le.hex(),
            "is_valid": item.is_valid,
            "failed_conditions": list(item.failed_conditions),
            "pipeline": item.pipeline,
            "latency_cycles": item.latency_cycles,
            "delay": item.delay,
            "occupancy": item.occupancy,
            "register_allocations": item.register_allocations,
            "static_valid": item.static_valid,
            "disasm_text": item.disasm_text
        })

    # Write JSON
    config.output_path.write_text(json.dumps(payload, indent=2))

    # Write binary if requested
    if config.binary_output:
        binary_data = b"".join(item.bytes_le for item in results)
        config.binary_output.write_bytes(binary_data)

    # Print statistics
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid

    print(f"Generated {total} instructions:")
    print(f"  Valid:   {valid}")
    print(f"  Invalid: {invalid}")

    # Always show delay encoding stats
    with_delay = sum(1 for r in results if r.delay > 0)
    delay_gt1 = sum(1 for r in results if r.delay > 1)
    print(f"  With delay > 0: {with_delay}")
    print(f"  With delay > 1: {delay_gt1}")

    if config.use_latency_info:
        with_latency = sum(1 for r in results if r.latency_cycles is not None)
        print(f"  With latency info: {with_latency}")

    if config.verbose:
        print("\nInvalid instructions:")
        for r in results:
            if not r.is_valid:
                print(f"  {r.instruction_name}: {', '.join(r.failed_conditions)}")


# ============================================================================
# cuobjdump Static Validation
# ============================================================================

def _validate_single_instruction(args: tuple) -> tuple:
    """Validate a single instruction by patching it into the harness and running cuobjdump.

    This is a module-level function so it can be pickled for ProcessPoolExecutor.

    Args:
        args: Tuple of (index, inst_bytes_hex, delay, harness_path, patch_offset,
              trigger_offset, tmp_dir)

    Returns:
        Tuple of (index, static_valid, disasm_text)
    """
    idx, inst_bytes_hex, delay, harness_path, patch_offset, trigger_offset, tmp_dir = args

    inst_bytes = bytes.fromhex(inst_bytes_hex)
    tmp_path = os.path.join(tmp_dir, f"_cuobj_validate_{idx}.bin")

    try:
        target_hex = f"{trigger_offset:04x}"

        shutil.copy2(harness_path, tmp_path)
        with open(tmp_path, "r+b") as f:
            f.seek(patch_offset)
            f.write(inst_bytes[:16])

        result = subprocess.run(
            [cuobjdump_path(), "-sass", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            disasm_text = None
            for line in result.stdout.splitlines():
                line_stripped = line.strip()
                if not (line_stripped.startswith("/*") and "*/" in line_stripped):
                    continue
                try:
                    offset_str = line_stripped.split("*/")[0].replace("/*", "").strip()
                    if offset_str.lower() == target_hex:
                        disasm_text = line_stripped
                        break
                except (ValueError, IndexError):
                    pass
            return (idx, True, disasm_text)

        return (idx, False, None)

    except subprocess.TimeoutExpired:
        return (idx, False, None)
    except Exception:
        return (idx, False, None)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def cuobjdump_validate(
    results: List[InstructionResult],
    harness_path: Path,
    offsets_path: Path,
    max_workers: int = 24,
    verbose: bool = False
) -> List[InstructionResult]:
    """Run cuobjdump static validation on all instructions in parallel.

    Patches each instruction into slot 0's trigger_offset in the harness binary,
    then runs cuobjdump -sass to check if the binary still disassembles correctly.

    Args:
        results: List of generated instruction results
        harness_path: Path to the harness binary
        offsets_path: Path to perslot_clean_offsets.json
        max_workers: Maximum parallel cuobjdump workers
        verbose: Print progress information

    Returns:
        Updated list of InstructionResult with static_valid and disasm_text set
    """
    # Load offsets (unified v2 schema)
    from ..schemas import OffsetsDoc
    doc = OffsetsDoc.load(offsets_path)
    code_base = doc.code_base_offset
    trigger_offset = doc.slots[0].anchors["trigger"]
    patch_offset = code_base + trigger_offset

    harness_str = str(harness_path)

    # Validate ALL instructions — condition-invalid ones may still have valid encodings
    to_validate = list(enumerate(results))
    total = len(to_validate)

    if verbose:
        print(f"\ncuobjdump validation: {total} instructions to check")
        print(f"  Harness: {harness_path}")
        print(f"  Patch offset: {patch_offset} (code_base={code_base} + trigger={trigger_offset})")
        print(f"  Workers: {max_workers}")

    # Create temp directory for worker files
    tmp_dir = tempfile.mkdtemp(prefix="cuobj_validate_")

    try:
        # Build work items
        work_items = []
        for orig_idx, result in to_validate:
            work_items.append((
                orig_idx,
                result.bytes_le.hex(),
                result.delay,
                harness_str,
                patch_offset,
                trigger_offset,
                tmp_dir
            ))

        # Run validation in parallel
        completed = 0
        static_valid_count = 0
        static_invalid_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_validate_single_instruction, item): item[0]
                for item in work_items
            }

            for future in as_completed(futures):
                idx, is_valid, disasm_text = future.result()
                results[idx].static_valid = is_valid
                results[idx].disasm_text = disasm_text

                if is_valid:
                    static_valid_count += 1
                else:
                    static_invalid_count += 1

                completed += 1
                if verbose and completed % 100 == 0:
                    print(f"  Progress: {completed}/{total} "
                          f"(valid={static_valid_count}, invalid={static_invalid_count})")

        # cuobjdump is the authority: promote static_valid instructions to is_valid
        promoted = 0
        for r in results:
            if r.static_valid and not r.is_valid:
                r.is_valid = True
                promoted += 1

        if verbose:
            valid_count = sum(1 for r in results if r.is_valid)
            print(f"\ncuobjdump validation complete:")
            print(f"  Static valid:   {static_valid_count}")
            print(f"  Static invalid: {static_invalid_count}")
            print(f"  Promoted (condition-invalid but cuobjdump OK): {promoted}")
            print(f"  Total usable (is_valid=True): {valid_count}")

    finally:
        # Clean up temp directory
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Constraint-aware SASS instruction encoding generator"
    )

    # Input/Output
    parser.add_argument("--json", required=True, help="Path to ISA JSON description")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--binary-output", help="Optional path to write binary stream")

    # Generation modes
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Generate all modifier/suffix combinations for each instruction"
    )
    parser.add_argument(
        "--random-class",
        action="store_true",
        help="Randomly choose one variant per opcode"
    )
    parser.add_argument(
        "--encodings-per-class",
        type=int,
        default=1,
        help="Emit up to K distinct constraint-satisfying operand fills per "
             "class (default 1 = one representative encoding)"
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        help="Maximum number of instructions to generate"
    )

    # Register configuration
    parser.add_argument(
        "--no-zero-registers",
        action="store_true",
        help="Don't prefer RZ/URZ for zero values"
    )
    parser.add_argument(
        "--no-predicate-true",
        action="store_true",
        help="Don't prefer PT for predicates"
    )
    parser.add_argument(
        "--safe-dest-registers",
        type=str,
        help="Force destination registers (Rd) to safe range, e.g., '24-31' or '255' for RZ"
    )
    parser.add_argument(
        "--safe-source-registers",
        type=str,
        help="Force source registers (Ra, Rb, Rc) to safe range, e.g., '0-7' or '255' for RZ"
    )
    parser.add_argument(
        "--all-rz",
        action="store_true",
        help="Force ALL register arguments to RZ (255) - zero inputs, discarded outputs, no side effects"
    )
    parser.add_argument(
        "--rz-gpr-only",
        action="store_true",
        help="Force GPR to RZ (255) but allow proper UR allocation (0-62) for UR instructions"
    )
    parser.add_argument(
        "--zero-immediates",
        action="store_true",
        help="Force all immediate values and non-register arguments to 0 for maximum safety"
    )
    parser.add_argument(
        "--source-gpr",
        type=int,
        metavar="N",
        help="Use register RN for all GPR source operands (e.g., 0 for R0 which holds ~1.0f at trigger time)"
    )
    parser.add_argument(
        "--source-ugpr",
        type=int,
        metavar="N",
        help="Use register URN for all UGPR source operands"
    )
    parser.add_argument(
        "--immediate-value",
        type=int,
        metavar="V",
        help="Use value V for all immediate operands (overrides --zero-immediates)"
    )

    # Latency and scheduling
    parser.add_argument(
        "--use-latency",
        action="store_true",
        help="Use real latency information from sm_xx_latencies.txt"
    )
    parser.add_argument(
        "--schedule-by-pipeline",
        action="store_true",
        help="Group instructions by pipeline for realistic scheduling"
    )

    # Validation
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip condition tree validation"
    )
    parser.add_argument(
        "--no-validate-sizes",
        action="store_true",
        help="Skip SIZE predicate validation for registers"
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Fail if any generated instruction is invalid"
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        help="Comma-separated list of pipelines to include (e.g., 'mio_pipe,int_pipe,udp_pipe')"
    )

    # cuobjdump static validation
    parser.add_argument(
        "--cuobjdump-validate",
        action="store_true",
        help="Run cuobjdump static validation (requires --harness and --offsets)"
    )
    parser.add_argument(
        "--harness",
        type=str,
        help="Path to harness binary (for cuobjdump validation)"
    )
    parser.add_argument(
        "--offsets",
        type=str,
        help="Path to perslot_clean_offsets.json (for cuobjdump validation)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Max parallel cuobjdump workers (default: 24)"
    )

    # Options
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Main entry point."""
    args = parse_arguments(argv)

    # Load JSON data
    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    data = json.loads(json_path.read_text())

    # Parse pipeline filter
    filter_pipelines = None
    if args.pipelines:
        filter_pipelines = set(p.strip() for p in args.pipelines.split(',') if p.strip())
        if args.verbose:
            print(f"Filtering to pipelines: {', '.join(sorted(filter_pipelines))}")

    # Parse safe destination register range
    safe_dest_registers = None
    if args.safe_dest_registers:
        reg_spec = args.safe_dest_registers.strip()
        if '-' in reg_spec:
            # Range format: "24-31"
            parts = reg_spec.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid register range format: {reg_spec}")
            min_reg = int(parts[0])
            max_reg = int(parts[1])
            if min_reg > max_reg or min_reg < 0 or max_reg > 255:
                raise ValueError(f"Invalid register range: {min_reg}-{max_reg}")
            safe_dest_registers = (min_reg, max_reg)
        else:
            # Single value: "255" (for RZ) or any other single register
            reg_num = int(reg_spec)
            if reg_num < 0 or reg_num > 255:
                raise ValueError(f"Invalid register number: {reg_num}")
            safe_dest_registers = (reg_num, reg_num)

        if args.verbose:
            if safe_dest_registers[0] == safe_dest_registers[1]:
                reg_name = "RZ" if safe_dest_registers[0] == 255 else f"R{safe_dest_registers[0]}"
                print(f"Forcing destination registers to: {reg_name}")
            else:
                print(f"Forcing destination registers to range: R{safe_dest_registers[0]}-R{safe_dest_registers[1]}")

    # Parse safe source register range
    safe_source_registers = None
    if args.safe_source_registers:
        reg_spec = args.safe_source_registers.strip()
        if '-' in reg_spec:
            # Range format: "0-7"
            parts = reg_spec.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid register range format: {reg_spec}")
            min_reg = int(parts[0])
            max_reg = int(parts[1])
            if min_reg > max_reg or min_reg < 0 or max_reg > 255:
                raise ValueError(f"Invalid register range: {min_reg}-{max_reg}")
            safe_source_registers = (min_reg, max_reg)
        else:
            # Single value: "255" (for RZ) or any other single register
            reg_num = int(reg_spec)
            if reg_num < 0 or reg_num > 255:
                raise ValueError(f"Invalid register number: {reg_num}")
            safe_source_registers = (reg_num, reg_num)

        if args.verbose:
            if safe_source_registers[0] == safe_source_registers[1]:
                reg_name = "RZ" if safe_source_registers[0] == 255 else f"R{safe_source_registers[0]}"
                print(f"Forcing source registers to: {reg_name}")
            else:
                print(f"Forcing source registers to range: R{safe_source_registers[0]}-R{safe_source_registers[1]}")

    # Check --all-rz flag
    if args.all_rz and args.verbose:
        print(f"Forcing ALL registers to RZ (255) - zero side effects mode")

    if args.zero_immediates and args.verbose:
        print(f"Forcing all immediate values to 0 - maximum safety mode")

    if args.source_gpr is not None and args.verbose:
        print(f"Using R{args.source_gpr} for all GPR source operands")
    if args.source_ugpr is not None and args.verbose:
        print(f"Using UR{args.source_ugpr} for all UGPR source operands")
    if args.immediate_value is not None and args.verbose:
        print(f"Using {args.immediate_value} for all immediate operands")

    # Create configuration
    config = GeneratorConfig(
        json_path=json_path,
        output_path=Path(args.output),
        binary_output=Path(args.binary_output) if args.binary_output else None,
        generate_all_variants=args.all_variants,
        random_class=args.random_class,
        encodings_per_class=args.encodings_per_class,
        use_zero_registers=not args.no_zero_registers,
        use_predicate_true=not args.no_predicate_true,
        safe_dest_registers=safe_dest_registers,
        safe_source_registers=safe_source_registers,
        all_rz=args.all_rz,
        rz_gpr_only=args.rz_gpr_only,
        zero_immediates=args.zero_immediates,
        source_gpr=args.source_gpr,
        source_ugpr=args.source_ugpr,
        immediate_value=args.immediate_value,
        use_latency_info=args.use_latency,
        schedule_by_pipeline=args.schedule_by_pipeline,
        validate_conditions=not args.no_validate,
        validate_register_sizes=not args.no_validate_sizes,
        fail_on_invalid=args.fail_on_invalid,
        filter_pipelines=filter_pipelines,
        verbose=args.verbose,
        seed=args.seed,
        max_instructions=args.max_instructions
    )

    # Generate instructions
    results = generate_all_instructions(data, config)

    # Check for failures
    if config.fail_on_invalid:
        invalid = [r for r in results if not r.is_valid]
        if invalid:
            print(f"ERROR: {len(invalid)} invalid instructions generated")
            for r in invalid[:10]:  # Show first 10
                print(f"  {r.instruction_name}: {', '.join(r.failed_conditions)}")
            raise RuntimeError("Invalid instructions generated with --fail-on-invalid")

    # Write outputs
    write_outputs(results, config)

    # Run cuobjdump static validation if requested
    if args.cuobjdump_validate:
        if not args.harness or not args.offsets:
            raise ValueError("--cuobjdump-validate requires --harness and --offsets")

        harness_path = Path(args.harness)
        offsets_path = Path(args.offsets)

        if not harness_path.exists():
            raise FileNotFoundError(f"Harness binary not found: {harness_path}")
        if not offsets_path.exists():
            raise FileNotFoundError(f"Offsets file not found: {offsets_path}")

        results = cuobjdump_validate(
            results,
            harness_path,
            offsets_path,
            max_workers=args.workers,
            verbose=args.verbose
        )

        # Re-write JSON output with static_valid fields
        write_outputs(results, config)

        # Print final summary
        usable = sum(1 for r in results if r.is_valid)
        static_invalid = sum(1 for r in results if r.static_valid is False)
        print(f"\nFinal summary:")
        print(f"  Usable (is_valid=True):    {usable}")
        print(f"  Rejected (cuobjdump BAD):  {static_invalid}")
        print(f"  Total:                     {len(results)}")

    print(f"\nResults written to: {config.output_path}")
    if config.binary_output:
        print(f"Binary stream written to: {config.binary_output}")


# ============================================================================
# Programmable Encoder API
# ============================================================================

_GPR_RE = re.compile(r"^R(\d+|Z)$", re.IGNORECASE)
_UGPR_RE = re.compile(r"^UR(\d+|Z)$", re.IGNORECASE)
_PRED_RE = re.compile(r"^P(\d+|T)$", re.IGNORECASE)


def resolve_field_value(raw: Any) -> int:
    """Resolve a field override value to an integer.

    Accepts ints, numeric strings (decimal/hex/binary), register spellings
    (``R5``, ``RZ``, ``UR3``, ``URZ``) and predicate spellings
    (``P0``..``P6``, ``PT``).
    """
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    for pattern, special in ((_GPR_RE, 255), (_UGPR_RE, 255), (_PRED_RE, 7)):
        m = pattern.match(text)
        if m:
            tok = m.group(1).upper()
            return special if tok in ("Z", "T") else int(tok)
    return parse_int_literal(text)


class EncodedInstruction:
    """One encoded instruction with validation diagnostics."""

    def __init__(self, result: InstructionResult):
        self.result = result

    @property
    def bytes(self) -> bytes:
        """The 16-byte encoding (little-endian)."""
        return self.result.bytes_le

    @property
    def hex(self) -> str:
        return self.result.bytes_le.hex()

    @property
    def valid(self) -> bool:
        """True if all constraint conditions are satisfied."""
        return self.result.is_valid

    @property
    def failed_conditions(self) -> List[str]:
        """Precise violated constraints (empty when valid)."""
        return self.result.failed_conditions

    @property
    def argument_values(self) -> Dict[str, int]:
        return self.result.argument_values

    @property
    def pipeline(self) -> Optional[str]:
        return self.result.pipeline

    @property
    def control(self) -> Dict[str, int]:
        """Decoded control fields (stall/yield/scoreboard barriers)."""
        from .control import read_control
        return read_control(self.result.bytes_le)

    def check(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        return (f"<EncodedInstruction {self.result.instruction_name} "
                f"valid={self.valid} bytes={self.hex}>")


class Encoder:
    """Programmable SASS encoder.

    Builds any instruction class from the ISA spec, with any modifier or
    field values, validating against the constraint trees::

        enc = Encoder(sm=120)
        ins = enc.build("fadd__RRR_RR", Rd="R5", Ra="R2", Rb="R3")
        ins.bytes     # 16-byte SASS encoding
        ins.valid     # constraint check result

    Fields are ISA-JSON ARGUMENTS names (or their aliases); values may be
    ints, numeric strings, or register spellings ("R5", "RZ", "PT").
    Control fields (``dst_wr_sb``, ``src_rel_sb``, ``req_bit_set``,
    ``pm_pred``, ``opex``) are plain fields. Convenience: ``stall=N``
    sets the opex stall/usched low bits (batch_t=NOP).

    By default the base register/operand policy mirrors ``make generate``
    (``all_rz=True, zero_immediates=True``); pass config overrides as
    keyword arguments (e.g. ``all_rz=False``) to change it.
    """

    def __init__(self, spec=None, *, sm=None, isa_json=None, seed: int = 0,
                 validate: bool = True, **config_kwargs):
        from ..isa.model import ISASpec  # local import to avoid a cycle

        if spec is None:
            spec = ISASpec.load(sm=sm, path=isa_json)
        self.spec = spec

        cfg = dict(
            json_path=Path(str(getattr(spec, "source", None) or "<spec>")),
            output_path=Path("<memory>"),
            all_rz=True,
            zero_immediates=True,
            validate_conditions=validate,
            seed=seed,
        )
        cfg.update(config_kwargs)
        self.config = GeneratorConfig(**cfg)

    def build(self, name: str, fields: Optional[Mapping[str, Any]] = None,
              *, stall: Optional[int] = None, opex: Optional[int] = None,
              **field_kwargs: Any) -> EncodedInstruction:
        """Encode one instruction class with explicit field values."""
        ic = self.spec.get(name)
        if ic is None:
            raise KeyError(f"Unknown instruction class: {name!r}")

        overrides: Dict[str, Any] = dict(fields or {})
        overrides.update(field_kwargs)

        if stall is not None:
            if opex is not None:
                raise ValueError("pass only one of stall= / opex=")
            if not 0 <= stall <= 15:
                raise ValueError(f"stall out of range (0-15): {stall}")
            # batch_t=NOP (0), usched_info = stall count
            opex = stall & 0x1F
        if opex is not None:
            overrides["opex"] = opex

        import dataclasses
        config = dataclasses.replace(self.config, overrides=overrides)
        allocator = RegisterAllocator(config, random.Random(config.seed))
        result = assemble_instruction(
            ic.name,
            ic.data,
            config,
            allocator,
            class_name=None,
            defaults=self.spec.defaults,
            tables=self.spec.tables,
        )
        return EncodedInstruction(result)

    def generate_all(self) -> List[InstructionResult]:
        """Bulk generation: one constraint-satisfying encoding per class."""
        return generate_all_instructions(self.spec.raw, self.config)


if __name__ == "__main__":
    main()
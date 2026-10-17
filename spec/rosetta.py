"""Rosetta String parser: converts sm_XX_instructions.txt (+ latencies)
into the enriched, queryable ISA JSON consumed by the whole framework.

Absorbed from OSIRIS/json_maker.py. Reverse-engineered knowledge
(scheduling enums, shader types, special registers) lives in
``eoss.spec.data`` instead of being hardcoded here.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from .conditions import BinaryOp, DefinedCall, Identifier, Literal, Node, UnaryOp, parse_condition
from . import data as spec_data


def parse_int(value: str) -> int:
    value = value.strip().strip(";")
    if value.startswith("0x"):
        return int(value, 16)
    if value.startswith("0b"):
        return int(value, 2)
    return int(value)


def parse_parameters(lines: List[str]) -> Dict[str, int]:
    params: Dict[str, int] = {}
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip().startswith("PARAMETERS")) + 1
    except StopIteration:
        return params

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("CONSTANTS"):
            break
        if "=" in stripped:
            name, value = stripped.split("=", 1)
            params[name.strip()] = parse_int(value)
        idx += 1
    return params


def parse_constants(lines: List[str]) -> Dict[str, int]:
    constants: Dict[str, int] = {}
    try:
        idx = next(i for i, line in enumerate(lines) if line.strip().startswith("CONSTANTS")) + 1
    except StopIteration:
        return constants

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("STRING_MAP"):
            break
        if "=" in stripped:
            name, value = stripped.split("=", 1)
            constants[name.strip()] = parse_int(value)
        idx += 1
    return constants


def split_comma_aware(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    in_quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in {'"', '\''}:
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
        if ch == ',' and in_quote is None:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
        i += 1
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_default_entry(line: str) -> Tuple[Optional[str], Dict[str, Any]]:
    stripped = line.strip()
    if not stripped:
        return None, {}
    if stripped.endswith(';'):
        stripped = stripped[:-1].strip()
    if not stripped:
        return None, {}
    if ' ' not in stripped:
        return stripped, {}
    name, remainder = stripped.split(None, 1)
    values: Dict[str, Any] = {}
    for part in split_comma_aware(remainder):
        if not part:
            continue
        if '=' in part:
            key_raw, value_raw = part.split('=', 1)
            key = key_raw.strip().strip("\"'")
            value_text = value_raw.strip().strip("\"'")
            if not value_text:
                value: Any = None
            else:
                try:
                    value = parse_int(value_text)
                except ValueError:
                    value = value_text
        else:
            key = part.strip().strip("\"'")
            value = None
        values[key] = value
    return name, values

def parse_format_value(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if not value:
        return value
    try:
        return parse_int(value)
    except ValueError:
        return value


def split_format_token_parts(token: str) -> List[str]:
    parts: List[str] = []
    remaining = token
    while remaining:
        if remaining.startswith("','"):
            parts.append("','")
            remaining = remaining[3:]
            continue
        if remaining[0] in {',', ';'}:
            parts.append(remaining[0])
            remaining = remaining[1:]
            continue
        if remaining[0] == "'" and len(remaining) > 1:
            end = remaining.find("'", 1)
            if end != -1:
                parts.append(remaining[: end + 1])
                remaining = remaining[end + 1 :]
                continue
        parts.append(remaining)
        break
    return parts


def split_format_line(line: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    in_quote: str | None = None
    paren_depth = 0
    brace_depth = 0

    for ch in line:
        if ch in {'"', '\''}:
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
        elif in_quote is None:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth = max(paren_depth - 1, 0)
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth = max(brace_depth - 1, 0)

        if ch.isspace() and in_quote is None and paren_depth == 0 and brace_depth == 0:
            if current:
                token = ''.join(current)
                tokens.extend(split_format_token_parts(token))
                current = []
        else:
            current.append(ch)

    if current:
        token = ''.join(current)
        tokens.extend(split_format_token_parts(token))

    return [token for token in tokens if token]


def split_format_tokens(lines: List[str]) -> List[str]:
    tokens: List[str] = []
    for line in lines:
        tokens.extend(split_format_line(line))
    return tokens


def parse_format(lines: List[str], start_idx: int) -> Tuple[Dict[str, Any], int]:
    stripped = lines[start_idx].strip()
    if not stripped.startswith("FORMAT"):
        return {}, start_idx

    raw_lines: List[str] = []
    body_lines: List[str] = []

    idx = start_idx
    while idx < len(lines):
        current = lines[idx].rstrip('\n')
        current_stripped = current.strip()
        raw_lines.append(current_stripped)

        if idx == start_idx:
            body = current_stripped[len("FORMAT"):].strip()
            if body:
                body_lines.append(body)
        else:
            if current_stripped:
                body_lines.append(current_stripped)

        if current_stripped.endswith(';'):
            idx += 1
            break
        idx += 1
    else:
        idx = start_idx + 1

    processed_lines = [line[:-1].rstrip() if line.endswith(';') else line for line in body_lines]
    processed_lines = [line for line in processed_lines if line]

    tokens = split_format_tokens(processed_lines)
    joined_text = " ".join(processed_lines).strip()

    entry_pattern = re.compile(
        r"([A-Za-z0-9_@!/%\[\]\*\?\.\-\+]+(?:\([^)]*\))?(?:\*+)?)\s*:\s*([A-Za-z0-9_]+)"
    )
    entries: List[Dict[str, Any]] = []
    defaults: Dict[str, Any] = {}

    for token in tokens:
        for raw_source, target in entry_pattern.findall(token):
            source = raw_source
            default_value: Any | None = None
            if '(' in raw_source and raw_source.endswith(')'):
                paren_index = raw_source.index('(')
                source = raw_source[:paren_index]
                inner = raw_source[paren_index + 1 : -1]
                default_value = parse_format_value(inner)
            cleaned_source = source.lstrip('[')
            entry: Dict[str, Any] = {
                "source": cleaned_source or source,
                "target": target,
                "raw": raw_source,
            }
            if default_value is not None:
                entry["default"] = default_value
                defaults[target] = default_value
            entries.append(entry)

    format_text = "FORMAT"
    if joined_text:
        format_text = f"{format_text} {joined_text}"

    format_data: Dict[str, Any] = {
        "LINES": raw_lines,
        "TEXT": format_text,
    }
    if tokens:
        format_data["TOKENS"] = tokens
    if entries:
        format_data["FIELDS"] = entries
    if defaults:
        format_data["DEFAULTS"] = defaults

    return format_data, idx


def parse_usched_info(lines: List[str], start_idx: int) -> Tuple[str, Dict[str, Any], int]:
    name = lines[start_idx].strip().split()[0]
    values: Dict[str, Any] = {}
    i = start_idx + 1

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        done = False
        if stripped.endswith(';'):
            stripped = stripped[:-1].strip()
            done = True
        if stripped.endswith(','):
            stripped = stripped[:-1].strip()
        if stripped:
            for part in split_comma_aware(stripped):
                entry = part.strip()
                if not entry:
                    continue
                operator = '='
                if '*=' in entry:
                    key_raw, value_raw = entry.split('*=', 1)
                    operator = '*='
                elif '=' in entry:
                    key_raw, value_raw = entry.split('=', 1)
                else:
                    key_raw, value_raw = entry, ''
                key = key_raw.strip().strip('\"\'')
                value_text = value_raw.strip().strip('\"\'')
                if value_text:
                    try:
                        value: Any = parse_int(value_text)
                    except ValueError:
                        value = value_text
                else:
                    value = None
                if operator != '=':
                    key = f"{key} {operator}"
                values[key] = value
        i += 1
        if done:
            break

    return name, values, i

def parse_defaults(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    defaults: Dict[str, Dict[str, Any]] = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith('COLONLY'))
    except StopIteration:
        # SM70 and older don't have COLONLY — find enum definitions by pattern
        # Enum lines look like: NAME "val1"=N , "val2"=M ;
        enum_pattern = re.compile(r'^([A-Z][A-Za-z0-9_]*)\s+"')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('TABLES'):
                break
            m = enum_pattern.match(stripped)
            if m:
                name, values = parse_default_entry(stripped)
                if name and values:
                    defaults[name] = values
        return defaults

    buffer: List[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped == 'TABLES':
            break
        if stripped.startswith('USCHED_INFO'):
            name, values, i = parse_usched_info(lines, i)
            if name:
                defaults[name] = values
            continue
        if not buffer and ' ' not in stripped:
            i += 1
            continue
        buffer.append(stripped)
        if stripped.endswith(';'):
            combined = ' '.join(buffer)
            buffer = []
            name, values = parse_default_entry(combined)
            if name:
                defaults[name] = values
        i += 1
    return defaults


def parse_enum_values(lines: List[str]) -> Dict[str, int]:
    enum_values: Dict[str, int] = {}
    current_name: str | None = None
    buffer: List[str] = []

    enum_line = re.compile(r"^\s*([A-Za-z0-9_]+)\s+\"")

    for line in lines:
        if "->" in line:
            continue
        if current_name is None:
            match = enum_line.match(line)
            if not match:
                continue
            current_name = match.group(1)
            buffer = [line.strip()]
        else:
            buffer.append(line.strip())
        if current_name is not None and ";" in line:
            combined = " ".join(buffer)
            entries = combined.split(";", 1)[0]
            for part in entries.split(","):
                part = part.strip()
                if not part or "=" not in part:
                    continue
                entry, value = part.split("=", 1)
                entry = entry.strip()
                value = value.strip()
                entry_clean = entry.strip("\"")
                try:
                    numeric = parse_int(value)
                except ValueError:
                    continue
                key_base = f"{current_name}@{entry_clean}"
                enum_values[key_base] = numeric
                if entry != entry_clean:
                    enum_values[f"{current_name}@{entry}"] = numeric
                    enum_values[f"{current_name}@\"{entry_clean}\""] = numeric
            current_name = None
            buffer = []
    return enum_values


def parse_special_registers(lines: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Read the architecture's SpecialRegister enum and retain every alias.

    Numeric SR0..SR255 names cover unnamed encodings; the first descriptive
    name wins when several names share an index. Dotted coordinate names keep
    the framework's underscore spelling, with the source spelling as an alias.
    Framework-discovered names take precedence over the source's display name.
    """
    match = re.search(r"^\s*SpecialRegister\s+([^;]+);", "\n".join(lines), re.MULTILINE)
    if match is None:
        raise ValueError("SpecialRegister enumeration not found")

    names: Dict[str, int] = {}
    registers: Dict[int, str] = {}
    next_index = 0
    name_range = re.compile(r"([A-Za-z_][A-Za-z0-9_]*?)\((\d+)\.\.(\d+)\)")
    value_range = re.compile(r"\((\d+)\.\.(\d+)\)")

    for entry in split_comma_aware(match.group(1)):
        name, equals, value = entry.partition("=")
        name = name.strip().strip('"')
        value = value.strip()
        expanded = name_range.fullmatch(name)
        if expanded:
            prefix, first, last = expanded.groups()
            first, last = int(first), int(last)
            if last < first:
                raise ValueError(f"Invalid special register range: {entry}")
            entry_names = [f"{prefix}{i}" for i in range(first, last + 1)]
        else:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
                raise ValueError(f"Invalid special register name: {name}")
            entry_names = [name]

        if equals:
            indices = value_range.fullmatch(value)
            if indices:
                first, last = map(int, indices.groups())
                if last - first + 1 != len(entry_names):
                    raise ValueError(f"Mismatched special register ranges: {entry}")
                next_index = first
            else:
                next_index = parse_int(value)

        for entry_name in entry_names:
            if not 0 <= next_index <= 255:
                raise ValueError(f"Special register index out of range: {next_index}")
            if entry_name in names and names[entry_name] != next_index:
                raise ValueError(f"Conflicting special register name: {entry_name}")
            names[entry_name] = next_index
            current = registers.get(next_index, f"SR{next_index}")
            if next_index not in registers or current == f"SR{next_index}":
                registers[next_index] = entry_name.replace(".", "_")
            next_index += 1

    registers.update(spec_data.special_registers())
    aliases = {name: index for name, index in names.items()
               if registers[index] != name}
    return dict(sorted(registers.items())), aliases


def parse_bits(bits_string: str, alias: str = "") -> Dict[str, Any]:
    numbers = bits_string.split("_")[2:]
    ranges: List[Tuple[int, int]] = []
    number_of_bits = bits_string.split("_")[1]
    bit_name = ""
    i = 0
    while i < len(numbers):
        token = numbers[i]
        if token.isdigit():
            start = int(token)
            end = int(numbers[i + 1])
            ranges.append((start, end))
        else:
            bit_name = "_".join(numbers[i:])
            break
        i += 2
    return {
        "ranges": ranges,
        "number_of_bits": int(number_of_bits),
        "bit_name": bit_name,
        "alias": alias,
    }


def format_ranges(ranges: List[Tuple[int, int]]) -> Any:
    if not ranges:
        return None
    if len(ranges) == 1:
        start, end = ranges[0]
        if start == end:
            return f"{start}"
        return f"{start}_{end}"
    return [f"{start}_{end}" if start != end else f"{start}" for start, end in ranges]


def parse_conditions(lines: List[str], start_idx: int) -> Tuple[List[Dict[str, str]], int]:
    conditions: List[Dict[str, str]] = []
    i = start_idx
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("PROPERTIES"):
            break

        # Check if name and expression are on the same line.
        # Format: "ERROR_NAME expression :" (name followed by expression ending with :)
        # The name is the first word (all-uppercase with underscores), and the rest
        # up to the trailing : is the expression.
        # Normal format: name is on its own line, expression on the next line(s).
        name_match = re.match(r'^([A-Z][A-Z0-9_]*(?:_ERROR|_SASS_ONLY_ERROR)?)\s+(.*)', stripped)
        if name_match and stripped.rstrip().endswith(':'):
            # Name and expression on same line
            name = name_match.group(1)
            rest = name_match.group(2)
            # The expression is everything before the last ':'
            expression = rest.rsplit(":", 1)[0].strip()
            i += 1
        else:
            name = stripped
            i += 1
            expr_parts: List[str] = []
            while i < len(lines):
                line = lines[i].strip()
                expr_parts.append(line)
                if ":" in line:
                    break
                i += 1
            expr_line = " ".join(expr_parts)
            expression = expr_line.split(":", 1)[0].strip()
            i += 1

        message_parts: List[str] = []
        while i < len(lines):
            line = lines[i].strip()
            message_parts.append(line)
            if line.endswith('"'):
                break
            i += 1
        message = " ".join(message_parts).strip().strip('"')
        i += 1
        conditions.append({"name": name, "expression": expression, "message": message})
    return conditions, i


def parse_assignment_value(text: str) -> Any:
    value = text.strip().rstrip(";").strip()
    if not value:
        return value
    try:
        return parse_int(value)
    except ValueError:
        if value.startswith("\"") and value.endswith("\""):
            return value[1:-1]
        return value


def resolve_predicate_value(value: Any, constants: Dict[str, int]) -> Any:
    """Resolve predicate values like $VQ_ADU to their constant values."""
    if isinstance(value, str):
        if value.startswith("$"):
            constant_name = value[1:]
            if constant_name in constants:
                return constants[constant_name]
    return value


def parse_simple_block(lines: List[str], start_idx: int) -> Tuple[Dict[str, Any], int]:
    data: Dict[str, Any] = {}
    i = start_idx
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if "=" not in stripped:
            if re.fullmatch(r"[A-Z0-9_ ]+", stripped):
                break
            if stripped.endswith(":"):
                break
            if stripped.startswith("TABLE_"):
                break
            if stripped.startswith("CONNECTOR"):
                break
            if stripped.startswith("OPCODES"):
                break
            if stripped.startswith("FORMAT"):
                break
            if stripped.startswith("CONDITIONS"):
                break
            if stripped.startswith("ENCODING"):
                break
        if "=" in stripped:
            left, right = stripped.split("=", 1)
            key = left.strip()
            value = parse_assignment_value(right)
            data[key] = value
            i += 1
            continue
        break
    return data, i


def parse_opcodes_block(lines: List[str], start_idx: int) -> Tuple[str, Dict[str, str], int]:
    opcode_value = ""
    aliases: Dict[str, str] = {}
    i = start_idx
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("ENCODING"):
            break
        if "=" not in stripped:
            if re.fullmatch(r"[A-Z0-9_ ]+", stripped):
                break
            if stripped.startswith("TABLE_"):
                break
            if stripped.startswith("CONNECTOR"):
                break
        if "=" in stripped:
            # Handle multiple opcode definitions on one line separated by ;
            # e.g.: "LDmio_pipe = 0b110; LD = 0b110;"
            parts = [p.strip() for p in stripped.split(";") if "=" in p]
            if not parts:
                parts = [stripped]
            for part in parts:
                if "=" not in part:
                    continue
                left, right = part.split("=", 1)
                name = left.strip()
                value = right.strip().rstrip(";").strip()
                if not opcode_value:
                    opcode_value = value
                aliases[name] = value
            i += 1
            continue
        break
    return opcode_value, aliases, i


def parse_instructions(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    instructions: Dict[str, Dict[str, Any]] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not (line.startswith("CLASS ") or line.startswith("ALTERNATE CLASS ")):
            i += 1
            continue
        class_name = line.split("\"", 2)[1]
        i += 1


        while i < len(lines) and not lines[i].strip():
            i += 1

        format_info: Dict[str, Any] = {}
        if i < len(lines) and lines[i].strip().startswith("FORMAT"):
            format_info, i = parse_format(lines, i)

        while i < len(lines) and "CONDITIONS" not in lines[i]:
            i += 1
        if i >= len(lines):
            break
        # Handle case where CONDITIONS keyword and first condition name are on same line
        # e.g., "CONDITIONSOOR_REG_ERROR" instead of "CONDITIONS\nOOR_REG_ERROR"
        cond_line = lines[i].strip()
        cond_suffix = ""
        if cond_line.startswith("CONDITIONS") and len(cond_line) > len("CONDITIONS"):
            cond_suffix = cond_line[len("CONDITIONS"):]
        i += 1
        # If there was text after CONDITIONS on the same line, inject it back
        if cond_suffix:
            lines.insert(i, cond_suffix)

        conditions, i = parse_conditions(lines, i)

        properties: Dict[str, Any] = {}
        predicates: Dict[str, Any] = {}

        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith("PROPERTIES"):
                i += 1
                properties, i = parse_simple_block(lines, i)
                continue
            if stripped.startswith("PREDICATES"):
                i += 1
                predicates, i = parse_simple_block(lines, i)
                continue
            if stripped.startswith("OPCODES"):
                break
            i += 1

        while i < len(lines) and not lines[i].strip().startswith("OPCODES"):
            i += 1
        if i >= len(lines):
            break
        i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        opcode, opcode_aliases, i = parse_opcodes_block(lines, i)

        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith("BITS_"):
                break
            if stripped.startswith("ENCODING"):
                i += 1
                continue
            if stripped.startswith("!"):
                i += 1
                continue
            if stripped.startswith("CLASS "):
                break
            i += 1
        bits_encoding: List[Dict[str, Any]] = []
        while i < len(lines) and lines[i].strip().startswith("BITS_"):
            bits_line = lines[i].strip().rstrip(";")
            left, right = bits_line.split("=", 1)
            alias = right.strip()
            bits_encoding.append(parse_bits(left.strip(), alias))
            i += 1

        args: Dict[str, Dict[str, Any]] = {}
        for entry in bits_encoding:
            bit_name = entry["bit_name"]
            if not bit_name:
                continue
            ranges_formatted = format_ranges(entry["ranges"])
            args[bit_name] = {
                "bits": ranges_formatted,
                "alias": entry["alias"],
                "width": entry["number_of_bits"],
            }

        spec: Dict[str, Any] = {
            "INSTRUCTION": class_name,
            "ARGUMENTS": args,
            "CONDITIONS": conditions,
        }
        if opcode:
            spec["OPCODE"] = opcode
        if properties:
            spec["PROPERTIES"] = properties
        if predicates:
            spec["PREDICATES"] = predicates
        if opcode_aliases:
            spec["OPCODES"] = opcode_aliases
        instructions[class_name] = spec
        if format_info:
            instructions[class_name]["FORMAT"] = format_info
    return instructions


def resolve_identifier(name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    if name in context["arguments"]:
        data = context["arguments"][name]
        return {
            "type": "argument",
            "name": name,
            "bits": data.get("bits"),
            "width": data.get("width"),
            "alias": data.get("alias"),
        }

    if name.startswith("%"):
        key = name[1:]
        if key in context["parameters"]:
            return {"type": "parameter", "name": name, "value": context["parameters"][key]}
    if name.startswith("$"):
        key = name[1:]
        if key in context["constants"]:
            return {"type": "constant", "name": name, "value": context["constants"][key]}
    if name in context["constants"]:
        return {"type": "constant", "name": name, "value": context["constants"][name]}
    if name in context["enums"]:
        return {"type": "enum", "name": name, "value": context["enums"][name]}
    tables = context.get("tables", {})
    if name in tables:
        return {
            "type": "table",
            "name": name,
            "row_count": len(tables[name]),
        }
    if name.startswith("TABLES_"):
        return {"type": "table", "name": name}
    return {"type": "identifier", "name": name}


def literal_value(value: str) -> Any:
    value = value.strip()
    try:
        return parse_int(value)
    except ValueError:
        return value


def ast_to_dict(node: Node, context: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(node, BinaryOp):
        return {
            "type": "binary",
            "op": node.op,
            "left": ast_to_dict(node.left, context),
            "right": ast_to_dict(node.right, context),
        }
    if isinstance(node, UnaryOp):
        return {
            "type": "unary",
            "op": node.op,
            "operand": ast_to_dict(node.operand, context),
        }
    if isinstance(node, Identifier):
        return resolve_identifier(node.name, context)
    if isinstance(node, Literal):
        return {"type": "literal", "value": literal_value(node.value)}
    if isinstance(node, DefinedCall):
        return {
            "type": "defined",
            "table_name": node.table_name,
            "args": node.args,
            "negated": node.negated,
        }
    return {"type": "unknown", "repr": repr(node)}


def calculate_register_count(size_bits: int) -> int:
    """Calculate how many 32-bit registers are needed for a given bit size."""
    if size_bits <= 0:
        return 0
    return (size_bits + 31) // 32


def correlate_size_to_arguments(
    predicates: Dict[str, Any],
    arguments: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Correlate SIZE predicates to their corresponding arguments.
    Returns enriched predicates with argument correlations and register counts.
    """
    enriched_predicates = predicates.copy()
    size_correlations: Dict[str, Dict[str, Any]] = {}

    # Mapping of size predicates to argument names
    # ISRC_A_SIZE -> Ra, ISRC_B_SIZE -> Rb, IDEST_SIZE -> Rd, etc.
    size_to_arg_map = {
        "ISRC_A_SIZE": ["Ra", "URa"],
        "ISRC_B_SIZE": ["Rb", "URb"],
        "ISRC_B2_SIZE": ["Rb2", "URb2"],
        "ISRC_C_SIZE": ["Rc", "URc"],
        "ISRC_E_SIZE": ["Re", "URe"],
        "ISRC_H_SIZE": ["Rh", "URh"],
        "ISRC_I_SIZE": ["Ri", "URi"],
        "IDEST_SIZE": ["Rd", "URd"],
        "IDEST2_SIZE": ["Rd2", "URd2"],
        "ISRC_A_INDEX_RF_SIZE": ["Ra_index"],
        "ISRC_B_INDEX_RF_SIZE": ["Rb_index"],
        "ISRC_C_INDEX_RF_SIZE": ["Rc_index"],
        "ISRC_E_INDEX_RF_SIZE": ["Re_index"],
    }

    # Also handle ILABEL_*_SIZE predicates
    for key in predicates.keys():
        if key.startswith("ILABEL_") and key.endswith("_SIZE"):
            # Extract register name: ILABEL_Ra_SIZE -> Ra
            reg_name = key[7:-5]  # Remove "ILABEL_" and "_SIZE"
            size_to_arg_map[key] = [reg_name]

    # Correlate sizes to arguments
    for size_key, potential_args in size_to_arg_map.items():
        if size_key in predicates:
            size_value = predicates[size_key]
            if isinstance(size_value, int):
                reg_count = calculate_register_count(size_value)
                correlation: Dict[str, Any] = {
                    "size_bits": size_value,
                    "register_count": reg_count,
                    "arguments": []
                }

                # Find matching arguments
                for arg_name in potential_args:
                    if arg_name in arguments:
                        correlation["arguments"].append({
                            "name": arg_name,
                            "bits": arguments[arg_name].get("bits"),
                            "width": arguments[arg_name].get("width"),
                            "alias": arguments[arg_name].get("alias"),
                        })

                size_correlations[size_key] = correlation

    # Add correlations to enriched predicates
    if size_correlations:
        enriched_predicates["_SIZE_CORRELATIONS"] = size_correlations

    return enriched_predicates


def enrich_predicates(
    instructions: Dict[str, Dict[str, Any]],
    constants: Dict[str, int]
) -> None:
    """Enrich predicates with resolved constants and size correlations."""
    for spec in instructions.values():
        predicates = spec.get("PREDICATES", {})
        if not predicates:
            continue

        arguments = spec.get("ARGUMENTS", {})

        # Resolve constant values in predicates
        resolved_predicates: Dict[str, Any] = {}
        for key, value in predicates.items():
            resolved_value = resolve_predicate_value(value, constants)
            resolved_predicates[key] = resolved_value

        # Add size correlations
        enriched = correlate_size_to_arguments(resolved_predicates, arguments)
        spec["PREDICATES"] = enriched


def enrich_conditions(instructions: Dict[str, Dict[str, Any]], context: Dict[str, Any]) -> None:
    for spec in instructions.values():
        args = spec.get("ARGUMENTS", {})
        conds = spec.get("CONDITIONS", [])
        enriched = []
        for cond in conds:
            expression = cond.get("expression", "")
            tree = None
            error = None
            try:
                ast = parse_condition(expression)
                tree = ast_to_dict(ast, {**context, "arguments": args})
            except Exception as exc:  # pragma: no cover - defensive
                error = str(exc)
            enriched.append({
                "name": cond.get("name"),
                "message": cond.get("message"),
                "expression": expression,
                "tree": tree,
                "error": error,
            })
        spec["CONDITIONS"] = enriched


def parse_table_line(line: str) -> Dict[str, Any]:
    left, right = line.split("->", 1)
    left_text = left.strip()
    right_text = right.strip()
    try:
        output: Any = parse_int(right_text)
    except ValueError:
        output = right_text
    return {
        "expression": left_text,
        "inputs": left_text.split(),
        "output": output,
    }


def parse_table_at(lines: List[str], start_idx: int) -> Tuple[str, List[Dict[str, Any]], int]:
    name = lines[start_idx].strip()
    rows: List[Dict[str, Any]] = []
    i = start_idx + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped == ';':
            i += 1
            break
        if '->' not in stripped:
            break
        rows.append(parse_table_line(stripped))
        i += 1
    return name, rows, i


def parse_tables(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    tables: Dict[str, List[Dict[str, Any]]] = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == 'TABLES':
            i += 1
            while i < len(lines):
                inner = lines[i].strip()
                if not inner:
                    i += 1
                    continue
                if inner.startswith('TABLES_') or inner.startswith('CLASS '):
                    break
                if inner == ';':
                    i += 1
                    continue
                name, rows, i = parse_table_at(lines, i)
                tables[name] = rows
            continue
        if stripped.startswith('TABLES_'):
            name, rows, i = parse_table_at(lines, i)
            tables[name] = rows
            continue
        i += 1
    return tables


def parse_operation_sets(lines: List[str]) -> Dict[str, str]:
    """
    Parse OPERATION SETS to build instruction-to-pipeline mapping.
    Returns dict mapping instruction names to their pipeline (e.g., 'AL2P' -> 'mio_pipe')
    """
    instruction_to_pipeline: Dict[str, str] = {}
    in_operation_sets = False

    for line in lines:
        stripped = line.strip()

        if stripped == "OPERATION SETS":
            in_operation_sets = True
            continue

        if in_operation_sets:
            # Stop when we hit another section
            if stripped.startswith("HARD RESOURCE") or stripped.startswith("CONNECTOR") or stripped.startswith("TABLE_"):
                break

            # Look for lines like: int_pipe = {CS2R,CS2Rint_pipe,ICMP,ICMPint_pipe,...};
            if '=' in stripped and '{' in stripped:
                # Extract pipeline name and instruction list
                parts = stripped.split('=', 1)
                if len(parts) == 2:
                    pipeline_name = parts[0].strip()
                    inst_part = parts[1].strip()

                    # Extract instructions from braces
                    if '{' in inst_part:
                        start = inst_part.index('{') + 1
                        end = inst_part.rfind('}')
                        if end > start:
                            inst_list = inst_part[start:end]
                            # Split by comma and process each instruction
                            for inst in inst_list.split(','):
                                inst = inst.strip()
                                if inst and not inst.endswith('_pipe'):
                                    # Only store instruction names without _pipe suffix
                                    instruction_to_pipeline[inst] = pipeline_name

    return instruction_to_pipeline


def parse_latency_tables(lines: List[str]) -> Dict[str, Dict[str, List[int]]]:
    """
    Parse TABLE_TRUE, TABLE_OUTPUT, TABLE_ANTI to extract latency values.
    Returns dict like: {'TABLE_TRUE': {'FXU_OPS': [6,6,6,...], ...}, ...}

    Also stores GPR-specific metadata under special keys:
      '_GPR_TRUE_COLUMNS': list of column header op_set names for GPR TABLE_TRUE
      '_GPR_TRUE_ROWS': dict of {op_set: [values]} from GPR TABLE_TRUE only
    These enable computing the self-pipeline (diagonal) latency for each instruction.
    """
    tables: Dict[str, Dict[str, List[int]]] = {
        'TABLE_TRUE': {},
        'TABLE_OUTPUT': {},
        'TABLE_ANTI': {}
    }

    current_table = None
    current_resource = None  # GPR, UGPR, PRED, RPC, etc.
    current_section_lines: List[str] = []
    header_lines: List[str] = []  # Lines between TABLE start and body {
    in_table_body = False
    gpr_true_columns: List[str] = []  # Column headers for first GPR TABLE_TRUE
    gpr_true_rows: Dict[str, List[int]] = {}  # Body rows for first GPR TABLE_TRUE

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect table start — extract resource type from TABLE_xxx(RESOURCE)
        if stripped.startswith('TABLE_TRUE(') or stripped.startswith('TABLE_OUTPUT(') or stripped.startswith('TABLE_ANTI('):
            if stripped.startswith('TABLE_TRUE'):
                current_table = 'TABLE_TRUE'
            elif stripped.startswith('TABLE_OUTPUT'):
                current_table = 'TABLE_OUTPUT'
            else:
                current_table = 'TABLE_ANTI'
            # Extract resource type: TABLE_TRUE(GPR) → GPR
            paren_start = stripped.index('(')
            paren_end = stripped.index(')')
            current_resource = stripped[paren_start + 1:paren_end]
            current_section_lines = []
            header_lines = [stripped]  # Start collecting header lines
            in_table_body = False
            continue

        # Look for the opening brace that starts the table body
        # Must distinguish body-start "{" from connector specs like OP`{Ra...}
        # Body starts with a standalone "{" line or "= {" pattern
        if current_table and not in_table_body:
            is_body_start = stripped.startswith('{') or '= {' in stripped
            if is_body_start:
                in_table_body = True
                # Extract GPR TABLE_TRUE column headers from header lines
                # Each header line has OP_SET`{connectors} patterns
                if (current_table == 'TABLE_TRUE' and current_resource == 'GPR'
                        and not gpr_true_columns):
                    for hline in header_lines:
                        for match in re.finditer(r'(\w+)`\{', hline):
                            gpr_true_columns.append(match.group(1))
                header_lines = []
                # Start collecting body from this line
                current_section_lines.append(stripped)
                continue
            else:
                # Still in header (continuation lines before {)
                header_lines.append(stripped)
                continue

        # Collect lines until we hit the closing brace and semicolon
        if current_table and in_table_body:
            current_section_lines.append(stripped)
            if stripped.endswith('};'):
                # Parse the collected section
                parse_latency_table_body(current_section_lines, tables[current_table])
                # Also store GPR TABLE_TRUE rows separately
                if (current_table == 'TABLE_TRUE' and current_resource == 'GPR'
                        and not gpr_true_rows):
                    parse_latency_table_body(current_section_lines, gpr_true_rows)
                current_table = None
                current_resource = None
                in_table_body = False
                current_section_lines = []

    # Store GPR metadata under special keys
    tables['_GPR_TRUE_COLUMNS'] = gpr_true_columns
    tables['_GPR_TRUE_ROWS'] = gpr_true_rows

    return tables


def parse_latency_table_body(lines: List[str], table_dict: Dict[str, List[int]]) -> None:
    """Parse the body of a latency table and extract operation set latencies."""
    for line in lines:
        # Look for lines like: FXU_WITHOUT_DUALALU`{Rd @RdRange,Rd2 @Rd2Range} : 6 6 6 6 6 6 6 6 8 6 6 7 7 7 7 7 7 6
        if ':' in line and not line.strip().startswith('{'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                # Extract operation set name (before the backtick or colon)
                left = parts[0].strip()
                # Remove everything after ` if present
                if '`' in left:
                    op_set_name = left.split('`')[0].strip()
                else:
                    op_set_name = left.strip()

                # Extract latency numbers
                right = parts[1].strip()
                # Remove trailing semicolon if present
                if right.endswith(';'):
                    right = right[:-1].strip()

                # Extract numbers
                latencies: List[int] = []
                for token in right.split():
                    try:
                        latencies.append(int(token))
                    except ValueError:
                        pass

                if latencies and op_set_name:
                    table_dict[op_set_name] = latencies


def parse_pipeline_resources(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse OPERATION PIPELINE RESOURCES section.
    Returns dict like: {'FXU_OPS': {'resource': 'FXU_Occupancy', 'occupancy': 2}, ...}
    """
    resources: Dict[str, Dict[str, Any]] = {}
    in_section = False

    for line in lines:
        stripped = line.strip()

        if stripped == "OPERATION PIPELINE RESOURCES":
            in_section = True
            continue

        if in_section:
            # Stop at next major section
            if stripped.startswith("RESOURCE") or stripped.startswith("CONNECTOR") or stripped.startswith("TABLE_") or stripped.startswith("OPERATION SETS"):
                if not stripped.startswith("OPERATION PIPELINE"):
                    break

            # Look for lines like: FXU_OPS : FXU_Occupancy [2];
            if ':' in stripped and '[' in stripped:
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    op_set = parts[0].strip()
                    right = parts[1].strip()

                    # Extract resource name and occupancy
                    if '[' in right and ']' in right:
                        resource_part = right.split('[')[0].strip()
                        occupancy_part = right.split('[')[1].split(']')[0].strip()

                        try:
                            occupancy = int(occupancy_part)
                            resources[op_set] = {
                                'resource': resource_part,
                                'occupancy': occupancy
                            }
                        except ValueError:
                            pass

    return resources


def match_instruction_to_latency(
    instruction_name: str,
    pipeline_mapping: Dict[str, str],
    operation_sets_expanded: Dict[str, Set[str]],
    latency_tables: Dict[str, Dict[str, List[int]]],
    pipeline_resources: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Match an instruction to its pipeline and latency information.

    Prefers GPR TABLE_TRUE entries (multi-value arrays from the register
    dependency matrix) over single-value entries from RPC/PRED/etc. tables.
    Also computes gpr_self_latency: the self-pipeline diagonal latency
    from the GPR TABLE_TRUE matrix, which represents the RAW latency when
    the same instruction type reads its own output.
    """
    result: Dict[str, Any] = {}

    # Find pipeline
    pipeline = pipeline_mapping.get(instruction_name)
    if pipeline:
        result['pipeline'] = pipeline

    # Find which operation sets this instruction belongs to
    matching_op_sets: List[str] = []
    for op_set, instructions in operation_sets_expanded.items():
        if instruction_name in instructions:
            matching_op_sets.append(op_set)

    # Retrieve GPR TABLE_TRUE metadata (stored by parse_latency_tables)
    gpr_true_columns = latency_tables.get('_GPR_TRUE_COLUMNS', [])
    gpr_true_rows = latency_tables.get('_GPR_TRUE_ROWS', {})

    if matching_op_sets:
        result['operation_sets'] = matching_op_sets

        # Select the best TABLE_TRUE match: prefer GPR entries (longer arrays)
        # over RPC/PRED entries (single values). This ensures we get the actual
        # register RAW latencies, not unrelated RPC dependency latencies.
        best_latencies = None
        best_tt_len = -1
        for op_set in matching_op_sets:
            latencies: Dict[str, List[int]] = {}
            for table_name, table_data in latency_tables.items():
                if table_name.startswith('_'):
                    continue  # Skip metadata keys
                if isinstance(table_data, dict) and op_set in table_data:
                    latencies[table_name] = table_data[op_set]

            if latencies:
                tt_len = len(latencies.get('TABLE_TRUE', []))
                if tt_len > best_tt_len:
                    best_tt_len = tt_len
                    best_latencies = latencies
                    result['_matched_op_set'] = op_set

        if best_latencies:
            result['latencies'] = best_latencies

        # Compute GPR self-pipeline (diagonal) latency
        # This is the RAW latency when the same pipeline reads its own output,
        # matching nak's self-RAW computation.
        if gpr_true_rows and gpr_true_columns:
            for op_set in matching_op_sets:
                if op_set in gpr_true_rows:
                    row_values = gpr_true_rows[op_set]
                    # Find self-column: where column header matches this op_set
                    try:
                        self_col = gpr_true_columns.index(op_set)
                        if self_col < len(row_values):
                            result['gpr_self_latency'] = row_values[self_col]
                    except ValueError:
                        # Op_set is a writer but not a reader column header
                        # (e.g., CLMAD_OP). Use first column as fallback.
                        if row_values:
                            result['gpr_self_latency'] = row_values[0]
                    break

        # Get pipeline resource info for the first matching operation set
        for op_set in matching_op_sets:
            if op_set in pipeline_resources:
                result['pipeline_resource'] = pipeline_resources[op_set]
                break

    return result


def expand_operation_sets(lines: List[str]) -> Dict[str, Set[str]]:
    """
    Parse and expand OPERATION SETS to get all instructions in each set.
    Handles set arithmetic like: FXU_OPS = int_pipe + fe_pipe - IMMA_OP - MOVM_OP;
    """
    operation_sets: Dict[str, Set[str]] = {}

    # First pass: collect all raw definitions
    raw_definitions: Dict[str, str] = {}
    in_operation_sets = False

    for line in lines:
        stripped = line.strip()

        if stripped == "OPERATION SETS":
            in_operation_sets = True
            continue

        if in_operation_sets:
            if stripped.startswith("HARD RESOURCE") or stripped.startswith("CONNECTOR") or stripped.startswith("TABLE_") or stripped.startswith("PIPELINE RESOURCE"):
                in_operation_sets = False
                continue

            if '=' in stripped:
                parts = stripped.split('=', 1)
                if len(parts) == 2:
                    set_name = parts[0].strip()
                    definition = parts[1].strip().rstrip(';').strip()
                    raw_definitions[set_name] = definition

    # Second pass: expand definitions into instruction sets
    for set_name, definition in raw_definitions.items():
        instructions = expand_set_definition(definition, raw_definitions, operation_sets)
        if instructions:
            operation_sets[set_name] = instructions

    return operation_sets


def expand_set_definition(
    definition: str,
    raw_definitions: Dict[str, str],
    already_expanded: Dict[str, Set[str]]
) -> Set[str]:
    """Recursively expand a set definition to get all instructions."""
    result: Set[str] = set()

    # Handle set with braces: {A,B,C}
    if '{' in definition:
        start = definition.index('{') + 1
        end = definition.rfind('}')
        if end > start:
            inst_list = definition[start:end]
            for inst in inst_list.split(','):
                inst = inst.strip()
                if inst and not inst.endswith('_pipe'):
                    result.add(inst)
        return result

    # Handle set arithmetic: A + B - C
    # Simple parsing: split by + and -, track operations
    tokens: List[Tuple[str, str]] = []  # [(op, name), ...]
    current_op = '+'
    current_token: List[str] = []

    for char in definition + ' ':  # Add space to flush last token
        if char in {'+', '-', ' ', '\t'}:
            if current_token:
                token_str = ''.join(current_token).strip()
                if token_str:
                    tokens.append((current_op, token_str))
                current_token = []
            if char in {'+', '-'}:
                current_op = char
        else:
            current_token.append(char)

    # Evaluate tokens
    for op, name in tokens:
        # Check if it's a reference to another set
        if name in already_expanded:
            instructions = already_expanded[name]
        elif name in raw_definitions:
            # Recursively expand
            instructions = expand_set_definition(raw_definitions[name], raw_definitions, already_expanded)
            already_expanded[name] = instructions
        else:
            # It's a single instruction or unknown
            instructions = {name} if name else set()

        if op == '+':
            result |= instructions
        elif op == '-':
            result -= instructions

    return result


def enrich_instructions_with_latencies(
    instructions: Dict[str, Dict[str, Any]],
    latency_lines: List[str]
) -> None:
    """Add latency and pipeline information to each instruction."""
    # Parse latency file
    pipeline_mapping = parse_operation_sets(latency_lines)
    latency_tables = parse_latency_tables(latency_lines)
    pipeline_resources = parse_pipeline_resources(latency_lines)
    operation_sets_expanded = expand_operation_sets(latency_lines)

    # Match each instruction
    for spec in instructions.values():
        inst_name = spec.get("INSTRUCTION")
        if not inst_name:
            continue

        # Extract base instruction name (remove variant suffix after __)
        # e.g., "al2p__RaRZ" -> "al2p" -> "AL2P"
        base_name = inst_name.split('__')[0].split('_')[0].upper()

        latency_info = match_instruction_to_latency(
            base_name,
            pipeline_mapping,
            operation_sets_expanded,
            latency_tables,
            pipeline_resources
        )

        if latency_info:
            spec["LATENCY_INFO"] = latency_info




def build_isa_json(instruction_file, latency_file=None, verbose: bool = True) -> dict:
    """Build the enriched ISA JSON document from Rosetta String files.

    Args:
        instruction_file: path to sm_XX_instructions.txt
        latency_file: optional path to sm_XX_latencies.txt
        verbose: print progress

    Returns:
        The ISA JSON dict (top-level keys: DEFAULTS, INSTRUCTIONS,
        PARAMETERS, CONSTANTS, TABLES, SCHEDULING, SHADER_TYPES,
        SPECIAL_REGISTERS, SPECIAL_REGISTER_ALIASES).
    """
    instruction_file = Path(instruction_file)
    latency_file = Path(latency_file) if latency_file else None

    lines = [line.rstrip("\n") for line in instruction_file.read_text(encoding="utf-8").splitlines()]
    parameters = parse_parameters(lines)
    constants = parse_constants(lines)
    defaults = parse_defaults(lines)
    enum_values = parse_enum_values(lines)
    special_registers, special_register_aliases = parse_special_registers(lines)
    tables = parse_tables(lines)
    instructions = parse_instructions(lines)

    context = {
        "parameters": parameters,
        "constants": constants,
        "enums": enum_values,
        "tables": tables,
    }

    # Enrich predicates with resolved constants and size correlations
    enrich_predicates(instructions, constants)

    # Enrich conditions with parsed expression trees
    enrich_conditions(instructions, context)

    # Enrich with latency and pipeline information
    if latency_file and latency_file.exists():
        latency_lines = [line.rstrip("\n") for line in latency_file.read_text(encoding="utf-8").splitlines()]
        enrich_instructions_with_latencies(instructions, latency_lines)
    elif verbose:
        print(f"Warning: Latency file not found: {latency_file}")

    # ====================================================================
    # Post-processing: evaluate VALID_IN_SHADERS, extract VIRTUAL_QUEUE,
    # add scheduling enumerations, and build special register map.
    # ====================================================================

    ishader_values = spec_data.ishader_values()

    # Evaluate VALID_IN_SHADERS bitmask for each instruction
    for name, spec in instructions.items():
        props = spec.get("PROPERTIES", {})
        vis_str = props.get("VALID_IN_SHADERS", "")
        if vis_str:
            try:
                # Evaluate expression like "(1<<ISHADER_TRAP)+(1<<ISHADER_CS)+0"
                expr = str(vis_str)
                for k, v in ishader_values.items():
                    expr = expr.replace(k, str(v))
                vis_value = eval(expr)  # noqa: S307 - trusted spec input
                props["VALID_IN_SHADERS_BITMASK"] = vis_value
            except Exception:
                pass

        # Extract VIRTUAL_QUEUE from PREDICATES
        preds = spec.get("PREDICATES", {})
        vq = preds.get("VIRTUAL_QUEUE", "")
        if vq:
            vq_str = str(vq).replace("$", "").strip()
            try:
                preds["VIRTUAL_QUEUE_ID"] = int(vq_str)
            except ValueError:
                # Try looking up in constants (e.g., "VQ_TTU" -> 21)
                vq_num = constants.get(vq_str)
                if vq_num is not None:
                    preds["VIRTUAL_QUEUE_ID"] = vq_num

        # Extract MIN_WAIT_NEEDED
        min_wait = props.get("MIN_WAIT_NEEDED")
        if min_wait is not None:
            try:
                props["MIN_WAIT_NEEDED"] = int(min_wait)
            except (ValueError, TypeError):
                pass

    return {
        "DEFAULTS": defaults,
        "INSTRUCTIONS": instructions,
        "PARAMETERS": parameters,
        "CONSTANTS": constants,
        "TABLES": tables,
        "SCHEDULING": spec_data.scheduling_enums(),
        "SHADER_TYPES": spec_data.shader_types(),
        "SPECIAL_REGISTERS": special_registers,
        "SPECIAL_REGISTER_ALIASES": special_register_aliases,
    }


def main(argv=None) -> None:
    from .. import paths as _paths

    parser = argparse.ArgumentParser(
        description="Generate ISA JSON from Rosetta String (SASS) specification files"
    )
    parser.add_argument(
        "--sm", default="120",
        help="SM architecture version (e.g. 70, 80, 86, 89, 90, 100, 120)"
    )
    parser.add_argument(
        "--sass-dir", default=None,
        help="Path to SASS directory (default: $EOSS_SASS_DIR or <repo>/SASS)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: json_<SM>_with_conditions.json in current dir)"
    )
    args = parser.parse_args(argv)

    sm = args.sm
    base_path = Path(args.sass_dir) if args.sass_dir else _paths.sass_dir()

    instruction_file = base_path / f"sm_{sm}_instructions.txt"
    latency_file = base_path / f"sm_{sm}_latencies.txt"

    if not instruction_file.exists():
        print(f"Error: Instruction file not found: {instruction_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating ISA JSON for SM{sm}...")
    print(f"  Instructions: {instruction_file}")
    print(f"  Latencies:    {latency_file}")

    output = build_isa_json(instruction_file, latency_file)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"json_{sm}_with_conditions.json")

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({len(output['INSTRUCTIONS'])} instructions)")


if __name__ == "__main__":
    main()

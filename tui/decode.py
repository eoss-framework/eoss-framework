"""Pure instruction decoding and rendering for the TUI.

Mirrors the decode pipeline of ``VisualModifier/modifier.html``:

* opcode -> candidate classes (via :meth:`ISASpec.opcode_index`)
* variant disambiguation by condition score (pass - 4 * fail)
* per-argument values read straight from the instruction bits
* tri-state condition evaluation (pass / fail / unknown)
* an nvdisasm-flavored one-line rendering built from FORMAT TOKENS

No curses in this module; everything is unit-testable.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..enc.bits import parse_bit_spec, read_bit_segments, spec_width
from ..enc.control import read_control
from ..enc.opcode import extract_opcode
from ..isa.model import InstructionClass, ISASpec

PT = 7  # Predicate@PT
RZ = 255  # Register@RZ


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@dataclass
class FieldView:
    """One editable argument of a decoded instruction."""

    name: str
    alias: str
    bits: Any
    bits_text: str
    width: int
    value: int
    display: str
    bucket: Optional[str] = None  # DEFAULTS enum bucket used for display
    source: Optional[str] = None  # FORMAT source type (e.g. SpecialRegister)


@dataclass
class CheckView:
    """One evaluated CONDITION."""

    name: str
    status: str  # "pass" | "fail" | "unknown"
    message: str


@dataclass
class DecodedInstruction:
    offset: int  # absolute file offset
    raw: bytes
    opcode: int
    defn: Optional[InstructionClass]
    candidates: List[str] = field(default_factory=list)
    values: Dict[str, int] = field(default_factory=dict)
    fields: List[FieldView] = field(default_factory=list)
    control: Dict[str, int] = field(default_factory=dict)
    checks: List[CheckView] = field(default_factory=list)
    mnemonic: str = "?"
    operands: str = ""

    @property
    def class_name(self) -> Optional[str]:
        return self.defn.name if self.defn else None

    @property
    def text(self) -> str:
        return f"{self.mnemonic} {self.operands}".strip()

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_alias_keys(name: str, alias: Any) -> List[str]:
    """All identifier spellings under which an argument's value is known.

    Mirrors ``collectAliasKeys`` in modifier.html (case variants included).
    """
    keys = set()

    def add(k: str) -> None:
        k = k.strip()
        if k:
            keys.update({k, k.lower(), k.upper()})

    add(name)
    if isinstance(alias, str):
        cleaned = alias.split(";")[0].strip()
        add(cleaned)
        no_quotes = cleaned.replace("`", "").replace('"', "").replace("'", "")
        add(no_quotes)
        for text in (cleaned, no_quotes):
            if "@" in text:
                add(text[:text.index("@")])
            if "(" in text and text.endswith(")"):
                base = text[:text.index("(")]
                add(base)
                if not base.startswith("TABLES_"):
                    # a TABLES_x(a,b) alias's paren names are table INPUTS
                    # (decomposed in values_for), not aliases for this value
                    for part in text[text.index("(") + 1:-1].split(","):
                        add(part)
    return [k for k in keys if k]


_TABLECALL_ALIAS = re.compile(r"^(TABLES_\w+)\(([^)]*)\)$")


def bits_text(bit_spec: Any) -> str:
    """Human-readable bit ranges, MSB-first: ``"91, 11-0"``."""
    parts = []
    for lo, hi in parse_bit_spec(bit_spec):
        parts.append(str(hi) if hi == lo else f"{hi}-{lo}")
    return ", ".join(parts)


def _clean_source(source: str) -> str:
    """FORMAT FIELDS 'source' -> DEFAULTS bucket name.

    ``"@[!]Predicate"`` -> ``Predicate``; ``"/COP"`` -> ``COP``;
    ``"SImm(24/0)*"`` -> ``SImm``.
    """
    s = source.strip()
    for junk in ("@", "/", "[!]", "[-]", "[||]", "[~]"):
        s = s.replace(junk, "")
    if "(" in s:
        s = s[:s.index("(")]
    return s.strip("*").strip()


class _Token:
    """One parsed FORMAT token."""

    __slots__ = ("kind", "literal", "source", "target", "default",
                 "optional_not", "bracket_open", "bracket_close")

    def __init__(self, kind: str, literal: str = "", source: str = "",
                 target: str = "", default: Any = None, optional_not: bool = False,
                 bracket_open: bool = False, bracket_close: bool = False):
        self.kind = kind  # "literal" | "opcode" | "guard" | "modifier" | "operand"
        self.literal = literal
        self.source = source
        self.target = target
        self.default = default
        self.optional_not = optional_not
        self.bracket_open = bracket_open  # token text began with "["
        self.bracket_close = bracket_close  # target ended with "]"


def _parse_token(tok: Any) -> Optional[_Token]:
    if not isinstance(tok, str):
        return None
    t = tok.strip()
    if not t or t in ("PREDICATE", "FORMAT"):
        return None
    if t.startswith("$("):  # scheduling group: shown in the SCHEDULING pane
        return None
    if t in ("[-]", "[||]", "[!]", "[~]"):  # optional sign/abs wrappers
        return None
    if t.startswith("{") and t.endswith("}"):  # optional group, e.g. {/REUSE...}
        t = t[1:-1].strip()
        if not t:
            return None
    if t == "Opcode":
        return _Token("opcode")
    # Quoted literals: "','", "'+'", "'['" ...
    if len(t) >= 2 and t[0] == t[-1] == "'":
        return _Token("literal", literal=t[1:-1])
    if t in ("[", "]", "+", ",", "."):
        return _Token("literal", literal=t)
    if ":" in t:
        head, target = t.rsplit(":", 1)
        bracket_open = (head.startswith("[")
                        and not head.startswith(("[!]", "[-]", "[||]", "[~]")))
        if bracket_open:
            head = head[1:]
        target = target.strip()
        bracket_close = target.endswith("]")
        target = target.rstrip("]*").strip()
        default = None
        if "(" in head and head.rstrip("*").endswith(")"):
            inner = head[head.index("(") + 1:head.rstrip("*").rindex(")")]
            default = inner.strip().strip('"').strip("'")
        kind = "modifier" if head.startswith("/") else "operand"
        if head.startswith("@"):
            kind = "guard"
        return _Token(kind, source=_clean_source(head), target=target,
                      default=default, optional_not="[!]" in head,
                      bracket_open=bracket_open, bracket_close=bracket_close)
    # Bare word (e.g. stray keyword) -- render literally
    return _Token("literal", literal=t)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder:
    """Decodes 16-byte SASS instructions against one :class:`ISASpec`."""

    def __init__(self, spec: ISASpec):
        self.spec = spec
        self._rev_buckets: Dict[str, Dict[int, str]] = {}
        self._token_cache: Dict[str, List[_Token]] = {}

    # -- enum bucket helpers ------------------------------------------------

    def bucket(self, name: str) -> Optional[dict]:
        b = self.spec.defaults.get(name)
        return b if isinstance(b, dict) else None

    def bucket_rev(self, name: str) -> Dict[int, str]:
        """value -> enum member name (shortest name wins on collision)."""
        if name not in self._rev_buckets:
            rev: Dict[int, str] = {}
            for k, v in (self.bucket(name) or {}).items():
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if iv not in rev or len(k) < len(rev[iv]):
                    rev[iv] = k
            self._rev_buckets[name] = rev
        return self._rev_buckets[name]

    # -- values / fields ----------------------------------------------------

    def values_for(self, defn: InstructionClass, instr: bytes) -> Dict[str, int]:
        values: Dict[str, int] = {}
        for arg_name, info in defn.arguments.items():
            if not isinstance(info, dict):
                continue
            bits = info.get("bits")
            if not parse_bit_spec(bits):
                continue
            value = read_bit_segments(instr, bits)
            self._decompose_tablecall(info.get("alias"), value, values)
            for key in collect_alias_keys(arg_name, info.get("alias")):
                values.setdefault(key, value)
            values[arg_name] = value
        return values

    def _decompose_tablecall(self, alias: Any, value: int,
                             values: Dict[str, int]) -> None:
        """Resolve a packed TABLES_x(a,b) argument into its input fields.

        The alias means: this argument's value is the table OUTPUT for inputs
        (a, b). Looking the output up yields the individual field values
        (e.g. opex -> batch_t + usched_info), which conditions reference.
        A value with no table row maps the inputs to -1 so DEFINED lookups
        fail (the packed encoding is invalid).
        """
        if not isinstance(alias, str):
            return
        m = _TABLECALL_ALIAS.match(alias.split(";")[0].strip())
        if not m:
            return
        table = self.spec.tables.get(m.group(1))
        names = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if not isinstance(table, list) or not names:
            return
        inputs = None
        for row in table:
            if isinstance(row, dict) and _to_number(row.get("output")) == value:
                row_inputs = row.get("inputs")
                if isinstance(row_inputs, list) and len(row_inputs) == len(names):
                    inputs = [_to_number(v) for v in row_inputs]
                break
        if inputs is None:
            inputs = [-1] * len(names)
        for nm, iv in zip(names, inputs):
            for key in (nm, nm.lower(), nm.upper()):
                values.setdefault(key, iv)

    def _alias_to_arg(self, defn: InstructionClass) -> Dict[str, str]:
        """FORMAT target name -> ARGUMENTS key.

        FORMAT FIELDS name their target by the *alias* a disassembler
        prints, which need not match the ARGUMENTS key (SM120's s2r_
        stores the SpecialRegister in an argument called ``imm8`` with
        alias ``SRa``; cs2r_'s /QInteger sz modifier lives in ``ftz``).
        Real argument names always win over aliases.
        """
        out: Dict[str, str] = {arg: arg for arg in defn.arguments}
        for arg, info in defn.arguments.items():
            if not isinstance(info, dict):
                continue
            alias = info.get("alias")
            if isinstance(alias, str):
                base = alias.split(";")[0].strip()
                if base:
                    out.setdefault(base, arg)
        return out

    def field_buckets(self, defn: InstructionClass) -> Dict[str, str]:
        """argument/target name -> DEFAULTS bucket name (from FORMAT FIELDS).

        Keyed under both the ARGUMENTS name and the FORMAT target alias so
        field views (arg names) and token rendering (targets) both resolve.
        """
        out: Dict[str, str] = {}
        amap = self._alias_to_arg(defn)
        fmt = defn.data.get("FORMAT") or {}
        for f in fmt.get("FIELDS") or []:
            if isinstance(f, dict) and f.get("target") and f.get("source"):
                bucket = _clean_source(str(f["source"]))
                if self.bucket(bucket):
                    target = str(f["target"])
                    out[target] = bucket
                    out[amap.get(target, target)] = bucket
        # Common well-known buckets for args the FORMAT does not mention.
        for arg, info in defn.arguments.items():
            if arg in out or not isinstance(info, dict):
                continue
            alias = str(info.get("alias") or "")
            base = alias.split("@")[0]
            if self.bucket(base):
                out[arg] = base
        return out

    def field_sources(self, defn: InstructionClass) -> Dict[str, str]:
        """argument/target name -> cleaned FORMAT source type
        (e.g. SpecialRegister), alias-resolved like :meth:`field_buckets`."""
        out: Dict[str, str] = {}
        amap = self._alias_to_arg(defn)
        fmt = defn.data.get("FORMAT") or {}
        for f in fmt.get("FIELDS") or []:
            if isinstance(f, dict) and f.get("target") and f.get("source"):
                target = str(f["target"])
                source = _clean_source(str(f["source"]))
                out[target] = source
                out[amap.get(target, target)] = source
        return out

    def field_views(self, defn: InstructionClass, instr: bytes) -> List[FieldView]:
        buckets = self.field_buckets(defn)
        sources = self.field_sources(defn)
        views = []
        for arg_name, info in defn.arguments.items():
            if not isinstance(info, dict):
                continue
            bits = info.get("bits")
            segments = parse_bit_spec(bits)
            if not segments:
                continue
            value = read_bit_segments(instr, bits)
            width = info.get("width") or spec_width(bits)
            bucket = buckets.get(arg_name)
            display = self.display_value(arg_name, value, width, bucket,
                                         sources.get(arg_name))
            views.append(FieldView(
                name=arg_name, alias=str(info.get("alias") or arg_name),
                bits=bits, bits_text=bits_text(bits), width=width,
                value=value, display=display, bucket=bucket,
                source=sources.get(arg_name)))
        return views

    def special_register_name(self, value: int) -> str:
        return self.spec.special_registers.get(value, f"SR{value}")

    def display_value(self, arg_name: str, value: int, width: int,
                      bucket: Optional[str], source: Optional[str] = None) -> str:
        if arg_name == "opcode":
            return f"0x{value:04X}"
        if bucket:
            name = self.bucket_rev(bucket).get(value)
            if name is not None:
                return name
        if source == "SpecialRegister":
            return self.special_register_name(value)
        if width == 1:
            return str(value)
        return f"0x{value:X}"

    # -- condition evaluation ------------------------------------------------

    def eval_tree(self, node: Any, values: Dict[str, int]) -> Tuple[str, Any]:
        """Evaluate a condition-tree node -> ("ok", value) or ("unknown", why)."""
        if not isinstance(node, dict):
            return "unknown", "missing node"
        ntype = node.get("type")

        if ntype in ("literal", "constant", "parameter", "enum"):
            return "ok", _to_number(node.get("value"))

        if ntype == "argument":
            name = node.get("name")
            if name in values:
                return "ok", values[name]
            return "unknown", f"unknown argument {name}"

        if ntype == "identifier":
            return self._resolve_identifier(str(node.get("name", "")), values)

        if ntype == "defined":
            return self._eval_defined(node, values)

        if ntype == "unary":
            status, val = self.eval_tree(node.get("operand"), values)
            if status != "ok":
                return status, val
            op = node.get("op")
            if op == "!":
                return "ok", int(not _to_bool(val))
            if op == "-":
                return "ok", -_to_number(val)
            if op == "~":
                return "ok", ~_to_number(val)
            return "unknown", f"unary {op}"

        if ntype == "binary":
            ls, lv = self.eval_tree(node.get("left"), values)
            rs, rv = self.eval_tree(node.get("right"), values)
            if ls != "ok" or rs != "ok":
                return "unknown", lv if ls != "ok" else rv
            return self._apply_binary(node.get("op"), lv, rv)

        return "unknown", f"node type {ntype}"

    @staticmethod
    def _apply_binary(op: Any, lv: Any, rv: Any) -> Tuple[str, Any]:
        ln, rn = _to_number(lv), _to_number(rv)
        if op == "&&":
            return "ok", int(_to_bool(lv) and _to_bool(rv))
        if op == "||":
            return "ok", int(_to_bool(lv) or _to_bool(rv))
        table = {
            "==": lambda: int(ln == rn), "!=": lambda: int(ln != rn),
            "<": lambda: int(ln < rn), ">": lambda: int(ln > rn),
            "<=": lambda: int(ln <= rn), ">=": lambda: int(ln >= rn),
            "+": lambda: ln + rn, "-": lambda: ln - rn,
            "*": lambda: ln * rn, "/": lambda: ln // rn if rn else 0,
            "%": lambda: ln % rn if rn else 0,
            "&": lambda: ln & rn, "|": lambda: ln | rn,
            "<<": lambda: ln << rn, ">>": lambda: ln >> rn,
        }
        fn = table.get(op)
        if fn is None:
            return "unknown", f"operator {op}"
        return "ok", fn()

    def _resolve_identifier(self, raw: str,
                            values: Dict[str, int]) -> Tuple[str, Any]:
        name = raw.strip()
        for key in (name, name.lower(), name.upper()):
            if key in values:
                return "ok", values[key]
        cleaned = name.replace("`", "").replace('"', "").replace("'", "")
        if "@" in cleaned:
            base, entry = cleaned.split("@", 1)
            b = self.bucket(base)
            if b is not None:
                for k in (entry, entry.upper(), entry.lower()):
                    if k in b:
                        return "ok", _to_number(b[k])
            return "unknown", f"enum {cleaned}"
        if cleaned.startswith("%"):
            p = self.spec.parameters.get(cleaned[1:])
            if p is not None:
                return "ok", _to_number(p)
        if cleaned.startswith("$"):
            c = self.spec.constants.get(cleaned[1:])
            if c is not None:
                return "ok", _to_number(c)
        for table in (self.spec.parameters, self.spec.constants):
            if cleaned in table and not isinstance(table[cleaned], (dict, list)):
                return "ok", _to_number(table[cleaned])
        strict = _parse_strict_int(cleaned)
        if strict is not None:
            return "ok", strict
        return "unknown", f"identifier {name}"

    def _eval_defined(self, node: dict, values: Dict[str, int]) -> Tuple[str, Any]:
        table = self.spec.tables.get(str(node.get("table_name", "")))
        if not isinstance(table, list):
            return "unknown", f"table {node.get('table_name')}"
        arg_vals = []
        for arg in node.get("args") or []:
            status, val = self._resolve_identifier(str(arg), values)
            if status != "ok":
                return "unknown", f"table arg {arg}"
            arg_vals.append(_to_number(val))
        found = False
        for row in table:
            inputs = row.get("inputs") if isinstance(row, dict) else None
            if not isinstance(inputs, list) or len(inputs) != len(arg_vals):
                continue
            if all(_to_number(i) == v for i, v in zip(inputs, arg_vals)):
                found = True
                break
        result = (not found) if node.get("negated") else found
        return "ok", int(result)

    def check_condition(self, condition: dict, values: Dict[str, int]) -> CheckView:
        name = str(condition.get("name") or "CONDITION")
        message = str(condition.get("message") or "")
        err = condition.get("error")
        tree = condition.get("tree")
        if (err and err != "None") or not isinstance(tree, dict):
            return CheckView(name, "unknown", message or "unparsed condition")
        expression = str(condition.get("expression") or "")
        if "->" in expression:
            # '->' was rewritten to a top-level '&&' at parse time; treat that
            # top-level node as an implication (antecedent -> consequent).
            if tree.get("type") == "binary" and tree.get("op") == "&&":
                ls, lv = self.eval_tree(tree.get("left"), values)
                rs, rv = self.eval_tree(tree.get("right"), values)
                if ls != "ok" or rs != "ok":
                    return CheckView(name, "unknown", str(lv if ls != "ok" else rv))
                passed = (not _to_bool(lv)) or _to_bool(rv)
                return CheckView(name, "pass" if passed else "fail", message)
        status, value = self.eval_tree(tree, values)
        if status != "ok":
            return CheckView(name, "unknown", str(value))
        return CheckView(name, "pass" if _to_bool(value) else "fail", message)

    def checks_for(self, defn: InstructionClass,
                   values: Dict[str, int]) -> List[CheckView]:
        out = []
        for cond in defn.conditions:
            if not isinstance(cond, dict):
                continue
            if str(cond.get("name", "")).startswith("FORMAT "):
                continue
            out.append(self.check_condition(cond, values))
        return out

    # -- disambiguation ------------------------------------------------------

    def disambiguate(self, candidates: List[InstructionClass],
                     instr: bytes) -> Optional[InstructionClass]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        best, best_score = candidates[0], None
        for cand in candidates:
            values = self.values_for(cand, instr)
            score = 0
            for check in self.checks_for(cand, values):
                if check.status == "pass":
                    score += 1
                elif check.status == "fail":
                    score -= 4
            if best_score is None or score > best_score:
                best, best_score = cand, score
        return best

    # -- rendering -----------------------------------------------------------

    def _tokens(self, defn: InstructionClass) -> List[_Token]:
        if defn.name not in self._token_cache:
            raw = (defn.data.get("FORMAT") or {}).get("TOKENS") or []
            parsed = [t for t in map(_parse_token, raw) if t is not None]
            self._token_cache[defn.name] = parsed
        return self._token_cache[defn.name]

    def _default_value(self, token: _Token, bucket: Optional[str]) -> Optional[int]:
        if token.default is None:
            return None
        default = str(token.default)
        if bucket:
            b = self.bucket(bucket) or {}
            for k in (default, default.upper()):
                if k in b:
                    return _to_number(b[k])
        if "/" in default:  # "width/value" form, e.g. "4/0xf" or "24/0"
            default = default.rsplit("/", 1)[1]
        try:
            return _to_number(default)
        except Exception:
            return None

    def mnemonic_for(self, defn: InstructionClass, opcode: int) -> str:
        """Disassembler-style base mnemonic (shortest alias for this opcode)."""
        names = [alias for alias, val in defn.opcodes.items() if val == opcode]
        if not names:
            names = list(defn.opcodes.keys())
        if names:
            return min(names, key=len).upper()
        return defn.name.split("_")[0].upper() or defn.name

    def render(self, defn: InstructionClass, instr: bytes,
               values: Dict[str, int]) -> Tuple[str, str]:
        """Return (mnemonic-with-modifiers, operand-text)."""
        opcode = extract_opcode(instr)
        buckets = self.field_buckets(defn)
        mnemonic = self.mnemonic_for(defn, opcode)
        guard = ""
        operands: List[str] = []
        seen_opcode = False
        pending_sep = ""
        in_group = False  # inside a [ ... ] memory-operand group
        group_opened = False  # "[" has actually been emitted

        def emit(text: str) -> None:
            nonlocal pending_sep, group_opened
            if not text:
                return
            if in_group and not group_opened:
                operands.append(pending_sep + "[" + text)
                group_opened = True
            else:
                operands.append(pending_sep + text)
            pending_sep = ""

        for token in self._tokens(defn):
            if token.kind == "opcode":
                seen_opcode = True
                continue
            if token.kind == "guard":
                value = values.get(token.target)
                if value is None or value == PT:
                    continue
                bucket = buckets.get(token.target, "Predicate")
                pname = self.bucket_rev(bucket).get(value, str(value))
                inverted = token.optional_not and values.get(f"{token.target}_not")
                guard = f"@{'!' if inverted else ''}{pname} "
                continue
            if token.kind == "modifier":
                value = values.get(token.target)
                if value is None:
                    continue
                bucket = buckets.get(token.target) or (
                    token.source if self.bucket(token.source) else None)
                default = self._default_value(token, bucket)
                if default is not None and value == default:
                    continue
                text = self.bucket_rev(bucket).get(value) if bucket else None
                if text is not None:  # unmapped values would just be noise
                    mnemonic += "." + text
                continue
            if token.kind == "literal":
                lit = token.literal
                if lit == ",":
                    pending_sep = ", " if operands else ""
                elif lit == "[":
                    in_group, group_opened = True, False
                elif lit == "]":
                    if group_opened and operands:
                        operands[-1] += "]"
                    in_group = group_opened = False
                    pending_sep = ""
                elif lit == "+":
                    pending_sep = "+" if (not in_group or group_opened) else ""
                elif not seen_opcode:
                    continue
                else:
                    emit(lit)
                continue
            # operand
            if token.bracket_open:
                in_group, group_opened = True, False
            value = values.get(token.target)
            elided = value is None
            if not elided:
                bucket = buckets.get(token.target) or (
                    token.source if self.bucket(token.source) else None)
                default = self._default_value(token, bucket)
                if default is not None and value == default:
                    elided = True  # default operand (e.g. trailing PT, RZ base)
            if elided:
                if pending_sep.startswith(","):
                    pending_sep = ""
                if token.bracket_close:
                    if group_opened and operands:
                        operands[-1] += "]"
                    in_group = group_opened = False
                continue
            if bucket:
                text = self.bucket_rev(bucket).get(value)
                if text is None:
                    text = f"0x{value:X}"
            elif token.source == "SpecialRegister":
                text = self.special_register_name(value)
            elif "Imm" in token.source:
                width = spec_width((defn.arguments.get(token.target) or {}).get("bits"))
                text = _format_imm(value, width, signed=token.source.startswith("S"))
            else:
                text = f"0x{value:X}"
            not_bit = values.get(f"{token.target}_not")
            if token.optional_not and not_bit:
                text = "!" + text
            emit(text)
            if token.bracket_close:
                if group_opened and operands:
                    operands[-1] += "]"
                in_group = group_opened = False

        return guard + mnemonic, "".join(operands)

    # -- top level -----------------------------------------------------------

    def decode(self, offset: int, instr: bytes,
               override: Optional[str] = None) -> DecodedInstruction:
        opcode = extract_opcode(instr)
        candidates = self.spec.classes_for_opcode(opcode)
        defn = None
        if override:
            defn = next((c for c in candidates if c.name == override), None)
        if defn is None:
            defn = self.disambiguate(candidates, instr)
        d = DecodedInstruction(
            offset=offset, raw=bytes(instr), opcode=opcode, defn=defn,
            candidates=[c.name for c in candidates],
            control=read_control(instr))
        if defn is None:
            d.mnemonic = f"0x{opcode:04X}"
            d.operands = "(unknown opcode)"
            return d
        d.values = self.values_for(defn, instr)
        d.fields = self.field_views(defn, instr)
        d.checks = self.checks_for(defn, d.values)
        d.mnemonic, d.operands = self.render(defn, instr, d.values)
        return d


# ---------------------------------------------------------------------------
# Scalar coercions (match modifier.html toNumber/toBoolean)
# ---------------------------------------------------------------------------

def _to_number(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace("`", "").replace("'", "").replace('"', "")
        try:
            if text.lower().startswith("0x"):
                return int(text, 16)
            if text.lower().startswith("0b"):
                return int(text, 2)
            return int(text, 10)
        except ValueError:
            return 0
    return 0


def _parse_strict_int(text: str) -> Optional[int]:
    """Parse a numeric literal, or None if the text is not a number."""
    text = text.strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        if text.lower().startswith("0b"):
            return int(text, 2)
        return int(text, 10)
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _format_imm(value: int, width: int, signed: bool) -> str:
    if signed and width and value >= (1 << (width - 1)):
        value -= 1 << width
    if -10 < value < 10:
        return str(value)
    sign = "-" if value < 0 else ""
    return f"{sign}0x{abs(value):X}"

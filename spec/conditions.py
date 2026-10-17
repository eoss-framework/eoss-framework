"""Parser for Rosetta String constraint expressions.

Parses condition expressions from instruction CLASS definitions
(e.g. ``(Rd)!=`Register@RZ` && %MAX_REG_COUNT == 254``) into an
evaluable AST.

Important contract
------------------
The Rosetta String uses ``->`` for implication. The tokenizer rewrites
``->`` to ``&&`` at parse time (implication is *erased*), because inside
a boolean context that is usually what disassemblers do. Downstream
consumers that need true implication semantics (the encoding generator)
must re-interpret the *top-level* ``&&`` node of the resulting AST as an
implication. See ``eoss.enc.generator.evaluate_top_level_implication``.

This module is self-contained (stdlib only).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# AST node definitions
# ---------------------------------------------------------------------------


@dataclass
class Node:
    pass


@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node


@dataclass
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass
class Identifier(Node):
    name: str


@dataclass
class Literal(Node):
    value: str


@dataclass
class DefinedCall(Node):
    """Represents DEFINED TABLE_NAME(arg1, arg2, ...) expressions.

    These are table lookup validations that can't be evaluated statically.
    Also used to represent TABLECALL value lookups such as
    TABLES_mem_0(sem, sco, private).
    """

    table_name: str
    args: list  # list of argument name strings
    negated: bool = False  # True for !DEFINED


# ---------------------------------------------------------------------------
# Operator precedences and associativity
# ---------------------------------------------------------------------------

_PRECEDENCE = {
    '||': (1, 'L'),
    '&&': (2, 'L'),
    '|':  (2, 'L'),   # bitwise OR (same precedence as logical for simplicity)
    '&':  (2, 'L'),   # bitwise AND
    '==': (3, 'L'), '!=': (3, 'L'),
    '<':  (3, 'L'), '>':  (3, 'L'),
    '<=': (3, 'L'), '>=': (3, 'L'),
    '<<': (4, 'L'), '>>': (4, 'L'),  # bitwise shift
    '+':  (4, 'L'), '-':  (4, 'L'),
    '*':  (5, 'L'), '/':  (5, 'L'), '%':  (5, 'L'),
    '!':  (6, 'R'),
}


# ---------------------------------------------------------------------------
# Tokenizer (no external libraries, no recursion)
# ---------------------------------------------------------------------------

def tokenize(expr: str):
    # normalize implies: see module docstring for the implication contract
    expr = expr.replace('->', '&&')
    tokens = []
    i = 0
    length = len(expr)
    while i < length:
        c = expr[i]
        if c.isspace():
            i += 1
            continue

        # DEFINED / !DEFINED table lookup expressions
        # Match: DEFINED TABLE_NAME(arg1,arg2,...)
        # These are consumed as a single DEFINED token
        if expr[i:].startswith('DEFINED '):
            j = i + len('DEFINED ')
            # Skip whitespace
            while j < length and expr[j].isspace():
                j += 1
            # Read table name
            name_start = j
            while j < length and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            table_name = expr[name_start:j]
            # Read (arg1,arg2,...) if present
            args_str = ""
            if j < length and expr[j] == '(':
                paren_start = j
                depth = 1
                j += 1
                while j < length and depth > 0:
                    if expr[j] == '(':
                        depth += 1
                    elif expr[j] == ')':
                        depth -= 1
                    j += 1
                args_str = expr[paren_start:j]
            tokens.append(('DEFINED', f"{table_name}{args_str}"))
            i = j
            continue

        # If % followed by uppercase letter, it's a parameter name
        if c == '%' and i + 1 < length and expr[i + 1].isupper():
            j = i + 1
            # include letters, digits, underscore, @, ., and $ in the name
            while j < length and (expr[j].isalnum() or expr[j] in '_@.$'):
                j += 1
            tokens.append(('IDENT', expr[i:j]))
            i = j
            continue

        # If $ followed by uppercase letter, it's a constant name
        if c == '$' and i + 1 < length and expr[i + 1].isupper():
            j = i + 1
            # include letters, digits, underscore in the name
            while j < length and (expr[j].isalnum() or expr[j] in '_'):
                j += 1
            tokens.append(('IDENT', expr[i:j]))
            i = j
            continue

        # Parentheses
        if c == '(':
            tokens.append(('LPAR', c)); i += 1; continue
        if c == ')':
            tokens.append(('RPAR', c)); i += 1; continue

        # Two-character operators (check longest matches first)
        if expr.startswith('&&', i):
            tokens.append(('AND', '&&')); i += 2; continue
        if expr.startswith('||', i):
            tokens.append(('OR', '||')); i += 2; continue
        if expr.startswith('<<', i):
            tokens.append(('SHIFT', '<<')); i += 2; continue
        if expr.startswith('>>', i):
            tokens.append(('SHIFT', '>>')); i += 2; continue
        if expr.startswith('==', i) or expr.startswith('!=', i) or \
           expr.startswith('<=', i) or expr.startswith('>=', i):
            tokens.append(('COMP', expr[i:i + 2])); i += 2; continue

        # Single-character bitwise operators (after && and || are consumed)
        if c == '&':
            tokens.append(('BITAND', '&')); i += 1; continue
        if c == '|':
            tokens.append(('BITOR', '|')); i += 1; continue

        # Single-character operators (now % only if not part of a %VAR)
        if c in '!<>*/%+-':
            if c == '!':
                tokens.append(('NOT', c))
            elif c in '<>':
                tokens.append(('COMP', c))
            elif c in '*/%':
                tokens.append(('MULDIV', c))
            else:
                tokens.append(('ADDSUB', c))
            i += 1
            continue

        # Backtick-led identifier (enum reference like `Register@RZ).
        # Note: the Rosetta String uses a single LEADING backtick with no
        # closing backtick; the name runs until whitespace or a paren.
        if c == '`':
            j = i + 1
            while j < length and not expr[j].isspace() and expr[j] not in '()':
                j += 1
            name = expr[i + 1:j]
            tokens.append(('IDENT', name))
            i = j
            continue

        # Number literal (decimal, hex 0x..., binary 0b...)
        if c.isdigit():
            j = i
            if c == '0' and j + 1 < length and expr[j + 1] in 'xX':
                # Hex literal: 0x[0-9a-fA-F]+
                j += 2  # skip '0x'
                while j < length and (expr[j].isdigit() or expr[j] in 'abcdefABCDEF'):
                    j += 1
            elif c == '0' and j + 1 < length and expr[j + 1] in 'bB':
                # Binary literal: 0b[01]+
                j += 2  # skip '0b'
                while j < length and expr[j] in '01':
                    j += 1
            else:
                # Decimal literal
                while j < length and expr[j].isdigit():
                    j += 1
            tokens.append(('NUMBER', expr[i:j]))
            i = j
            continue

        # General identifier (letters, underscore, @, ., $)
        if c.isalpha() or c == '_':
            j = i
            while j < length and (expr[j].isalnum() or expr[j] in '_@.$'):
                j += 1
            ident_name = expr[i:j]
            # Check for function-call syntax: IDENT(arg1,arg2,...) with commas
            # This handles table lookups like TABLES_mem_0(sem,sco,private)
            if j < length and expr[j] == '(':
                # Peek ahead to see if there's a comma before the closing paren
                peek = j + 1
                depth = 1
                has_comma = False
                while peek < length and depth > 0:
                    if expr[peek] == '(':
                        depth += 1
                    elif expr[peek] == ')':
                        depth -= 1
                    elif expr[peek] == ',' and depth == 1:
                        has_comma = True
                    peek += 1
                if has_comma and depth == 0:
                    # This is a function call - consume IDENT(args) as TABLECALL
                    full_call = expr[i:peek]
                    tokens.append(('TABLECALL', full_call))
                    i = peek
                    continue
            tokens.append(('IDENT', ident_name))
            i = j
            continue

        # Anything else is unexpected
        raise ValueError(f"Unknown character '{c}' at position {i}")

    return tokens


# ---------------------------------------------------------------------------
# Shunting-yard to convert to RPN
# ---------------------------------------------------------------------------

def to_rpn(tokens):
    out = []
    stack = []
    for typ, val in tokens:
        if typ in ('IDENT', 'NUMBER', 'DEFINED', 'TABLECALL'):
            out.append((typ, val))
        elif typ in ('AND', 'OR', 'COMP', 'MULDIV', 'ADDSUB', 'NOT',
                     'BITAND', 'BITOR', 'SHIFT'):
            op = val
            prec, assoc = _PRECEDENCE[op]
            while stack and stack[-1] != '(':
                top = stack[-1]
                tprec, tassoc = _PRECEDENCE[top]
                if (assoc == 'L' and prec <= tprec) or (assoc == 'R' and prec < tprec):
                    out.append(('OP', stack.pop()))
                else:
                    break
            stack.append(op)
        elif typ == 'LPAR':
            stack.append('(')
        elif typ == 'RPAR':
            while stack and stack[-1] != '(':
                out.append(('OP', stack.pop()))
            if not stack:
                raise ValueError('Mismatched parentheses')
            stack.pop()
    while stack:
        op = stack.pop()
        if op == '(':
            raise ValueError('Mismatched parentheses')
        out.append(('OP', op))
    return out


# ---------------------------------------------------------------------------
# Build AST from RPN without recursion
# ---------------------------------------------------------------------------

def build_ast(rpn):
    stack = []
    for typ, val in rpn:
        if typ == 'IDENT':
            stack.append(Identifier(val))
        elif typ == 'NUMBER':
            stack.append(Literal(val))
        elif typ == 'DEFINED':
            # Parse table name and args from the value string
            if '(' in val:
                tbl_name = val[:val.index('(')]
                args_raw = val[val.index('(') + 1:val.rindex(')')]
                args_list = [a.strip() for a in args_raw.split(',')]
            else:
                tbl_name = val
                args_list = []
            stack.append(DefinedCall(table_name=tbl_name, args=args_list))
        elif typ == 'TABLECALL':
            # Table value lookup: TABLES_mem_0(sem,sco,private) -> returns a value
            # Represent as DefinedCall with a marker that it's a value reference
            if '(' in val:
                tbl_name = val[:val.index('(')]
                args_raw = val[val.index('(') + 1:val.rindex(')')]
                args_list = [a.strip() for a in args_raw.split(',')]
            else:
                tbl_name = val
                args_list = []
            stack.append(DefinedCall(table_name=tbl_name, args=args_list))
        elif typ == 'OP':
            if val == '!':
                if not stack:
                    raise ValueError('Error building AST: empty stack for unary op')
                operand = stack.pop()
                stack.append(UnaryOp('!', operand))
            else:
                if len(stack) < 2:
                    raise ValueError('Error building AST: empty stack for binary op')
                right = stack.pop()
                left = stack.pop()
                stack.append(BinaryOp(val, left, right))
    if len(stack) != 1:
        raise ValueError('Error building AST')
    return stack[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_condition(expr: str) -> Node:
    """Parse a Rosetta String condition expression into an AST."""
    tokens = tokenize(expr)
    rpn = to_rpn(tokens)
    return build_ast(rpn)


def print_ascii(node, prefix="", is_last=True):
    """Recursively prints an ASCII tree for a Node (debug helper)."""
    pointer = "└── " if is_last else "├── "

    if isinstance(node, BinaryOp):
        label = f"BinaryOp(op='{node.op}')"
        children = [node.left, node.right]
    elif isinstance(node, UnaryOp):
        label = f"UnaryOp(op='{node.op}')"
        children = [node.operand]
    elif isinstance(node, Identifier):
        label = f"Identifier(name='{node.name}')"
        children = []
    elif isinstance(node, Literal):
        label = f"Literal(value={node.value!r})"
        children = []
    elif isinstance(node, DefinedCall):
        neg = "!" if node.negated else ""
        label = f"{neg}DEFINED({node.table_name}({', '.join(node.args)}))"
        children = []
    else:
        label = repr(node)
        children = []

    print(prefix + pointer + label)

    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        is_child_last = (i == len(children) - 1)
        print_ascii(child, new_prefix, is_child_last)


try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


def draw_ast(node, filename='ast', fmt='pdf'):
    """Render the AST rooted at `node` to filename.(pdf|svg).

    Requires the optional ``graphviz`` package.
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("graphviz module is required for draw_ast function")
    dot = Digraph(comment='AST', format=fmt)

    def add(n):
        """Recursively add nodes/edges and return this node's unique ID."""
        nid = str(id(n))
        if nid in dot.body:
            return nid

        if isinstance(n, BinaryOp):
            label = f"{n.op}"
            children = [n.left, n.right]
        elif isinstance(n, UnaryOp):
            label = f"{n.op}"
            children = [n.operand]
        elif isinstance(n, Identifier):
            label = f"Id:{n.name}"
            children = []
        elif isinstance(n, Literal):
            label = f"Lit:{n.value}"
            children = []
        else:
            label = repr(n)
            children = []

        dot.node(nid, label, shape='box')
        for child in children:
            cid = add(child)
            dot.edge(nid, cid)
        return nid

    add(node)
    out_path = dot.render(filename)  # writes filename + .fmt and returns path
    print(f"AST rendered to: {out_path}")

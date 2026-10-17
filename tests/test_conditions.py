"""Unit tests for eoss.spec.conditions (constraint expression parser)."""

import pytest

from eoss.spec import conditions as C


class TestParse:
    def test_comparison(self):
        ast = C.parse_condition("%MAX_REG_COUNT == 254")
        assert isinstance(ast, C.BinaryOp)
        assert ast.op == "=="
        assert isinstance(ast.left, C.Identifier) and ast.left.name == "%MAX_REG_COUNT"
        assert isinstance(ast.right, C.Literal) and ast.right.value == "254"

    def test_logical_precedence(self):
        ast = C.parse_condition("a == 1 && b == 2 || c == 3")
        # || binds loosest -> top-level ||
        assert isinstance(ast, C.BinaryOp) and ast.op == "||"
        assert isinstance(ast.left, C.BinaryOp) and ast.left.op == "&&"

    def test_backtick_enum_identifier(self):
        # Rosetta format: single LEADING backtick, name runs to paren/space
        ast = C.parse_condition("((Rd) != `Register@RZ)")
        assert isinstance(ast, C.BinaryOp) and ast.op == "!="
        assert isinstance(ast.right, C.Identifier)
        assert ast.right.name == "Register@RZ"

    def test_implication_erased_to_and(self):
        """Documented contract: -> is rewritten to && at parse time."""
        ast = C.parse_condition("a == 1 -> b == 2")
        assert isinstance(ast, C.BinaryOp) and ast.op == "&&"

    def test_defined_call(self):
        ast = C.parse_condition("DEFINED TABLES_opex_1(batch_t,usched_info)")
        assert isinstance(ast, C.DefinedCall)
        assert ast.table_name == "TABLES_opex_1"
        assert ast.args == ["batch_t", "usched_info"]

    def test_tablecall(self):
        ast = C.parse_condition("x == TABLES_mem_0(sem,sco,private)")
        assert isinstance(ast, C.BinaryOp)
        assert isinstance(ast.right, C.DefinedCall)
        assert ast.right.table_name == "TABLES_mem_0"

    def test_hex_and_binary_literals(self):
        ast = C.parse_condition("a == 0x1F && b == 0b101")
        assert isinstance(ast, C.BinaryOp)
        assert ast.left.right.value == "0x1F"
        assert ast.right.right.value == "0b101"

    def test_unary_not(self):
        ast = C.parse_condition("!a")
        assert isinstance(ast, C.UnaryOp) and ast.op == "!"

    def test_real_fadd_condition(self):
        expr = ("(((Rd)==`Register@RZ`)||(((Rd)<=(%MAX_REG_COUNT-1))"
                "&&((Rd)!=`Register@R254`)))")
        ast = C.parse_condition(expr)
        assert isinstance(ast, C.BinaryOp) and ast.op == "||"

    def test_errors(self):
        with pytest.raises(ValueError):
            C.parse_condition("a == ==")

"""Tests for the Lean AST parser."""
import pytest
from leantree.repl_adapter.ast_parser import LeanAST, LeanASTObject, LeanASTLiteral, LeanASTArray


class TestParseBasic:
    def test_simple_literal(self):
        ast = LeanAST.parse_from_string('`y')
        assert isinstance(ast.root, LeanASTLiteral)
        assert ast.root.value == '`y'

    def test_backtick_literal(self):
        ast = LeanAST.parse_from_string('`H')
        assert isinstance(ast.root, LeanASTLiteral)
        assert ast.root.value == '`H'

    def test_string_literal(self):
        ast = LeanAST.parse_from_string('"rfl"')
        assert isinstance(ast.root, LeanASTLiteral)
        assert ast.root.value == '"rfl"'

    def test_simple_node(self):
        ast = LeanAST.parse_from_string('(Tactic.tacticRfl "rfl")')
        assert isinstance(ast.root, LeanASTObject)
        assert ast.root.type == 'Tactic.tacticRfl'
        assert len(ast.root.args) == 1
        assert isinstance(ast.root.args[0], LeanASTLiteral)
        assert ast.root.args[0].value == '"rfl"'

    def test_empty_array(self):
        ast = LeanAST.parse_from_string('(Foo [])')
        assert isinstance(ast.root, LeanASTObject)
        assert len(ast.root.args) == 1
        assert isinstance(ast.root.args[0], LeanASTArray)
        assert ast.root.args[0].items == []

    def test_array_with_items(self):
        ast = LeanAST.parse_from_string('(Foo [`x `y])')
        assert isinstance(ast.root, LeanASTObject)
        arr = ast.root.args[0]
        assert isinstance(arr, LeanASTArray)
        assert len(arr.items) == 2

    def test_anonymous_backtick(self):
        ast = LeanAST.parse_from_string('`[anonymous]')
        assert isinstance(ast.root, LeanASTLiteral)
        assert ast.root.value == '`[anonymous]'

    def test_guillemet_identifier(self):
        """Test identifiers with «» delimiters like «term(↑)»."""
        ast = LeanAST.parse_from_string('(Foo «term(↑)»)')
        assert isinstance(ast.root, LeanASTObject)
        assert isinstance(ast.root.args[0], LeanASTLiteral)
        assert ast.root.args[0].value == '«term(↑)»'

    def test_nested_nodes(self):
        ast = LeanAST.parse_from_string('(A (B `x) (C "y"))')
        assert isinstance(ast.root, LeanASTObject)
        assert ast.root.type == 'A'
        assert len(ast.root.args) == 2
        assert isinstance(ast.root.args[0], LeanASTObject)
        assert ast.root.args[0].type == 'B'

    def test_num_literal(self):
        ast = LeanAST.parse_from_string('(num "1")')
        assert isinstance(ast.root, LeanASTObject)
        assert ast.root.type == 'num'

    def test_bare_identifier(self):
        """Plain identifier without backtick."""
        ast = LeanAST.parse_from_string('omega')
        assert isinstance(ast.root, LeanASTLiteral)
        assert ast.root.value == 'omega'


class TestExistingExamples:
    """Test the example AST strings already defined in ast_parser.py."""

    def test_ast_str3(self):
        from leantree.repl_adapter.ast_parser import ast_str3
        ast = LeanAST.parse_from_string(ast_str3)
        assert isinstance(ast.root, LeanASTObject)

    def test_ast_str4(self):
        from leantree.repl_adapter.ast_parser import ast_str4
        ast = LeanAST.parse_from_string(ast_str4)
        assert isinstance(ast.root, LeanASTObject)

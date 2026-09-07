"""Language adapters for syntax evidence. No type-resolution/call-graph claims."""

import importlib
from functools import lru_cache

from tree_sitter import Language, Parser

DECLARATIONS = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "class_definition": "class",
    "class_declaration": "class",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "class_specifier": "class",
    "enum_item": "enum",
    "enum_specifier": "enum",
    "enum_declaration": "enum",
    "trait_item": "trait",
    "interface_declaration": "interface",
    "type_spec": "type",
    "type_alias_declaration": "type",
    "mod_item": "module",
    "namespace_definition": "namespace",
    "const_item": "constant",
    "impl_item": "implementation",
}
IMPORTS = {
    "import_statement",
    "import_from_statement",
    "import_declaration",
    "import_spec",
    "use_declaration",
    "preproc_include",
    "export_statement",
}
IDENTIFIERS = {"identifier", "field_identifier", "type_identifier", "property_identifier"}


@lru_cache(maxsize=10)
def language_adapter(language):
    grammar = "cpp" if language == "cuda" else language
    if grammar not in {"python", "rust", "go", "c", "cpp", "javascript", "typescript"}:
        return None
    module = importlib.import_module("tree_sitter_" + grammar)
    factory = module.language_typescript if grammar == "typescript" else module.language
    return Language(factory())


def parse(text, language):
    adapter = language_adapter(language)
    if adapter is None:
        return None
    raw = text.encode()
    tree = Parser(adapter).parse(raw)
    symbols, references, calls, imports, inheritance = [], [], [], [], []

    def content(node):
        return raw[node.start_byte : node.end_byte].decode(errors="replace") if node else ""

    def name_node(node):
        named = node.child_by_field_name("name")
        if node.type == "impl_item":
            named = node.child_by_field_name("type")
        if named:
            return named
        declarator = node.child_by_field_name("declarator")
        while declarator:
            if declarator.type in IDENTIFIERS:
                return declarator
            next_node = declarator.child_by_field_name("declarator")
            if not next_node:
                return next((c for c in declarator.named_children if c.type in IDENTIFIERS), None)
            declarator = next_node
        return None

    def visit(node, parent="", definitions=frozenset()):
        name = name_node(node) if node.type in DECLARATIONS else None
        owner = parent
        if name:
            short = content(name)
            owner = parent + "." + short if parent else short
            symbols.append(
                {
                    "name": owner,
                    "short_name": short,
                    "kind": DECLARATIONS[node.type],
                    "line": node.start_point.row + 1,
                    "column": node.start_point.column + 1,
                    "end_line": node.end_point.row + 1,
                    "end_column": node.end_point.column + 1,
                    "enclosing": parent or None,
                    "signature": content(node).split("\n", 1)[0][:500],
                    "quality": "syntax-derived",
                }
            )
            definitions = definitions | {name.id}
        if node.type in IMPORTS:
            source = node.child_by_field_name("source") or node.child_by_field_name("path")
            imports.append(
                {
                    "text": content(node)[:1000],
                    "module": content(source).strip("\"'<>"),
                    "line": node.start_point.row + 1,
                    "quality": "syntax-derived",
                }
            )
        if node.type in {"call", "call_expression"}:
            function = node.child_by_field_name("function")
            if function:
                calls.append(
                    {
                        "name": content(function),
                        "enclosing": owner,
                        "line": node.start_point.row + 1,
                        "quality": "syntax-derived",
                    }
                )
        if node.type in {
            "argument_list",
            "base_class_clause",
            "superclasses",
            "class_heritage",
        } and (node.type != "argument_list" or node.parent.type == "class_definition"):
            inheritance.append(
                {
                    "name": content(node)[:500],
                    "enclosing": owner,
                    "line": node.start_point.row + 1,
                    "quality": "syntax-derived",
                }
            )
        if node.type in IDENTIFIERS and node.id not in definitions:
            references.append(
                {
                    "name": content(node),
                    "enclosing": owner,
                    "line": node.start_point.row + 1,
                    "column": node.start_point.column + 1,
                    "quality": "syntax-derived",
                }
            )
        for child in node.named_children:
            visit(child, owner, definitions)

    visit(tree.root_node)
    return {
        "symbols": symbols,
        "references": references,
        "calls": calls,
        "imports": imports,
        "inheritance": inheritance,
        "parser": "tree_sitter_" + ("cpp" if language == "cuda" else language),
        "dialect": "cuda_cpp_syntax_subset" if language == "cuda" else language,
        "parse_error": tree.root_node.has_error,
        "quality": "syntax-derived",
    }

"""Syntax-derived module bindings. Dynamic import expressions remain unresolved."""

from tree_sitter import Parser

from .syntax_index import language_adapter


def imports(text, language):
    adapter = language_adapter(language)
    if adapter is None:
        return []
    raw = text.encode()
    tree = Parser(adapter).parse(raw)
    result = []

    def value(node):
        return raw[node.start_byte : node.end_byte].decode() if node else ""

    def field(node, key):
        return value(node.child_by_field_name(key))

    def rust(node, prefix=""):
        if node is None or node.has_error:
            return
        if node.type == "scoped_use_list":
            scope = prefix + field(node, "path") + "::"
            for child in node.child_by_field_name("list").named_children:
                rust(child, scope)
        elif node.type == "use_list":
            for child in node.named_children:
                rust(child, prefix)
        else:
            name = field(node, "path") if node.type == "use_as_clause" else value(node)
            full = prefix + name
            module, _, symbol = full.rpartition("::")
            if symbol == "self":
                result.append((module, field(node, "alias") or module.rsplit("::", 1)[-1], ""))
            else:
                result.append((module, field(node, "alias") or symbol, symbol))

    def javascript(node):
        module = field(node, "source").strip("\"'")
        if not module or node.has_error:
            return
        found = []

        def names(child):
            if child.type in {"import_specifier", "export_specifier"}:
                name = field(child, "name")
                found.append((module, field(child, "alias") or name, name))
            elif child.type == "namespace_import":
                found.append((module, value(child.named_children[-1]), ""))
            elif child.type == "identifier" and child.parent.type == "import_clause":
                found.append((module, value(child), "default"))
            else:
                for item in child.named_children:
                    names(item)

        names(node)
        result.extend(found or [(module, "", "*")])

    def visit(node):
        if language in {"javascript", "typescript"} and node.type in {
            "import_statement",
            "export_statement",
        }:
            javascript(node)
            return
        if language == "rust":
            if node.type == "use_declaration":
                rust(node.child_by_field_name("argument"))
                return
            if node.type == "mod_item" and node.child_by_field_name("body") is None:
                name = field(node, "name")
                result.append((name, name, ""))
        if language == "go" and node.type == "import_spec" and not node.has_error:
            module = field(node, "path").strip('"`')
            result.append((module, field(node, "name") or module.rsplit("/", 1)[-1], ""))
            return
        if language in {"c", "cpp", "cuda"} and node.type == "preproc_include":
            path = node.child_by_field_name("path")
            if path and path.type in {"string_literal", "system_lib_string"}:
                result.append((value(path).strip('"<>'), "", "*"))
            return
        for child in node.named_children:
            visit(child)

    visit(tree.root_node)
    return result

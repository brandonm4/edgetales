import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_engine_helpers():
    source = Path("/Volumes/Data/Development/projects/edgetales/engine.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    wanted = {"_clean_model_output_text", "_response_text"}
    snippets = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            snippets.append(ast.get_source_segment(source, node))
    namespace = {"re": re}
    exec("\n\n".join(snippets), namespace)
    return namespace["_response_text"]


_response_text = _load_engine_helpers()


class EngineOutputCleaningTests(unittest.TestCase):
    def test_response_text_strips_think_blocks(self):
        response = SimpleNamespace(
            output_text="<think>hidden</think>Hello world",
            output=[],
        )
        self.assertEqual(_response_text(response), "Hello world")

    def test_response_text_strips_incomplete_html_comment_tail(self):
        response = SimpleNamespace(
            output_text="Visible narration<!--",
            output=[],
        )
        self.assertEqual(_response_text(response), "Visible narration")

    def test_response_text_preserves_game_data_tags(self):
        response = SimpleNamespace(
            output_text="Narration\n<game_data>{\"ok\":true}</game_data>",
            output=[],
        )
        self.assertEqual(_response_text(response), "Narration\n<game_data>{\"ok\":true}</game_data>")


if __name__ == "__main__":
    unittest.main()

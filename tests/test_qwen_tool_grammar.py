# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-style XML tool-call grammar processor."""

from __future__ import annotations

import pytest


class TestBuildXmlInnerGrammar:
    """Unit tests for the LLGuidance grammar builder."""

    def test_single_tool_grammar(self):
        from omlx.api.qwen_tool_grammar import build_xml_inner_grammar

        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                    },
                }
            }
        ]
        grammar = build_xml_inner_grammar(tools)
        assert "<function=get_weather>" in grammar
        assert "<parameter=location>" in grammar
        assert "<parameter=unit>" in grammar
        assert "PARAM_VALUE" in grammar

    def test_object_parameter_uses_json_constraint(self):
        from omlx.api.qwen_tool_grammar import build_xml_inner_grammar

        tools = [
            {
                "function": {
                    "name": "send_email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "object",
                                "properties": {"subject": {"type": "string"}},
                            }
                        },
                    },
                }
            }
        ]
        grammar = build_xml_inner_grammar(tools)
        assert '"type":"object"' in grammar.replace(" ", "")

    def test_multiple_tools_alternatives(self):
        from omlx.api.qwen_tool_grammar import build_xml_inner_grammar

        tools = [
            {"function": {"name": "a", "parameters": {"type": "object", "properties": {}}}},
            {"function": {"name": "b", "parameters": {"type": "object", "properties": {}}}},
        ]
        grammar = build_xml_inner_grammar(tools)
        assert "func_a" in grammar
        assert "func_b" in grammar
        assert "function: func_a | func_b" in grammar


class TestQwenXmlToolGrammarProcessor:
    """Tests for the logits processor itself."""

    def test_factory_returns_none_without_tool_markers(self):
        from omlx.api.qwen_tool_grammar import build_qwen_tool_grammar_processor

        class FakeTokenizer:
            vocab_size = 100

            def encode(self, text, add_special_tokens=False):
                return []

        assert build_qwen_tool_grammar_processor(FakeTokenizer(), []) is None

    def test_processor_transitions_on_tool_call_markers(self, monkeypatch):
        import llguidance

        from omlx.api.qwen_tool_grammar import QwenXmlToolGrammarProcessor

        class FakeMatcher:
            def __init__(self, *args, **kwargs):
                self.tokens = []

            def accept_token(self, token_id):
                self.tokens.append(token_id)
                return True

            def is_terminated(self):
                return False

            def fill_next_token_bitmask(self, bitmask):
                bitmask.fill(-1)

        monkeypatch.setattr(llguidance, "LLMatcher", FakeMatcher)

        proc = QwenXmlToolGrammarProcessor(
            llg_tokenizer=None,
            tools=[{"function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
            tool_call_start_id=10,
            tool_call_end_id=11,
            vocab_size=32,
        )

        assert not proc._in_tool_call
        proc.accept_token(10)
        assert proc._in_tool_call
        assert isinstance(proc._matcher, FakeMatcher)
        proc.accept_token(11)
        assert not proc._in_tool_call
        assert proc._matcher is None

        # Accept a regular token while inside the tool call
        fake = FakeMatcher()
        proc._in_tool_call = True
        proc._matcher = fake
        proc.accept_token(42)
        assert fake.tokens == [42]
        assert proc._in_tool_call

    def test_call_is_noop_outside_tool_call(self):
        import mlx.core as mx

        from omlx.api.qwen_tool_grammar import QwenXmlToolGrammarProcessor

        proc = QwenXmlToolGrammarProcessor(
            llg_tokenizer=None,
            tools=[{"function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
            tool_call_start_id=10,
            tool_call_end_id=11,
            vocab_size=32,
        )
        logits = mx.zeros((1, 32))
        result = proc(mx.array([1, 2, 3]), logits)
        assert result is logits

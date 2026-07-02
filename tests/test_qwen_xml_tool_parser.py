# SPDX-License-Identifier: Apache-2.0
"""Tests for the robust Qwen XML tool-call parser."""

import json

from omlx.api.tool_calling import _parse_qwen_xml_tool_calls_robust


def _edit_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string"},
                                "newText": {"type": "string"},
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    }


def test_simple_edit_call():
    tools = [_edit_tool_schema()]
    text = (
        "<tool_call>"
        '<function=edit>'
        '<parameter=path>/tmp/foo.py</parameter>'
        '<parameter=edits>[{"oldText":"a","newText":"b"}]</parameter>'
        '</function>'
        '</tool_call>'
    )
    cleaned, calls = _parse_qwen_xml_tool_calls_robust(text, tools)
    assert cleaned == ""
    assert len(calls) == 1
    args = json.loads(calls[0].function.arguments)
    assert args["path"] == "/tmp/foo.py"
    assert args["edits"] == [{"oldText": "a", "newText": "b"}]


def test_edit_call_with_closing_tag_inside_json_string():
    """The closing </parameter> tag inside a JSON string must not truncate."""
    tools = [_edit_tool_schema()]
    text = (
        "<tool_call>"
        '<function=edit>'
        '<parameter=path>/tmp/foo.py</parameter>'
        '<parameter=edits>[{"oldText":"prefix </parameter> suffix","newText":"replaced"}]</parameter>'
        '</function>'
        '</tool_call>'
    )
    cleaned, calls = _parse_qwen_xml_tool_calls_robust(text, tools)
    assert cleaned == ""
    assert len(calls) == 1
    args = json.loads(calls[0].function.arguments)
    assert args["edits"][0]["oldText"] == "prefix </parameter> suffix"


def test_edit_call_with_multiple_closing_tags_inside_value():
    tools = [_edit_tool_schema()]
    text = (
        "<tool_call>"
        '<function=edit>'
        '<parameter=path>/tmp/foo.py</parameter>'
        '<parameter=edits>[{"oldText":"a </parameter> b </parameter> c","newText":"x"}]</parameter>'
        '</function>'
        '</tool_call>'
    )
    cleaned, calls = _parse_qwen_xml_tool_calls_robust(text, tools)
    assert len(calls) == 1
    args = json.loads(calls[0].function.arguments)
    assert args["edits"][0]["oldText"] == "a </parameter> b </parameter> c"


def test_call_with_whitespace_and_newlines():
    tools = [_edit_tool_schema()]
    text = (
        "<tool_call>\n"
        '<function=edit>\n'
        '<parameter=path>\n/tmp/foo.py\n</parameter>\n'
        '<parameter=edits>\n[{"oldText":"a","newText":"b"}]\n</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    cleaned, calls = _parse_qwen_xml_tool_calls_robust(text, tools)
    assert len(calls) == 1
    args = json.loads(calls[0].function.arguments)
    assert args["path"] == "/tmp/foo.py"
    assert args["edits"] == [{"oldText": "a", "newText": "b"}]


def test_no_tool_call_returns_none():
    cleaned, calls = _parse_qwen_xml_tool_calls_robust("just some text", [])
    assert calls is None
    assert cleaned == "just some text"

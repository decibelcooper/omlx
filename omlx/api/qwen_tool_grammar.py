# SPDX-License-Identifier: Apache-2.0
"""Qwen3-style XML tool-call grammar via LLGuidance.

This module provides a logits processor that constrains generation only
inside ``<tool_call>...</tool_call>`` markers, matching the format used by
Qwen3 / Qwen3.5 / Qwen3.6 tool-calling models.  It is the oMLX port of the
constrained-decoding approach from ``qwen-perfect``.
"""

import json
import logging
from typing import Any, List, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def _param_value_grammar(schema: dict) -> str:
    """Grammar fragment for a parameter value based on its JSON schema."""
    ptype = schema.get("type")
    if isinstance(ptype, list):
        ptype = ptype[0] if ptype else None

    if ptype in ("object", "array"):
        schema_json = json.dumps(schema, separators=(",", ":"))
        return f"%json {schema_json}"

    # Unconstrained raw text (stops at closing </parameter> tag)
    return "PARAM_VALUE"


def build_xml_inner_grammar(tools: List[dict]) -> str:
    """Build a Lark grammar for the inner content of <tool_call>...</tool_call>."""
    func_alts = []
    func_rules = []

    for t in tools:
        fname = t["function"]["name"]
        props = t["function"]["parameters"].get("properties", {})

        param_rule_name = f"param_{fname}"
        param_alts = " | ".join(
            f'"<parameter={p}>" /\\n/? {_param_value_grammar(props[p])} /\\n/? "</parameter>" /\\n/?'
            for p in props.keys()
        )

        func_alts.append(f"func_{fname}")
        func_rules.append(
            f'func_{fname}: "<function={fname}>" /\\n/? {param_rule_name}* /\\n/? "</function>"'
        )
        func_rules.append(f"{param_rule_name}: {param_alts}")

    func_branch = " | ".join(func_alts)
    func_rules_str = "\n".join(func_rules)

    grammar = f"""
?start: /\\n/? function /\\n/?

function: {func_branch}

{func_rules_str}

PARAM_VALUE: /[^<]+/
"""
    return grammar


class QwenXmlToolGrammarProcessor:
    """Logits processor that applies an XML tool grammar inside <tool_call>."""

    def __init__(
        self,
        llg_tokenizer: Any,
        tools: List[dict],
        tool_call_start_id: int,
        tool_call_end_id: int,
        vocab_size: int,
    ):
        self.llg_tokenizer = llg_tokenizer
        self.tools = tools
        self.grammar = build_xml_inner_grammar(tools)
        self.tool_call_start_id = tool_call_start_id
        self.tool_call_end_id = tool_call_end_id
        self.vocab_size = vocab_size

        self._matcher: Optional[Any] = None
        self._in_tool_call = False
        self._terminated = False

        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)

    def accept_token(self, token_id: int) -> None:
        """Advance the grammar matcher after a token has been sampled."""
        if self._terminated:
            return

        if token_id == self.tool_call_start_id:
            self._in_tool_call = True
            try:
                import llguidance

                self._matcher = llguidance.LLMatcher(self.llg_tokenizer, self.grammar)
            except Exception as e:
                logger.warning("Failed to create LLGuidance matcher: %s", e)
                self._in_tool_call = False
                self._matcher = None
        elif token_id == self.tool_call_end_id:
            self._in_tool_call = False
            self._matcher = None
        elif self._in_tool_call and self._matcher is not None:
            try:
                if not self._matcher.accept_token(token_id):
                    logger.warning("LLGuidance matcher rejected token %d", token_id)
                if self._matcher.is_terminated():
                    self._terminated = True
            except Exception as e:
                logger.warning("LLGuidance accept_token failed: %s", e)
                self._terminated = True

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Fill and apply the grammar bitmask for the next token."""
        if self._terminated or not self._in_tool_call or self._matcher is None:
            return logits

        # Preserve the original shape so we can return logits with the same
        # number of dimensions the caller passed in.
        original_shape = logits.shape
        if logits.ndim == 1:
            logits = logits[None, :]

        self._bitmask.fill(-1)
        try:
            self._matcher.fill_next_token_bitmask(self._bitmask)
        except Exception as e:
            logger.warning("LLGuidance fill_next_token_bitmask failed: %s", e)
            return logits

        try:
            import llguidance.mlx

            masked = llguidance.mlx.apply_token_bitmask(logits, self._bitmask[0])
            masked = mx.array(masked)
            # If the caller passed a 1D logits vector, apply_token_bitmask
            # expanded it to 2D; squeeze back to match the original shape.
            if len(original_shape) == 1 and masked.ndim > 1:
                masked = masked[0]
            return masked
        except Exception as e:
            logger.warning("LLGuidance apply_token_bitmask failed: %s", e)
            return logits


def build_qwen_tool_grammar_processor(
    tokenizer: Any,
    tools: List[dict],
    vocab_size: Optional[int] = None,
) -> Optional[QwenXmlToolGrammarProcessor]:
    """Factory: create a Qwen XML tool grammar processor if possible.

    Args:
        tokenizer: mlx-lm tokenizer (or HF tokenizer).
        tools: OpenAI-style tool definitions.
        vocab_size: Optional vocab size; inferred from tokenizer if absent.

    Returns:
        A configured processor, or None if LLGuidance is unavailable or the
        tokenizer does not advertise the required tool-call markers.
    """
    try:
        import llguidance.hf
    except Exception as e:
        logger.warning("LLGuidance not available for Qwen tool grammar: %s", e)
        return None

    from ..utils.tokenizer import unwrap_tokenizer

    hf_tokenizer = unwrap_tokenizer(tokenizer)
    try:
        llg_tokenizer = llguidance.hf.from_tokenizer(hf_tokenizer)
    except Exception as e:
        logger.warning("Failed to create LLGuidance tokenizer: %s", e)
        return None

    # Resolve tool-call marker token ids from the tokenizer
    try:
        start_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
        end_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    except Exception as e:
        logger.warning("Tokenizer missing Qwen tool-call markers: %s", e)
        return None

    if not start_ids or not end_ids:
        logger.warning("Tokenizer missing Qwen tool-call marker token ids")
        return None

    if vocab_size is None:
        vocab_size = getattr(hf_tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(hf_tokenizer)
        except Exception:
            logger.warning("Could not determine tokenizer vocab_size")
            return None

    return QwenXmlToolGrammarProcessor(
        llg_tokenizer=llg_tokenizer,
        tools=tools,
        tool_call_start_id=int(start_ids[0]),
        tool_call_end_id=int(end_ids[0]),
        vocab_size=int(vocab_size),
    )

"""
OPRO (Optimization by PROmpting) — prompt and hook optimization for marketing.

OptimizerEntry  — shared data structure for both OPRO and OPRO2
Opro2           — optimizes marketing hooks directly
Opro            — optimizes LLM instruction prompts (wraps Opro2 when use_opro2=True)
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.models import CustomerSegment
from services.llm_service import LLMService
from agents.hooks import init_hooks, init_llm_instruction_prompts, segment_descriptions


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class OptimizerEntry:
    """
    Unified data structure for OPRO/OPRO2 optimization entries.

    accuracy_score represents the actual visit rate for the cycle (0.0 to 1.0).
    prompt is used by the original OPRO approach; leave as None for OPRO2.
    """
    accuracy_score: float
    hook: str
    prompt: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.accuracy_score <= 1.0:
            raise ValueError("Accuracy score must be between 0.0 and 1.0")
        if not self.hook.strip():
            raise ValueError("Hook cannot be empty")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("Prompt cannot be empty when provided")


# Backward-compatibility alias
OptimizerEntry2 = OptimizerEntry


# ---------------------------------------------------------------------------
# Opro2 — direct hook optimization
# ---------------------------------------------------------------------------

class Opro2:
    """
    Directly optimizes marketing hooks using meta-prompting.
    Unlike Opro, which optimizes the instruction prompts that generate hooks,
    Opro2 optimizes the hooks themselves.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def optimize_hook2(self, k: int, customer_segment: str,
                       optimization_data: List[OptimizerEntry], logging_config=None) -> str:
        if not optimization_data:
            return self._get_default_hook2(customer_segment)
        sorted_entries = sorted(optimization_data, key=lambda x: x.accuracy_score, reverse=True)
        top_performers = sorted_entries[:min(k, len(sorted_entries))]
        return self._generate_simple_optimized_hook2(customer_segment, top_performers, logging_config)

    def _get_default_hook2(self, customer_segment: str) -> str:
        return init_hooks.get(customer_segment, init_hooks[CustomerSegment.HIGH_VALUE_CUSTOMERS.value])

    def _generate_simple_optimized_hook2(self, customer_segment: str,
                                          top_performers: List[OptimizerEntry],
                                          logging_config=None) -> str:
        if not top_performers:
            return self._get_default_hook2(customer_segment)
        meta_prompt = self.create_meta_prompt(customer_segment, top_performers)
        try:
            raw_response = self.llm_service.generate_text(meta_prompt)
            optimized_hook = self._clean_llm_response2(raw_response)
            if not self._is_valid_hook2(optimized_hook):
                return self._get_best_performing_hook2(customer_segment, top_performers)
            return optimized_hook
        except Exception:
            return self._get_best_performing_hook2(customer_segment, top_performers)

    def create_meta_prompt(self, customer_segment: str, top_performers: List[OptimizerEntry]) -> str:
        raise NotImplementedError("create_meta_prompt method in class Opro2 has not been implemented yet.")

    def _clean_llm_response2(self, raw_response: str) -> str:
        cleaned = raw_response.strip()
        cleaned = cleaned.replace('```markdown', '').replace('```', '')
        cleaned = cleaned.replace('**', '').replace('*', '')
        for prefix in ['Hook:', 'Subject:', 'Marketing Hook:', 'Email Hook:', 'Hook Text:']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*', '#')):
                parts = line.split('.', 1) if '.' in line else line.split(' ', 1)
                if len(parts) > 1:
                    content = parts[1].strip().strip('"')
                    if content and len(content) > 10:
                        filtered_lines.append(content)
            else:
                filtered_lines.append(line.strip('"'))
        result = ' '.join(filtered_lines).strip()
        if len(result) > 200:
            for sentence in result.split('.'):
                sentence = sentence.strip()
                if 20 < len(sentence) < 150:
                    result = sentence + '.'
                    break
        return self._remove_emojis(result)

    def _remove_emojis(self, text: str) -> str:
        cleaned = ''.join(c for c in text if ord(c) <= 255)
        return ' '.join(cleaned.split())

    def _is_valid_hook2(self, hook: str) -> bool:
        if not hook or len(hook) < 10 or len(hook) > 200:
            return False
        return not any(p in hook for p in ['```', 'markdown', 'Example:', 'TASK:', 'CONTEXT:', '\n\n', 'Hook:', 'Subject:'])

    def _get_best_performing_hook2(self, customer_segment: str, top_performers: List[OptimizerEntry]) -> str:
        if not top_performers:
            return self._get_default_hook2(customer_segment)
        return max(top_performers, key=lambda x: x.accuracy_score).hook

    def _extract_optimization_patterns2(self, top_performers: List[OptimizerEntry], customer_segment: str) -> Dict[str, Any]:
        patterns: Dict[str, Any] = {
            'avg_accuracy': sum(e.accuracy_score for e in top_performers) / len(top_performers),
            'common_hook_keywords': [],
            'hook_length_range': [],
            'hook_characteristics': [],
        }
        for entry in top_performers:
            patterns['hook_length_range'].append(len(entry.hook))
            patterns['common_hook_keywords'].extend(entry.hook.lower().split())
            if '!' in entry.hook:
                patterns['hook_characteristics'].append('exclamation')
            if any(e in entry.hook for e in ['🌟', '✨', '💎']):
                patterns['hook_characteristics'].append('premium_emojis')
            if 'exclusive' in entry.hook.lower():
                patterns['hook_characteristics'].append('exclusivity')
            if 'limited' in entry.hook.lower() or 'time' in entry.hook.lower():
                patterns['hook_characteristics'].append('urgency')
            if 'free' in entry.hook.lower():
                patterns['hook_characteristics'].append('free_offer')
            if any(w in entry.hook.lower() for w in ['save', 'discount', 'deal', 'offer']):
                patterns['hook_characteristics'].append('value_proposition')
        return patterns

    def _generate_optimized_hook_with_patterns2(self, customer_segment: str,
                                                  patterns: Dict[str, Any],
                                                  top_performers: List[OptimizerEntry]) -> str:
        if patterns['avg_accuracy'] > 0.5:
            return top_performers[0].hook
        return self._get_default_hook2(customer_segment)


# ---------------------------------------------------------------------------
# Opro — prompt optimization (wraps Opro2 when use_opro2=True)
# ---------------------------------------------------------------------------

class Opro:
    """
    Supports both original prompt optimization and direct hook optimization (OPRO2).

    use_opro2=False  — optimizes LLM instruction prompts (original approach)
    use_opro2=True   — delegates to Opro2 for direct hook optimization
    """

    def __init__(self, llm_service: LLMService, use_opro2: bool = False):
        self.llm_service = llm_service
        self.use_opro2 = use_opro2
        if self.use_opro2:
            self.opro2 = Opro2(llm_service)
            self._hook_cache: Dict[str, str] = {}
        else:
            self._prompt_cache: Dict[str, str] = {}

    def optimize_prompt(self, k: int, customer_segment: str,
                        optimization_data: List[OptimizerEntry], logging_config=None) -> str:
        if self.use_opro2:
            return self._optimize_with_opro2(k, customer_segment, optimization_data, logging_config)
        return self._optimize_with_original_opro(k, customer_segment, optimization_data, logging_config)

    def _optimize_with_opro2(self, k: int, customer_segment: str,
                              optimization_data: List[OptimizerEntry], logging_config=None) -> str:
        if not optimization_data:
            return self._get_default_prompt(customer_segment)
        opro2_data = [OptimizerEntry(accuracy_score=e.accuracy_score, hook=e.hook)
                      for e in optimization_data]
        optimized_hook = self.opro2.optimize_hook2(k, customer_segment, opro2_data, logging_config)
        self._hook_cache[customer_segment] = optimized_hook
        return f"OPRO2_HOOK:{optimized_hook}"

    def _optimize_with_original_opro(self, k: int, customer_segment: str,
                                      optimization_data: List[OptimizerEntry], logging_config=None) -> str:
        if not optimization_data:
            return self._get_default_prompt(customer_segment)
        sorted_entries = sorted(optimization_data, key=lambda x: x.accuracy_score, reverse=True)
        top_performers = sorted_entries[:min(k, len(sorted_entries))]
        optimized_prompt = self._generate_optimized_prompt(customer_segment, top_performers, logging_config)
        self._prompt_cache[customer_segment] = optimized_prompt
        return optimized_prompt

    def _generate_optimized_prompt(self, customer_segment: str,
                                    top_performers: List[OptimizerEntry], logging_config=None) -> str:
        if not top_performers:
            return self._get_default_prompt(customer_segment)
        meta_prompt = self._create_prompt_optimization_meta_prompt(customer_segment, top_performers)
        if logging_config and getattr(logging_config, 'opro_meta_prompt', False):
            self._print_meta_prompt(meta_prompt, customer_segment)
        try:
            raw_response = self.llm_service.generate_text(meta_prompt)
            optimized_prompt = self._clean_prompt_response(raw_response)
            if not self._is_valid_prompt(optimized_prompt):
                return self._get_best_performing_prompt(customer_segment, top_performers)
            return optimized_prompt
        except Exception as e:
            if logging_config and getattr(logging_config, 'opro_fallbacks', False):
                print(f"Original OPRO optimization failed for segment {customer_segment}: {e}")
            return self._get_best_performing_prompt(customer_segment, top_performers)

    def _create_prompt_optimization_meta_prompt(self, customer_segment: str,
                                                 top_performers: List[OptimizerEntry]) -> str:
        segment_desc = segment_descriptions.get(customer_segment, "general customers")
        examples_text = ""
        for i, entry in enumerate(top_performers, 1):
            examples_text += f"""
Example {i} (Visit Rate: {entry.accuracy_score:.3f}):
Instruction Prompt: "{entry.prompt}"
Generated Hook: "{entry.hook}"
"""
        return f"""You are an expert prompt engineer. Analyze the successful instruction prompts below and create ONE improved instruction prompt for generating marketing hooks.

TARGET: {segment_desc}
BEST VISIT RATE: {top_performers[0].accuracy_score:.3f} ({top_performers[0].accuracy_score*100:.1f}%)

SUCCESSFUL EXAMPLES:
{examples_text}

CRITICAL INSTRUCTIONS:
1. Write ONLY a single instruction prompt for an LLM to generate marketing hooks
2. Do NOT include "Prompt:", "Instruction:", or any prefixes
3. Do NOT use markdown, bullets, or numbering
4. Do NOT include explanations or examples
5. Do NOT use code blocks or formatting
6. Focus on what made the successful prompts effective
7. The prompt should instruct the LLM to generate a single marketing hook
8. Include guidance about format, tone, and key elements to include

Write the improved instruction prompt now:"""

    def _print_meta_prompt(self, meta_prompt: str, customer_segment: str) -> None:
        print("\n" + "="*80)
        print(f"OPRO META-PROMPT - Segment: {customer_segment}")
        print("="*80)
        print(meta_prompt)
        print("="*80 + "\n")

    def _clean_prompt_response(self, raw_response: str) -> str:
        cleaned = raw_response.strip()
        cleaned = cleaned.replace('```markdown', '').replace('```', '')
        cleaned = cleaned.replace('**', '').replace('*', '')
        for prefix in ['Prompt:', 'Instruction:', 'Optimized Prompt:', 'New Prompt:']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*', '#')):
                parts = line.split('.', 1) if '.' in line else line.split(' ', 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    if content and len(content) > 20:
                        filtered_lines.append(content)
            else:
                filtered_lines.append(line)
        return ' '.join(filtered_lines).strip()

    def _is_valid_prompt(self, prompt: str) -> bool:
        if not prompt or len(prompt) < 50:
            return False
        return not any(p in prompt for p in ['```', 'markdown', 'Example:', 'TASK:', 'CONTEXT:', '\n\n\n'])

    def _get_best_performing_prompt(self, customer_segment: str, top_performers: List[OptimizerEntry]) -> str:
        if not top_performers:
            return self._get_default_prompt(customer_segment)
        return max(top_performers, key=lambda x: x.accuracy_score).prompt

    def _get_default_prompt(self, customer_segment: str) -> str:
        if self.use_opro2:
            hook = init_hooks.get(customer_segment, init_hooks[CustomerSegment.HIGH_VALUE_CUSTOMERS.value])
            return f"OPRO2_HOOK:{hook}"
        return init_llm_instruction_prompts.get(
            customer_segment,
            init_llm_instruction_prompts[CustomerSegment.HIGH_VALUE_CUSTOMERS.value]
        )

    def get_cached_hook(self, customer_segment: str) -> str:
        return self._hook_cache.get(customer_segment, "") if self.use_opro2 else ""

    def get_cached_prompt(self, customer_segment: str) -> str:
        return self._prompt_cache.get(customer_segment, "") if not self.use_opro2 else ""

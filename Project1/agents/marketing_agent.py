from typing import List, Dict
from dataclasses import dataclass

from agents.opro import Opro, OptimizerEntry
from agents.hooks import init_hooks
from models.registries import ItemCatalogue
from services.llm_service import LLMService
from models.models import CustomerSegment


@dataclass
class MarketingEmail:
    """
    Represents a customized promotion email consisting of a marketing hook 
    and contextualized item recommendations.
    
    Based on the document specification: e = (h, r)
    where h is the marketing hook and r is the contextualized recommendation.
    """
    
    # Marketing hook optimized for specific customer segment
    marketing_hook: str
    
    # Contextualized recommendation of items (typically R=3 items)
    contextualized_recommendation: str
    
    # List of recommended item IDs that were used to generate the recommendation text
    recommended_item_ids: List[int]
    
    # Target customer segment (N, H, R, L, A)
    target_segment: str
    
    # Customer ID this email is intended for
    customer_id: int
    
    # LLM instruction prompt used to generate the marketing hook
    llm_hook_instruction: str = ""

    def get_full_email_content(self) -> str:
        """
        Generate the complete email content combining hook and recommendations.
        
        Returns:
            str: Full email content as it would be sent to customer
        """
        return f"{self.marketing_hook}\n\n{self.contextualized_recommendation}"
    
    
    def get_item_count(self) -> int:
        """
        Get the number of recommended items in this email.
        
        Returns:
            int: Number of recommended items
        """
        return len(self.recommended_item_ids)


class MarketingAgent:
    def __init__(self, item_catalogue: ItemCatalogue, llm_service: LLMService, use_opro2: bool = False):
        self.item_catalogue = item_catalogue
        self.llm_service = llm_service
        self.opro: Opro = Opro(llm_service, use_opro2=use_opro2)

    def createMessage(self, client, email_hook=None, customer_segment=None, llm_hook_instruction=None) -> MarketingEmail:
        """Create personalized marketing message for a client."""
        # For now, create a basic MarketingEmail object
        # In a real implementation, this would use LLM and customer segmentation
        
        customer_id = client.cid

        # Use provided customer segment or determine it
        target_segment = customer_segment

        # Use provided email hook or default marketing hook
        marketing_hook = email_hook or "Check out our amazing deals just for you!"
        contextualized_recommendation = "We think you'll love these handpicked items from our collection."
        recommended_item_ids = [1, 2, 3]  # Placeholder - would come from recommendation system
        
        return MarketingEmail(
            marketing_hook=marketing_hook,
            contextualized_recommendation=contextualized_recommendation,
            recommended_item_ids=recommended_item_ids,
            target_segment=target_segment,
            customer_id=customer_id,
            llm_hook_instruction=llm_hook_instruction
        )

    def generate_hooks_for_segments(self, optimized_prompts: Dict[str, str]) -> Dict[str, str]:
        """
        Generate marketing hooks for each customer segment using optimized prompts.
        
        Supports both original OPRO and OPRO2 approaches:
        - Original OPRO: Uses LLM instruction prompts to generate hooks
        - OPRO2: Detects magic strings "OPRO2_HOOK:" and extracts pre-optimized hooks

        Args:
            optimized_prompts: Dictionary mapping segment to optimized LLM prompt or OPRO2 magic string

        Returns:
            Dict[str, str]: Dictionary mapping segment to generated hook
        """
        current_hooks = {}

        for segment, prompt in optimized_prompts.items():
            try:
                # Check if this is an OPRO2 magic string containing a pre-optimized hook
                if prompt.startswith("OPRO2_HOOK:"):
                    # Extract the hook directly from the magic string
                    hook = prompt[len("OPRO2_HOOK:"):].strip()
                    if hook:
                        current_hooks[segment] = hook
                    else:
                        # Fallback if magic string is malformed
                        current_hooks[segment] = self._get_fallback_hook(segment)
                else:
                    # Original OPRO approach: Generate hook using LLM with the optimized prompt
                    llm_response = self.llm_service.generate_text(prompt)

                    # Extract first line as the hook (or split and take first if multiple)
                    hooks = [hook.strip() for hook in llm_response.split('\n') if hook.strip()]
                    if hooks:
                        current_hooks[segment] = hooks[0]
                    else:
                        # Fallback hook if LLM fails
                        current_hooks[segment] = self._get_fallback_hook(segment)

            except Exception as e:
                print(f"Warning: Hook generation failed for segment {segment}: {e}")
                # Use fallback hook on failure
                current_hooks[segment] = self._get_fallback_hook(segment)

        return current_hooks

    def _get_fallback_hook(self, segment: str) -> str:
        """
        Get fallback hook when LLM generation fails.

        Args:
            segment: Customer segment (N, H, R, L, A)

        Returns:
            str: Fallback hook for the segment
        """
        fallback_hooks = init_hooks.copy()
        return fallback_hooks.get(segment, "Personalized Just For You: Special Offers Inside")

    def apply_opro_step(self, previous_scores: Dict[str, float], previous_prompts: Dict[str, str], previous_hooks: Dict[str, str], k: int = 3, logging_config=None) -> Dict[str, str]:
        """
        Apply OPRO optimization step for each customer segment.

        Args:
            previous_scores: Dictionary mapping segment to performance score (0.0-1.0)
            previous_prompts: Dictionary mapping segment to previous optimal prompt
            previous_hooks: Dictionary mapping segment to hooks generated from previous prompts
            k: Number of top-performing entries to consider for optimization
            logging_config: LoggingConfig object for controlling debug output

        Returns:
            Dict[str, str]: New optimal prompt dictionary mapping segment to optimized prompt
        """
        new_optimal_prompts = {}

        # Apply OPRO optimization for each customer segment
        for cust_seg in CustomerSegment:
            segment = cust_seg.value
            # Create optimization data entry from previous results
            optimization_data = []

            if (segment in previous_scores and
                segment in previous_prompts and
                segment in previous_hooks):
                # Create OptimizerEntry with previous performance data including the generated hook
                entry = OptimizerEntry(
                    accuracy_score=previous_scores[segment],
                    prompt=previous_prompts[segment],
                    hook=previous_hooks[segment]
                )
                optimization_data.append(entry)

            # Apply OPRO optimization to get new optimal prompt
            optimized_prompt = self.opro.optimize_prompt(
                k=k,
                customer_segment=segment,
                optimization_data=optimization_data,
                logging_config=logging_config
            )

            new_optimal_prompts[segment] = optimized_prompt

        return new_optimal_prompts

    def _determine_customer_segment(self, client) -> str:
        """
        Placeholder function to determine customer segment.
        Currently returns the same segment for all customers.
        
        Args:
            client: Customer data dictionary
            
        Returns:
            str: Customer segment (N, H, R, L, A)
        """
        # Placeholder - always return High Value Customers segment
        return CustomerSegment.HIGH_VALUE_CUSTOMERS.value

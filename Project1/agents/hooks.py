"""
Marketing hook data and quality evaluation utilities.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Union, Any

from models.models import CustomerSegment


# ---------------------------------------------------------------------------
# Static hook / prompt data
# ---------------------------------------------------------------------------

optimal_hooks = {
    CustomerSegment.NEWCOMERS.value: "Welcome aboard! Discover the quality our customers love and enjoy a special welcome discount on your very first order!",
    CustomerSegment.HIGH_VALUE_CUSTOMERS.value: "Your orders deserve a reward! As a top customer, receive a complimentary gift with your next qualifying purchase. It's our way of saying thank you!",
    CustomerSegment.REGULAR_CUSTOMERS.value: "Just for our regulars! See what's new and trending this week. A special thank-you discount is waiting for you at checkout for your continued support!",
    CustomerSegment.LOW_VALUE_CUSTOMERS.value: "Great finds, even better value! Explore our affordable best-sellers, and unlock free shipping when you build a bundle. Perfect for stocking up on your favorites!",
    CustomerSegment.AT_RISK_CUSTOMERS.value: "It's been a while, and we've missed you! Come see what's new, and accept a special welcome-back credit to use on your next purchase, on us!",
}

init_hooks = {
    CustomerSegment.NEWCOMERS.value: "Welcome! Your Personal Shopping Journey Starts Here",
    CustomerSegment.HIGH_VALUE_CUSTOMERS.value: "Exclusive VIP Preview: Your Premium Collection Awaits Inside",
    CustomerSegment.REGULAR_CUSTOMERS.value: "Thank You for Being with Us! Enjoy Exclusive Deals Inside",
    CustomerSegment.LOW_VALUE_CUSTOMERS.value: "Special Savings Just for You! Don't Miss Out on These Deals",
    CustomerSegment.AT_RISK_CUSTOMERS.value: "We Miss You! Here's What You've Been Missing + Special Return Offer",
}

init_llm_instruction_prompts = {
    CustomerSegment.NEWCOMERS.value: "Create a single welcoming email marketing hook for new customers. Emphasize discovery, first-time benefits, and introductory offers. Use friendly, encouraging language to build trust.",
    CustomerSegment.HIGH_VALUE_CUSTOMERS.value: "Generate a compelling email marketing hook for high-value customers. You should make these customers feel recognized for their high spending to reinforce their loyalty and encourage continued large purchases.",
    CustomerSegment.REGULAR_CUSTOMERS.value: "Craft an engaging email marketing hook for regular customers. Highlight appreciation and new arrivals to maintain their shopping habit and build a stronger relationship.",
    CustomerSegment.LOW_VALUE_CUSTOMERS.value: "Develop an enticing email marketing hook for budget-conscious customers. Emphasize value, affordability, and special promotions. Use persuasive language to encourage purchases.",
    CustomerSegment.AT_RISK_CUSTOMERS.value: "Develop a re-engagement email marketing hook for customers who haven't purchased recently. Focus on 'we miss you' messaging, comeback offers, and give them compelling reasons to return.",
}

fallback_hooks = {
    CustomerSegment.NEWCOMERS.value: [
        "Welcome! Your Personal Shopping Journey Starts Here 🎉",
        "New Here? Discover Why Thousands Love Shopping With Us",
        "Your Welcome Gift Is Ready - Plus Our Most Popular Picks!"
    ],
    CustomerSegment.HIGH_VALUE_CUSTOMERS.value: [
        "🌟 Exclusive VIP Preview: Your Premium Collection Awaits Inside",
        "As Our Valued VIP: First Access to Limited Edition Items",
        "Your Loyalty Deserves More: Exclusive Member-Only Deals Inside"
    ],
    CustomerSegment.REGULAR_CUSTOMERS.value: [
        "Personalized Just For You: Special Offers Inside",
        "Don't Miss Out: Your Curated Selection Awaits",
        "Exclusive Deals Matching Your Style"
    ],
    CustomerSegment.LOW_VALUE_CUSTOMERS.value: [
        "Special Savings Just for You! Don't Miss Out on These Deals",
        "Unlock Extra Discounts on Your Favorite Finds Today",
        "Budget-Friendly Picks: Quality Products at Unbeatable Prices"
    ],
    CustomerSegment.AT_RISK_CUSTOMERS.value: [
        "We Miss You! Here's What You've Been Missing + Special Return Offer",
        "Remember Us? Your Favorites Are Back & Better Than Ever",
        "Welcome Back! Exclusive Comeback Deal Just For You Inside"
    ],
}

segment_descriptions = {
    CustomerSegment.NEWCOMERS.value: "new customers who have never purchased before",
    CustomerSegment.HIGH_VALUE_CUSTOMERS.value: "customers who make frequent large purchases",
    CustomerSegment.REGULAR_CUSTOMERS.value: "regular customers who occasionally make purchases",
    CustomerSegment.AT_RISK_CUSTOMERS.value: "regular customers who haven't purchased recently or show low satisfaction",
    CustomerSegment.LOW_VALUE_CUSTOMERS.value: "customers who make purchases with a low average value",
}


# ---------------------------------------------------------------------------
# Hook quality evaluation
# ---------------------------------------------------------------------------

class HookEvaluator:
    """
    Evaluates marketing hook quality using cosine similarity with optimal hooks.
    Based on Section 4.2.1 of the AINIT project specification.
    """

    def __init__(self, optimal_hooks: Dict[str, str], model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.optimal_hooks = optimal_hooks.copy()
        if "Default" not in self.optimal_hooks:
            self.optimal_hooks["Default"] = "Personalized Just For You: Special Offers Inside"
        self._optimal_embeddings: Dict[str, np.ndarray] = {}
        self._compute_optimal_embeddings()

    def _to_numpy(self, embedding: Any) -> np.ndarray:
        if isinstance(embedding, np.ndarray):
            return embedding
        elif hasattr(embedding, 'numpy'):
            return embedding.numpy()
        return np.array(embedding)

    def _compute_optimal_embeddings(self) -> None:
        for segment, hook in self.optimal_hooks.items():
            embedding = self.encoder.encode(hook, normalize_embeddings=True)
            self._optimal_embeddings[segment] = self._to_numpy(embedding)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return max(0.0, float(np.dot(vec1_norm, vec2_norm)))

    def encode_hook(self, hook: str) -> np.ndarray:
        if not hook or not hook.strip():
            dimension = self.encoder.get_sentence_embedding_dimension()
            return np.zeros(dimension if dimension is not None else 384)
        try:
            embedding = self.encoder.encode(hook.strip(), normalize_embeddings=True)
            return self._to_numpy(embedding)
        except Exception as e:
            print(f"Error encoding hook: {e}")
            dimension = self.encoder.get_sentence_embedding_dimension()
            return np.zeros(dimension if dimension is not None else 384)

    def evaluate_hook_quality_with_embedding(self, hook_embedding: np.ndarray, customer_segment: str) -> float:
        try:
            optimal_embedding = self._optimal_embeddings.get(
                customer_segment, self._optimal_embeddings["Default"]
            )
            return float(self.cosine_similarity(hook_embedding, optimal_embedding))
        except Exception as e:
            print(f"Error evaluating hook quality with pre-computed embedding: {e}")
            return 0.0

    def evaluate_hook_quality(self, hook: str, customer_segment: str) -> float:
        """Evaluate hook quality using cosine similarity (eq. 4, Section 4.2.1)."""
        if not hook or not hook.strip():
            return 0.0
        try:
            hook_embedding = self._to_numpy(
                self.encoder.encode(hook.strip(), normalize_embeddings=True)
            )
            return self.evaluate_hook_quality_with_embedding(hook_embedding, customer_segment)
        except Exception as e:
            print(f"Error evaluating hook quality: {e}")
            return 0.0

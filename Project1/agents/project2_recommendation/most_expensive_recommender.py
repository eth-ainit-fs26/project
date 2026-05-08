from __future__ import annotations

from typing import List


class MostExpensiveRecommenderAgent:
    """Baseline recommender that always returns the highest-margin items."""

    def __init__(self, customer_registry, item_catalogue, transaction_registry):
        self.customer_registry = customer_registry
        self.item_catalogue = item_catalogue
        self.transaction_registry = transaction_registry

    def recommend(self, customer_id: int, k: int = 10) -> List[int]:
        items = self.item_catalogue.get_available_items()
        ranked = sorted(items, key=lambda item: item.price - item.cost, reverse=True)
        return [item.pid for item in ranked[: max(0, min(k, len(ranked)))]]

    def _kappa_hat(self, customer_id: int, item_id: int) -> float:
        return 0.5

    def update_from_session(self, customer_id: int, shown_items: List[int], purchased_items: List[int]) -> None:
        return None

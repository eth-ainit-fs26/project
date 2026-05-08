from __future__ import annotations

from typing import List

import numpy as np


class RandomRecommenderAgent:
    """Baseline recommender that samples available items uniformly."""

    def __init__(self, customer_registry, item_catalogue, transaction_registry, seed: int = 42):
        self.customer_registry = customer_registry
        self.item_catalogue = item_catalogue
        self.transaction_registry = transaction_registry
        self.rng = np.random.default_rng(seed)

    def recommend(self, customer_id: int, k: int = 10) -> List[int]:
        items = self.item_catalogue.get_available_items()
        if not items:
            return []
        item_ids = [item.pid for item in items]
        sample_size = max(0, min(k, len(item_ids)))
        return self.rng.choice(item_ids, size=sample_size, replace=False).tolist()

    def _kappa_hat(self, customer_id: int, item_id: int) -> float:
        return 0.5

    def update_from_session(self, customer_id: int, shown_items: List[int], purchased_items: List[int]) -> None:
        return None

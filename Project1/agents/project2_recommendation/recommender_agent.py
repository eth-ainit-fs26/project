from __future__ import annotations

import math
from typing import List

import numpy as np


class RecommenderAgent:
    """Collaborative filtering recommender used by the part 2 notebook.

    The notebook asks students to implement and monkey-patch the four core
    methods below. The constructor and initialization helpers live here so the
    exercise can focus on the recommender logic.
    """

    def __init__(
        self,
        customer_registry,
        item_catalogue,
        transaction_registry,
        d: int = 8,
        lr: float = 0.01,
        reg: float = 1e-4,
        w_purchase: float = 1.0,
        w_neg: float = 0.25,
        seed: int = 42,
    ):
        self.customer_registry = customer_registry
        self.item_catalogue = item_catalogue
        self.transaction_registry = transaction_registry
        self.d = int(d)
        self.lr = float(lr)
        self.reg = float(reg)
        self.w_purchase = float(w_purchase)
        self.w_neg = float(w_neg)

        self.U: dict[int, np.ndarray] = {}
        self.V: dict[int, np.ndarray] = {}
        self.b_c: dict[int, float] = {}
        self.b_i: dict[int, float] = {}

    def _init_customer(self, customer_id: int) -> None:
        if customer_id not in self.U:
            self.U[customer_id] = np.random.uniform(0, 1, self.d)
            self.b_c[customer_id] = 0.0

    def _init_item(self, item_id: int) -> None:
        if item_id not in self.V:
            self.V[item_id] = np.random.uniform(0, 1, self.d)
            self.b_i[item_id] = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _kappa_hat(self, customer_id: int, item_id: int) -> float:
        if customer_id not in self.U:
            self._init_customer(customer_id)
        if item_id not in self.V:
            self._init_item(item_id)

        z = float(np.dot(self.U[customer_id], self.V[item_id]) + self.b_c[customer_id] + self.b_i[item_id])
        return self._sigmoid(z)

    def recommend(self, customer_id: int, k: int = 10) -> List[int]:
        inventory = self.item_catalogue.get_available_items()
        scored_items = []
        for item in inventory:
            expected_margin = (item.price - item.cost) * self._kappa_hat(customer_id, item.pid)
            scored_items.append((item.pid, expected_margin))

        scored_items.sort(key=lambda pair: pair[1], reverse=True)
        return [item_id for item_id, _ in scored_items[: max(0, min(k, len(scored_items)))]]

    def update_from_session(self, customer_id: int, shown_items: List[int], purchased_items: List[int]) -> None:
        for item_id in purchased_items:
            self._update_embeddings(customer_id, item_id, target=1.0, weight=self.w_purchase)
        for item_id in shown_items:
            if item_id not in purchased_items:
                self._update_embeddings(customer_id, item_id, target=0.0, weight=self.w_neg)

    def _update_embeddings(self, customer_id: int, item_id: int, target: float, weight: float) -> None:
        self._init_customer(customer_id)
        self._init_item(item_id)

        prediction = self._kappa_hat(customer_id, item_id)
        err = target - prediction

        # Update U first, then V uses the already-updated U
        self.U[customer_id] += self.lr * (weight * err * self.V[item_id] - self.reg * self.U[customer_id])
        self.V[item_id] += self.lr * (weight * err * self.U[customer_id] - self.reg * self.V[item_id])
        self.b_c[customer_id] += self.lr * (weight * err - self.reg * self.b_c[customer_id])
        self.b_i[item_id] += self.lr * (weight * err - self.reg * self.b_i[item_id])

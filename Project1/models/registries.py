"""
Data-access registries and catalogues for all simulation models.
"""
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from models.models import (
    Customer, CustomerSegment, Item,
    MarketingOperation, ShopVisit, Transaction,
)


# ---------------------------------------------------------------------------
# CustomerRegistry
# ---------------------------------------------------------------------------

class CustomerRegistry:
    """Manages customers in the database."""

    def add_customer(self, customer_id: int, name: str, age: int, gender: str,
                     location: tuple, job_type: str, income: int,
                     family_status: str, preference_vector: np.ndarray,
                     preference_covariance: np.ndarray, visit_probability: float,
                     session_probability: float, trigger_keywords: List[str],
                     segment: str, satisfaction: float = 0.5) -> Customer:
        return Customer.create(
            cid=customer_id, name=name, age=age, gender=gender,
            location_x=location[0], location_y=location[1],
            job_type=job_type, income=income, segment=segment,
            family_status=family_status, satisfaction=satisfaction,
            feature_vector=preference_vector.tolist(),
            preference_covariance=preference_covariance.tolist() if preference_covariance is not None else None,
            visit_probability=visit_probability,
            session_probability=session_probability,
            trigger_keywords=trigger_keywords,
        )

    def get_customer(self, customer_id: int) -> Optional[Customer]:
        try:
            return Customer.get(Customer.cid == customer_id)
        except Customer.DoesNotExist:
            return None

    def get_all_customers(self) -> List[Customer]:
        return list(Customer.select())

    def update_satisfaction(self, customer_id: int, new_satisfaction: float):
        try:
            customer = Customer.get(Customer.cid == customer_id)
            customer.satisfaction = new_satisfaction
            customer.save()
        except Customer.DoesNotExist:
            raise ValueError(f"Customer {customer_id} not found")

    def get_customer_transactions(self, customer_id: int) -> List[Transaction]:
        return list(Transaction.select().where(Transaction.cid == customer_id))

    def get_customer_marketing_history(self, customer_id: int) -> List[MarketingOperation]:
        return list(MarketingOperation.select().where(MarketingOperation.cid == customer_id))

    def search_customers(self, **criteria) -> List[Customer]:
        query = Customer.select()
        for key, value in criteria.items():
            if key == 'age':
                query = query.where(Customer.age == value)
            elif key == 'income':
                query = query.where(Customer.income == value)
            elif key == 'gender':
                query = query.where(Customer.gender == value)
            elif key == 'job_type':
                query = query.where(Customer.job_type == value)
            elif key == 'family_status':
                query = query.where(Customer.family_status == value)
            elif key == 'segment':
                query = query.where(Customer.segment == value)
            elif key == 'name':
                query = query.where(Customer.name.contains(value))
        return list(query)

    def get_customers_by_segment(self, segment: CustomerSegment) -> List[Customer]:
        return list(Customer.select().where(Customer.segment == segment.value))

    def update_customer_preferences(self, customer_id: int, new_preferences: np.ndarray):
        try:
            customer = Customer.get(Customer.cid == customer_id)
            customer.feature_vector = new_preferences.tolist()
            customer.save()
        except Customer.DoesNotExist:
            raise ValueError(f"Customer {customer_id} not found")


# ---------------------------------------------------------------------------
# ItemCatalogue
# ---------------------------------------------------------------------------

class ItemCatalogue:
    """Manages shop items in the database."""

    def add_item(self, item_id: int, name: str, description: str,
                 price: float, cost: float, stock_level: int,
                 feature_vector: np.ndarray, discounted: bool = False) -> Item:
        return Item.create(
            pid=item_id, name=name, description=description,
            price=price, discounted=discounted, cost=cost,
            stock_level=stock_level, feature_vector=feature_vector.tolist(),
        )

    def get_item(self, item_id: int) -> Optional[Item]:
        try:
            return Item.get(Item.pid == item_id)
        except Item.DoesNotExist:
            return None

    def get_all_items(self) -> List[Item]:
        return list(Item.select())

    def get_available_items(self) -> List[Item]:
        return list(Item.select().where(Item.stock_level > 0))

    def search_items(self, query: str) -> List[Item]:
        return list(Item.select().where(
            (Item.name.contains(query)) | (Item.description.contains(query))
        ))

    def update_stock(self, item_id: int, new_stock_level: int):
        try:
            item = Item.get(Item.pid == item_id)
            item.stock_level = new_stock_level
            item.save()
        except Item.DoesNotExist:
            raise ValueError(f"Item {item_id} not found")

    def update_price(self, item_id: int, new_price: float, discounted: bool = False):
        try:
            item = Item.get(Item.pid == item_id)
            item.price = new_price
            item.discounted = discounted
            item.save()
        except Item.DoesNotExist:
            raise ValueError(f"Item {item_id} not found")

    def update_item_features(self, item_id: int, new_features: np.ndarray):
        try:
            item = Item.get(Item.pid == item_id)
            item.feature_vector = new_features.tolist()
            item.save()
        except Item.DoesNotExist:
            raise ValueError(f"Item {item_id} not found")


# ---------------------------------------------------------------------------
# TransactionRegistry
# ---------------------------------------------------------------------------

class TransactionRegistry:
    """Manages transactions in the database."""

    def add_transaction(self, transaction: Transaction) -> Transaction:
        transaction.save()
        return transaction

    def create_transaction(self, tid: int, cid: int, transaction_date: date,
                           order: List[tuple], delivery_time_window: tuple) -> Transaction:
        return Transaction.create(
            tid=tid, cid=cid, date=transaction_date,
            order_data=order,
            delivery_time_window_start=delivery_time_window[0],
            delivery_time_window_end=delivery_time_window[1],
        )

    def get_transaction(self, transaction_id: int) -> Optional[Transaction]:
        try:
            return Transaction.get(Transaction.tid == transaction_id)
        except Transaction.DoesNotExist:
            return None

    def get_all_transactions(self) -> List[Transaction]:
        return list(Transaction.select())

    def get_customer_transactions(self, customer_id: int) -> List[Transaction]:
        return list(Transaction.select().where(Transaction.cid == customer_id))

    def get_customer_transaction_count(self, customer_id: int) -> int:
        return Transaction.select().where(Transaction.cid == customer_id).count()

    def get_customer_total_spent(self, customer_id: int) -> float:
        total = 0.0
        for t in Transaction.select().where(Transaction.cid == customer_id):
            for pid, quantity, price in (t.order_data or []):
                total += quantity * price
        return total

    def get_customer_average_order_value(self, customer_id: int) -> float:
        count = self.get_customer_transaction_count(customer_id)
        return self.get_customer_total_spent(customer_id) / count if count else 0.0

    def get_customer_last_transaction_day(self, customer_id: int) -> Optional[date]:
        try:
            return (Transaction.select()
                    .where(Transaction.cid == customer_id)
                    .order_by(Transaction.date.desc())
                    .get()).date
        except Transaction.DoesNotExist:
            return None

    def get_customer_purchase_frequency(self, customer_id: int, days_back: int = 365) -> float:
        count = self.get_customer_transaction_count(customer_id)
        return count / days_back if count else 0.0

    def get_customer_purchased_items(self, customer_id: int) -> List[int]:
        purchased = set()
        for t in Transaction.select().where(Transaction.cid == customer_id):
            for pid, quantity, price in (t.order_data or []):
                purchased.add(pid)
        return list(purchased)

    def get_customer_item_quantities(self, customer_id: int) -> Dict[int, int]:
        quantities: Dict[int, int] = {}
        for t in Transaction.select().where(Transaction.cid == customer_id):
            for pid, quantity, price in (t.order_data or []):
                quantities[pid] = quantities.get(pid, 0) + quantity
        return quantities

    def get_transactions_by_date_range(self, start_date: date, end_date: date) -> List[Transaction]:
        return list(Transaction.select()
                    .where((Transaction.date >= start_date) & (Transaction.date <= end_date))
                    .order_by(Transaction.date))


# ---------------------------------------------------------------------------
# ShopVisitRegistry
# ---------------------------------------------------------------------------

class ShopVisitRegistry:
    """Manages shop visit records in the database."""

    def save_shop_visit(self, visit: ShopVisit) -> ShopVisit:
        visit.save()
        return visit

    def create_shop_visit(self, cid: int, day_number: int) -> ShopVisit:
        return ShopVisit.create(cid=cid, day_number=day_number)

    def get_all_visits(self) -> List[ShopVisit]:
        return list(ShopVisit.select())

    def get_visits_by_day(self, day_number: int) -> List[ShopVisit]:
        return list(ShopVisit.select().where(ShopVisit.day_number == day_number))

    def get_customer_visits(self, client_id: int) -> List[ShopVisit]:
        return list(ShopVisit.select().where(ShopVisit.cid == client_id))

    def get_customer_visits_by_day_range(self, client_id: int, start_day: int, end_day: int) -> List[ShopVisit]:
        return list(ShopVisit.select().where(
            (ShopVisit.cid == client_id) &
            (ShopVisit.day_number >= start_day) &
            (ShopVisit.day_number <= end_day)
        ))

    def get_visit_count_by_day(self, day_number: int) -> int:
        return ShopVisit.select().where(ShopVisit.day_number == day_number).count()

    def get_unique_visitors_by_day(self, day_number: int) -> int:
        return (ShopVisit.select(ShopVisit.cid)
                .where(ShopVisit.day_number == day_number)
                .distinct()).count()


shop_visit_registry = ShopVisitRegistry()


# ---------------------------------------------------------------------------
# MarketingOperationRegistry
# ---------------------------------------------------------------------------

class MarketingOperationRegistry:
    """Manages marketing operations (in-memory store)."""

    def __init__(self):
        self.operations: List[MarketingOperation] = []
        self._next_mid = 1

    def save_marketing_operation(self, operation: MarketingOperation) -> int:
        if operation.mid == 0 or operation.mid is None:
            operation.mid = self._next_mid
            self._next_mid += 1
        self.operations.append(operation)
        return operation.mid

    def get_marketing_operation(self, mid: int) -> Optional[MarketingOperation]:
        return next((op for op in self.operations if op.mid == mid), None)

    def get_customer_marketing_operations(self, customer_id: int) -> List[MarketingOperation]:
        return [op for op in self.operations if op.cid == customer_id]

    def get_operations_by_simulation_day(self, simulation_day: int) -> List[MarketingOperation]:
        return [op for op in self.operations if op.simulation_day == simulation_day]

    def get_all_operations(self) -> List[MarketingOperation]:
        return self.operations.copy()

    def clear_all_operations(self):
        self.operations.clear()
        self._next_mid = 1


marketing_operation_registry = MarketingOperationRegistry()

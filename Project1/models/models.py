"""
Unified model definitions for the retail simulation database.

All Peewee ORM models live here. Individual model files
(customer_model.py, item_model.py, etc.) re-export from this module
for backward compatibility.
"""
from datetime import date, timedelta
from enum import Enum

from peewee import (
    AutoField, BooleanField, CharField, DateField,
    FloatField, IntegerField, TextField,
)
from models.database import BaseModel, JSONField


_SIMULATION_START = date(2024, 10, 1)


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------

class CustomerSegment(Enum):
    NEWCOMERS = "N"
    HIGH_VALUE_CUSTOMERS = "H"
    REGULAR_CUSTOMERS = "R"
    LOW_VALUE_CUSTOMERS = "L"
    AT_RISK_CUSTOMERS = "A"


class Customer(BaseModel):
    """Customer model with observable and hidden parameters."""

    cid = IntegerField(primary_key=True)
    name = CharField()
    age = IntegerField()
    gender = CharField()
    location_x = FloatField()
    location_y = FloatField()
    job_type = CharField()
    income = IntegerField()
    segment = CharField()
    family_status = CharField()
    satisfaction = FloatField(default=0.5)
    feature_vector = JSONField()              # pc ∈ R^D: preference vector
    preference_covariance = JSONField(null=True)  # Σc ∈ R^(D×D)
    visit_probability = FloatField(default=0.5)
    session_probability = FloatField(default=0.3)
    trigger_keywords = JSONField(default=list)

    class Meta:
        table_name = 'customers'

    @property
    def location(self) -> tuple[float, float]:
        return (self.location_x, self.location_y)

    @location.setter
    def location(self, value: tuple[float, float]):
        self.location_x, self.location_y = value


# ---------------------------------------------------------------------------
# Item
# ---------------------------------------------------------------------------

class Item(BaseModel):
    """Item model with observable and hidden feature parameters."""

    pid = IntegerField(primary_key=True)
    name = CharField()
    description = TextField()
    price = FloatField()
    discounted = BooleanField(default=False)
    cost = FloatField()
    stock_level = IntegerField(default=0)
    feature_vector = JSONField()              # fi ∈ R^D: item feature vector

    class Meta:
        table_name = 'products'


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """Transaction model (Figure 3 Database schema)."""

    tid = IntegerField(primary_key=True)
    cid = IntegerField()                      # Reference to Customer.cid
    date = DateField()
    order_data = JSONField()                  # List[(PID, quantity, price)]
    delivery_time_window_start = IntegerField()
    delivery_time_window_end = IntegerField()

    class Meta:
        table_name = 'transactions'


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

class Delivery(BaseModel):
    """Delivery model (Figure 3 Database schema)."""

    did = IntegerField(primary_key=True)
    date = DateField()
    customers = JSONField()                   # List[CID]
    routes = JSONField()                      # List[List[CID]]
    actual_delivery_times = JSONField()       # List[timestamp]

    class Meta:
        table_name = 'deliveries'

    @property
    def simulation_day(self) -> int:
        if isinstance(self.date, date):
            return (self.date - _SIMULATION_START).days + 1
        return 1

    @simulation_day.setter
    def simulation_day(self, value: int):
        self.date = _SIMULATION_START + timedelta(days=value - 1)


# ---------------------------------------------------------------------------
# ShopVisit
# ---------------------------------------------------------------------------

class ShopVisit(BaseModel):
    """Shop visit model tracking customer visits per simulation day."""

    vid = AutoField(primary_key=True)
    cid = IntegerField(index=True)
    day_number = IntegerField(index=True)

    class Meta:
        table_name = 'shop_visits'

    @property
    def simulation_day(self) -> int:
        return self.day_number

    @simulation_day.setter
    def simulation_day(self, value: int):
        self.day_number = value

    def __str__(self) -> str:
        return f"ShopVisit(vid={self.vid}, customer={self.cid}, day={self.day_number})"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# MarketingOperation
# ---------------------------------------------------------------------------

class MarketingOperation(BaseModel):
    """Marketing operation model (Figure 3 Database schema)."""

    mid = IntegerField(primary_key=True)
    cid = IntegerField()                      # Reference to Customer.cid
    date = DateField()
    marketing_hook = TextField()
    item_recommendations = JSONField()        # List[PID]
    entire_ad_message = TextField()
    website_visited = BooleanField(default=False)
    client_response = TextField(default="")
    llm_hook_instruction = TextField(default="")

    class Meta:
        table_name = 'marketing_operations'

    @property
    def simulation_day(self) -> int:
        if isinstance(self.date, date):
            return (self.date - _SIMULATION_START).days + 1
        return 1

    @simulation_day.setter
    def simulation_day(self, value: int):
        self.date = _SIMULATION_START + timedelta(days=value - 1)

    @classmethod
    def create_from_marketing_email(cls, marketing_email, client, session_result, mid: int, simulation_day: int = 1):
        """Create a MarketingOperation from a MarketingEmail and session result."""
        operation_date = _SIMULATION_START + timedelta(days=simulation_day - 1)
        return cls.create(
            mid=mid,
            cid=client.cid,
            date=operation_date,
            marketing_hook=marketing_email.marketing_hook,
            item_recommendations=marketing_email.recommended_item_ids,
            entire_ad_message=marketing_email.get_full_email_content(),
            website_visited=False,
            client_response=session_result.get('client_response', '')
        )

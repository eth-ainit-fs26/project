"""
Data loading and database seeding utilities.

Loads TSV files from the data/ directory and seeds the SQLite database
with customers, products, and historical transactions.
"""
import ast
import os

import numpy as np
import pandas as pd
from peewee import IntegrityError


def load_data_from_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load customers, products, and optimal hooks from TSV files in data/."""
    data_dir = os.path.join(os.getcwd(), "data")

    customers_file = os.path.join(data_dir, "customers.tsv")
    products_file = os.path.join(data_dir, "products.tsv")
    hooks_file = os.path.join(data_dir, "optimal_hooks.tsv")

    for path in (customers_file, products_file, hooks_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    customers_df = pd.read_csv(customers_file, sep='\t')
    products_df = pd.read_csv(products_file, sep='\t')
    hooks_df = pd.read_csv(hooks_file, sep='\t')

    for name, df in [("customers.tsv", customers_df), ("products.tsv", products_df), ("optimal_hooks.tsv", hooks_df)]:
        if df.empty:
            raise ValueError(f"{name} is empty")

    return customers_df, products_df, hooks_df


def setup_real_data(customers_df: pd.DataFrame, products_df: pd.DataFrame) -> None:
    """Seed the database with customers and products from TSV data."""
    from models.models import Customer, Item

    segment_satisfaction = {'H': 0.8, 'R': 0.7, 'N': 0.6, 'L': 0.5, 'A': 0.4}

    for _, row in customers_df.iterrows():
        location = (np.random.uniform(-100, 100), np.random.uniform(-100, 100))
        income_factor = min(row['Income'] / 300000, 1.0)
        base_satisfaction = segment_satisfaction.get(row['Segment'], 0.6)
        satisfaction = min(base_satisfaction + income_factor * 0.2, 1.0)
        feature_vector = np.array(ast.literal_eval(row['FeatureVector'])).tolist()

        Customer.create(
            cid=row['CID'],
            name=row['Name'],
            age=row['Age'],
            gender=row['Gender'],
            location_x=location[0],
            location_y=location[1],
            job_type=row['JobType'],
            income=row['Income'],
            segment=row['Segment'],
            family_status=row['FamilyStatus'],
            feature_vector=feature_vector,
            satisfaction=satisfaction,
            visit_probability=0.1,
            session_probability=0.3,
            trigger_keywords=[],
        )

    for _, row in products_df.iterrows():
        cost = row['Cost'] if 'Cost' in row and pd.notna(row['Cost']) else row['Price'] * 0.5
        feature_vector = np.array(ast.literal_eval(row['FeatureVector'])).tolist()

        Item.create(
            pid=row['PID'],
            name=row['Name'],
            description=row['Description'],
            price=row['Price'],
            discounted=False,
            cost=cost,
            stock_level=np.random.randint(10, 101),
            feature_vector=feature_vector,
        )


def setup_real_transactions(customers_df: pd.DataFrame, products_df: pd.DataFrame) -> None:
    """Seed the database with historical transactions from transactions.tsv.

    The TSV schema is: TID, CID, Date, Order, OrderProfit, OrderSize
    where Order is a Python-literal list of (pid, qty) tuples.
    """
    from models.models import Transaction
    from datetime import date as date_type

    data_dir = os.path.join(os.getcwd(), "data")
    transactions_file = os.path.join(data_dir, "transactions.tsv")

    if not os.path.exists(transactions_file):
        print("Warning: transactions.tsv not found, skipping transaction loading")
        return

    transactions_df = pd.read_csv(transactions_file, sep='\t')
    print(f"Loading {len(transactions_df)} transactions...")

    price_lookup = {row['PID']: row['Price'] for _, row in products_df.iterrows()}

    for _, row in transactions_df.iterrows():
        # Parse Order column: "[(pid, qty), ...]"
        raw_order = ast.literal_eval(row['Order'])
        # Build order_data as List[(PID, quantity, price)]
        order_data = [(pid, qty, price_lookup.get(pid, 0.0)) for pid, qty in raw_order]

        try:
            tx_date = date_type.fromisoformat(row['Date']) if pd.notna(row['Date']) else date_type.today()
        except (ValueError, TypeError):
            tx_date = date_type.today()

        try:
            Transaction.create(
                tid=int(row['TID']),
                cid=int(row['CID']),
                date=tx_date,
                order_data=order_data,
                delivery_time_window_start=0,
                delivery_time_window_end=0,
            )
        except IntegrityError:
            pass


def create_system_parameters_from_data(customers_df: pd.DataFrame, products_df: pd.DataFrame, hooks_df: pd.DataFrame):
    """Build a SystemParameters instance from the loaded data."""
    from system_config.system_parameters import SystemParameters
    return SystemParameters(K=5)


def build_hook_evaluator(hooks_df: pd.DataFrame, customers_df: pd.DataFrame, products_df: pd.DataFrame):
    """Build optimal_hooks dict and HookEvaluator from the loaded hooks data.

    Returns:
        (system_parameters, hook_evaluator)
    """
    from agents.hooks import HookEvaluator

    optimal_hooks = {row['Segment']: row['Hook'] for _, row in hooks_df.iterrows()}
    system_parameters = create_system_parameters_from_data(customers_df, products_df, hooks_df)
    hook_evaluator = HookEvaluator(optimal_hooks=optimal_hooks)
    return system_parameters, hook_evaluator

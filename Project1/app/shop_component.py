from models.registries import CustomerRegistry
from models.registries import ItemCatalogue
from models.models import Transaction, ShopVisit
from models.registries import TransactionRegistry
from models.registries import shop_visit_registry
import random
from typing import Optional, List, Any


class ShopComponent:
    """
    Shop Component that manages customer sessions and interactions.
    Coordinates between customer registry and item catalog for shop operations.
    """
    
    def __init__(self, customer_registry: CustomerRegistry, item_catalogue: ItemCatalogue, transaction_registry: TransactionRegistry, logging_config: Optional[Any] = None):
        """
        Initialize ShopComponent with customer registry, item catalog, and transaction registry.
        
        Args:
            customer_registry: Registry for managing customer data
            item_catalogue: Catalog of available items in the shop
            transaction_registry: Registry for storing transactions
            logging_config: Configuration for controlling logging output (optional)
        """
        self.customer_registry = customer_registry
        self.item_catalogue = item_catalogue
        self.transaction_registry = transaction_registry
        self.logging_config = logging_config  # Store logging configuration
        self.current_simulation_day = 1  # Track current simulation day
        self._next_transaction_id = 1  # Track transaction IDs
    
    def run_session(self):
        """
        Run sessions for all customers in the registry.
        Iterates through all customers and runs individual sessions.
        """
        # Get all customers from the registry
        customers = self.customer_registry.get_all_customers()
        
        visits_count = 0
        transactions_count = 0
        daily_visits = []
        
        # Run session for each customer
        for i, customer in enumerate(customers, 1):
            customer_name = customer.name
            customer_id = customer.cid
            
            # First test if the customer will visit the shop
            if self.determine_visit(customer):
                visits_count += 1
                
                # Save the shop visit and get the visit object
                visit = self.save_shop_visit(customer)
                daily_visits.append(visit)
                
                # Run the customer session
                transactions = self.run_session_customer(customer)
                
                # Save all transactions to the registry
                for transaction in transactions:
                    self.save_transaction(transaction)
                    transactions_count += 1
                
                # Print one line summary for visited customer (if logging enabled)
                if self.logging_config is None or self.logging_config.shopping_individual:
                    print(f"🛒 {customer_name}: Segment: {customer.segment}, Visit Prob: {customer.visit_probability:.3f}, Visited: YES")
            else:
                # Print one line summary for non-visited customer (if logging enabled)
                if self.logging_config is None or self.logging_config.shopping_individual:
                    print(f"🛒 {customer_name}: Segment: {customer.segment}, Visit Prob: {customer.visit_probability:.3f}, Visited: NO")
        
        # Print daily summary if logging enabled
        if self.logging_config is None or self.logging_config.shopping_daily_summary:
            print(f"\n📊 DAILY SHOPPING SUMMARY - Day {self.current_simulation_day}")
            print(f"   👥 Total customers processed: {len(customers)}")
            print(f"   🏪 Total shop visits: {visits_count}")
            print(f"   💳 Total transactions: {transactions_count}")
            print(f"   📈 Visit rate: {(visits_count/len(customers)*100):.1f}%")
    
    def run_session_customer(self, customer):
        """
        Run a session for a specific customer.
        Session logic: pick random number 0-3 for different products to buy in one transaction.
        
        Args:
            customer: Customer object to run session for
            
        Returns:
            list: List containing single Transaction object (or empty if no purchase)
        """
        # Pick a random number between 0 and 3 (number of different products to buy)
        num_different_products = random.randint(0, 3)
        
        transactions = []
        
        # Create one transaction with the determined number of different products
        if num_different_products > 0:
            transaction = self.create_arbitrary_transaction(customer, tid=self._next_transaction_id, num_products=num_different_products)
            if transaction:
                self._next_transaction_id += 1
                transactions.append(transaction)
        
        return transactions
    
    def determine_visit(self, customer) -> bool:
        """
        Determine if a customer will visit using a Bernoulli trial.
        
        Args:
            customer: Customer object containing visit probability
            
        Returns:
            bool: True if customer will visit, False otherwise
        """
        # Get the customer's probability of visiting
        visit_probability = customer.visit_probability
        
        # Perform Bernoulli trial
        return random.random() < visit_probability
    
    def create_arbitrary_transaction(self, customer, tid: Optional[int] = None, num_products: Optional[int] = None) -> Optional[Transaction]:
        """
        Create arbitrary transaction for a customer using the Transaction model.
        Randomly selects items from the catalog and generates purchase quantities.
        
        Args:
            customer: Customer object making the transaction
            tid: Transaction ID (optional, defaults to 0)
            num_products: Number of different products to include (optional, will be random if not specified)
            
        Returns:
            Transaction: Transaction model object or None if no items available
        """
        # Get available items from catalog
        available_items = self.item_catalogue.get_all_items()
        
        if not available_items:
            return None  # No items available
        
        # Determine how many different items to purchase
        if num_products is not None:
            num_items = min(num_products, len(available_items))  # Can't exceed available items
        else:
            # Default behavior: randomly decide how many different items to purchase (1-5 items)
            num_items = random.randint(1, min(5, len(available_items)))
        
        # If num_items is 0, return None (no purchase)
        if num_items == 0:
            return None
        
        # Randomly select items
        selected_items = random.sample(available_items, num_items)
        
        # Create order list: List[(PID, quantity, price)]
        order = []
        
        for item in selected_items:
            # Random quantity for each item (1-3)
            quantity = random.randint(1, 3)
            
            # Get item price and ID
            item_price = getattr(item, 'price', 10.0)  # Default price if not available
            item_id = getattr(item, 'id', random.randint(1, 1000))  # Default ID if not available
            
            # Add to order as tuple (PID, quantity, price)
            order.append((item_id, quantity, item_price))
        
        # Generate delivery time window (simulation day + 1-7 days)
        delivery_start_day = self.current_simulation_day + random.randint(1, 7)  # 1-7 days from current day
        delivery_end_day = delivery_start_day + 1  # 1 day delivery window
        delivery_time_window = (delivery_start_day, delivery_end_day)
        
        # Create Transaction object
        transaction = Transaction(
            tid=tid or 0,
            cid=customer.cid,
            simulation_day=self.current_simulation_day,
            order=order,
            delivery_time_window=delivery_time_window
        )
        
        return transaction
    
    def reset(self):
        """Reset day counter and clear all shop visit records for a fresh run."""
        self.current_simulation_day = 1
        shop_visit_registry.clear_all_visits()

    def advance_simulation_day(self):
        """Advance the simulation to the next day."""
        self.current_simulation_day += 1
    
    def set_simulation_day(self, day: int):
        """Set the current simulation day."""
        self.current_simulation_day = day
    
    def get_current_simulation_day(self) -> int:
        """Get the current simulation day."""
        return self.current_simulation_day
    
    def save_shop_visit(self, customer):
        """
        Save a shop visit record when a customer visits the shop.
        
        Args:
            customer: Customer object who is visiting
            
        Returns:
            ShopVisit: The saved shop visit object
        """
        # Create shop visit record
        visit = ShopVisit(
            day_number=self.current_simulation_day,
            cid=customer.cid
        )
        
        # Save to registry and return the visit object
        return shop_visit_registry.save_shop_visit(visit)
    
    def get_visits_for_day(self, day_number: Optional[int] = None) -> List:
        """
        Get all shop visits for a specific day.
        
        Args:
            day_number: Simulation day to get visits for (defaults to current day)
            
        Returns:
            List of ShopVisit objects
        """
        if day_number is None:
            day_number = self.current_simulation_day
        
        return shop_visit_registry.get_visits_by_day(day_number)
    
    def get_visit_count_for_day(self, day_number: Optional[int] = None) -> int:
        """
        Get the number of visits for a specific day.
        
        Args:
            day_number: Simulation day (defaults to current day)
            
        Returns:
            int: Number of visits
        """
        if day_number is None:
            day_number = self.current_simulation_day
        
        return shop_visit_registry.get_visit_count_by_day(day_number)
    
    def save_transaction(self, transaction: Transaction):
        """
        Save a transaction to the transaction registry.
        
        Args:
            transaction: Transaction object to save
        """
        self.transaction_registry.add_transaction(transaction)
from agents.marketing_agent import MarketingAgent
from agents.marketing_session import MarketingSession
from agents.project2_recommendation.random_recommender import RandomRecommenderAgent
from models.registries import ItemCatalogue
from models.registries import CustomerRegistry
from models.registries import TransactionRegistry
from models.models import MarketingOperation, Transaction
from models.database import initialize_database
from services.llm_service import LLMService
from system_config.system_parameters import SystemParameters
from datetime import date
import random
import numpy as np


class WorldSimulator:
    """
    World Simulator class for managing the retail transformation simulation.
    
    From Section 9.1.1: The World Simulator covers everything that the model does not
    see/control. It maintains the hidden customer states and is the main source of 
    contact between the model parts and the customers.
    """
    
    def __init__(self, llm_service: LLMService, system_parameters: SystemParameters):
        """Initialize the World Simulator."""
        # Initialize database
        initialize_database()

        self.llm_service = llm_service
        self.system_parameters = system_parameters
        self.item_catalogue = ItemCatalogue()
        self.customer_registry = CustomerRegistry()
        self.transaction_registry = TransactionRegistry()
        self.marketing_agent = MarketingAgent(self.item_catalogue, llm_service)

        # Initialize recommendation agent
        self.recommender_agent = RandomRecommenderAgent(
            self.customer_registry,
            self.item_catalogue,
            self.transaction_registry
        )

        self._marketing_operation_counter = 0  # For incremental MID generation
        self._transaction_counter = 0  # For incremental TID generation
    
    def runSimulation(self):
        """Run the complete simulation."""
        pass
    
    def runMarketing(self, client):
        """Run single marketing step for a specific client."""
        # Run only one marketing step instead of multiple steps
        marketing_session = MarketingSession(
            client=client,
            item_catalogue=self.item_catalogue,
            marketing_agent=self.marketing_agent,
            transaction_registry=self.transaction_registry,
            k=1  # Run only one step
        )
        session_history = marketing_session.run_session()
        
        session_result = session_history[0]
        
        if session_result:
            # Create MarketingOperation object from session result
            marketing_operation = self._create_marketing_operation_from_session(client, session_result)
            return marketing_operation
        
        return None
    
    def _create_marketing_operation_from_session(self, client, session_result):
        """Create a MarketingOperation object from session result."""
        # Generate incremental marketing operation ID
        self._marketing_operation_counter += 1
        mid = self._marketing_operation_counter

        marketing_email = session_result['marketing_email']

        if marketing_email:
            return MarketingOperation.create_from_marketing_email(marketing_email, client, session_result, mid)
        else:
            # Fallback in case there's no marketing_email
            return MarketingOperation(
                mid=mid,
                cid=client.cid,
                date=date.today(),
                marketing_hook="",
                item_recommendations=[],
                entire_ad_message="",
                website_visited=False,
                client_response=""
            )

    def runRecommendations(self, customer_id: int, k: int = 10) -> list[int]:
        """Generate recommendations for visiting customer"""
        return self.recommender_agent.recommend(customer_id, k)

    def processCustomerSession(self, customer_id: int) -> tuple[int, float]:
        """Complete customer session: Customer Shopping Process"""
        # 1. Customer visits website (already happened via marketing)
        # Record the shop visit to database
        from models.registries import shop_visit_registry
        simulation_day = getattr(self, 'simulation_day', 1)
        shop_visit_registry.create_shop_visit(
            cid=customer_id,
            day_number=simulation_day
        )

        # 2. Get K recommendations from recommender algorithm
        k = self.system_parameters.K if hasattr(self.system_parameters, 'K') else 10
        shown_items = self.runRecommendations(customer_id, k)

        if not shown_items:
            return 0, 0.0

        # 3. Initialize shopping cart and simulate browsing session
        shopping_cart = []
        available_items = shown_items.copy()  # Items still available to purchase
        customer = self.customer_registry.get_customer(customer_id)

        if not customer:
            return 0, 0.0

        # 4. Shopping loop
        iteration_count = 0

        while available_items:
            iteration_count += 1

            # Check if customer continues shopping: Bernoulli(P_s^(c)) = 0 means continue
            # For simplicity, use segment-based stopping probability
            stop_probability = self._get_customer_stop_probability(customer)
            if self._bernoulli_trial(stop_probability):
                break

            # Find most preferred item: i* = arg max κ̂(pc, fi)
            best_item_id = self._find_most_preferred_item(customer_id, available_items)

            # Purchase decision: pbuy = κ(c, i*) (using hidden ground truth)
            purchase_prob = self._get_true_kernel_value(customer, best_item_id)

            if self._bernoulli_trial(purchase_prob):
                shopping_cart.append(best_item_id)

            # Remove item from available set (purchased or rejected)
            available_items.remove(best_item_id)

        # 5. Place order with items in shopping cart
        total_revenue = 0.0
        if shopping_cart:
            transaction = self._create_transaction(customer_id, shopping_cart)
            self.transaction_registry.add_transaction(transaction)
            total_revenue = self._get_items_total_price(shopping_cart)

        # 6. Update recommender with session feedback
        self.recommender_agent.update_from_session(customer_id, shown_items, shopping_cart)

        return len(shopping_cart), total_revenue

    def _get_customer_stop_probability(self, customer) -> float:
        """Get customer's session stopping probability based on segment"""
        segment = customer.segment
        if segment == 'H':  # High-value: browse longer
            return 0.2
        elif segment == 'B':  # Bargain hunters: browse longer looking for deals
            return 0.15
        elif segment == 'N':  # Newcomers: stop earlier, less confident
            return 0.4
        else:  # At-risk: stop quickly
            return 0.5

    def _find_most_preferred_item(self, customer_id: int, available_items: list[int]) -> int:
        """Find i* = arg max κ̂(pc, fi) using recommender's learned model"""
        best_score = -1.0
        best_item = available_items[0]

        for item_id in available_items:
            # Use recommender's affinity estimate
            score = self.recommender_agent._kappa_hat(customer_id, item_id)
            if score > best_score:
                best_score = score
                best_item = item_id

        return best_item

    def _get_true_kernel_value(self, customer, item_id: int) -> float:
        """Get true kernel value κ(c,i) using cosine similarity: κ(c,i) := cos-sim(fi, pc)"""
        # Get customer preference vector (pc) and item feature vector (fi)
        customer_id = customer.cid

        # Get customer preference vector from customer model
        pc = np.array(customer.feature_vector)  # Hidden preference vector (already parsed by Peewee)

        # Get item feature vector from item model
        item = self.item_catalogue.get_item(item_id)
        if item is None:
            return 0.0
        fi = np.array(item.feature_vector)  # Hidden feature vector (already parsed by Peewee)

        # Calculate cosine similarity: cos-sim(fi, pc) = ⟨fi,pc⟩ / (∥fi∥∥pc∥)
        dot_product = np.dot(fi, pc)
        norm_fi = np.linalg.norm(fi)
        norm_pc = np.linalg.norm(pc)

        if norm_fi > 0 and norm_pc > 0:
            cosine_sim = dot_product / (norm_fi * norm_pc)
            # Ensure result is in [0,1] range (cosine similarity is in [-1,1])
            return max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))
        else:
            return 0.0

    def _bernoulli_trial(self, probability: float) -> bool:
        """Perform a Bernoulli trial with given probability"""
        return random.random() < probability

    def _create_transaction(self, customer_id: int, purchased_items: list[int]) -> Transaction:
        """Create a Transaction object"""

        # Create order list with (item_id, quantity, price) tuples
        order = []
        for item_id in purchased_items:
            # Ensure item_id is an integer, in case it somehow got wrapped
            item_id = int(item_id) if isinstance(item_id, dict) else item_id
            item = self.item_catalogue.get_item(item_id)
            if item is not None:
                order.append((item_id, 1, item.price))  # Quantity = 1 for simplicity

        tid = random.randint(100, 100000000000)

        return Transaction(
            tid=tid,
            cid=customer_id,
            date=date.today(),
            order=order,
            delivery_time_window=(None, None)  # Simplified for now
        )

    def _get_items_total_price(self, item_ids: list[int]) -> float:
        """Calculate total price for list of items"""
        total = 0.0
        for item_id in item_ids:
            # Ensure item_id is an integer, in case it somehow got wrapped
            item_id = int(item_id) if isinstance(item_id, dict) else item_id
            item = self.item_catalogue.get_item(item_id)
            if item is not None:
                total += item.price
        return total
    
    def runSimulationStep(self):
        """Run a single step of the simulation."""
        # 1. Customer Interaction Cycle
        
        # Get all clients and run marketing step for each
        all_clients = self.customer_registry.get_all_customers()
        for client in all_clients:
            self.runMarketing(client)

        # 1.1 Marketing engagement - Send personalized marketing messages to customers
        # 1.2 Update customer visit probabilities P_v^(c) based on marketing response
        # 1.3 Simulate website visits - customers decide whether to visit based on P_v^(c)
        # 1.4 For visiting customers, generate product recommendations using Component II
        # 1.5 Simulate customer browsing and order placement decisions
        # 1.6 Update customer preferences pc based on interactions and purchases
        
        # 2. Delivery Optimization Cycle  
        # 2.1 Collect all orders placed during this day
        # 2.2 Run CVRP-TW optimization using Component III (RL delivery system)
        # 2.3 Execute deliveries and track actual delivery times
        # 2.4 Update customer satisfaction based on delivery performance
        
        # 3. System State Updates
        # 3.1 Update customer hidden states: preferences, visit probabilities, keywords, locations
        # 3.2 Update item inventory levels based on purchases
        # 3.3 Record all transactions, marketing operations, and deliveries to database
        # 3.4 Calculate and log daily performance metrics (visits, revenue, deliveries)
        # 3.5 Apply feedback loops between components (marketing→recommendations→delivery→marketing)
        
        pass


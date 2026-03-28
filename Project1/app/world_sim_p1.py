"""
World Simulator P1 - Core simulation engine for AI-powered retail transformation.

This module manages the complete simulation lifecycle including:
- Daily marketing campaigns with LLM-generated hooks
- Customer engagement and response simulation  
- Shop visit probability updates based on marketing effectiveness
- Transaction processing and data persistence
"""
from agents.marketing_agent import MarketingAgent
from agents.marketing_session import MarketingSession
from agents.hooks import HookEvaluator
from agents.opro import OptimizerEntry
from app.shop_component import ShopComponent
from models.registries import ItemCatalogue
from models.registries import CustomerRegistry
from models.registries import TransactionRegistry
from models.models import MarketingOperation, CustomerSegment
from services.llm_service import LLMService
from system_config.system_parameters import SystemParameters
from datetime import date
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """
    Configuration for controlling different types of logging output in WorldSimulatorP1.
    """
    # Daily simulation progress
    daily_progress: bool = False          # "Running simulation day X/Y"
    
    # Marketing phase logging
    marketing_hooks: bool = False         # Generated hooks for segments
    marketing_individual: bool = False    # Individual customer marketing results
    
    # Shopping phase logging
    shopping_phase: bool = False          # "Shopping Phase - Day X"
    shopping_individual: bool = False     # Individual customer visit details ("🛒 Customer: Visited: YES/NO")
    shopping_daily_summary: bool = False # Daily shopping summary with visit counts and percentages
    
    # Accuracy and metrics logging
    accuracy_daily: bool = False          # Daily accuracy calculations
    accuracy_cycle: bool = False          # OPRO cycle progress and differences
    
    # OPRO optimization logging
    opro_optimization: bool = False       # OPRO optimization process
    opro_prompts: bool = False           # New optimized prompts
    opro_fallbacks: bool = False         # OPRO fallback messages
    opro_meta_prompt: bool = False      # Show complete meta-prompt content
    
    # Day advancement logging
    day_advancement: bool = False         # "Advanced to Day X"
    
    # Final summary logging
    final_history: bool = False           # OPRO history at end
    
    # Simulation setup logging
    setup_messages: bool = False          # Initialization and setup messages


class WorldSimulatorP1:
    """
    World Simulator P1 class for managing the retail transformation simulation.
    
    From Section 9.1.1: The World Simulator covers everything that the model does not
    see/control. It maintains the hidden customer states and is the main source of 
    contact between the model parts and the customers.
    """
    
    def __init__(self, llm_service: LLMService, system_parameters: SystemParameters, hook_evaluator: HookEvaluator, simulation_duration: int = 30, opro_cycle_days: int = 3, logging_config: Optional[LoggingConfig] = None, use_opro2: bool = False):
        """
        Initialize the World Simulator P1.
        
        Args:
            llm_service: LLM service for text generation
            system_parameters: System configuration parameters
            hook_evaluator: HookEvaluator for marketing hook quality assessment
            simulation_duration: Number of days to run simulation (default: 30)
            opro_cycle_days: Number of days between OPRO optimizations (default: 3)
            logging_config: Configuration for controlling logging output (default: all enabled)
            use_opro2: If True, use OPRO2 direct hook optimization. If False, use original prompt optimization. (default: False)
        """
        self.llm_service = llm_service
        self.system_parameters = system_parameters
        self.hook_evaluator = hook_evaluator
        self.simulation_duration = simulation_duration  # Number of days to run simulation
        self.opro_cycle_days = opro_cycle_days  # Number of days between OPRO optimizations
        self.current_simulation_day = 1  # Track current simulation day
        self.logging_config = logging_config if logging_config is not None else LoggingConfig()  # Logging configuration
        self.item_catalogue = ItemCatalogue()
        self.customer_registry = CustomerRegistry()
        self.transaction_registry = TransactionRegistry()
        self.marketing_agent = MarketingAgent(self.item_catalogue, llm_service, use_opro2=use_opro2)
        self.shop_component = ShopComponent(self.customer_registry, self.item_catalogue, self.transaction_registry, self.logging_config)
        
        # Clear the Peewee database marketing operations from previous runs
        from models.models import MarketingOperation
        try:
            MarketingOperation.delete().execute()
            print("Cleared Peewee marketing operations")
        except Exception as e:
            print(f"Warning: Could not clear Peewee marketing operations: {e}")
        
        # Clear the global marketing operation registry to avoid ID conflicts
        from models.registries import marketing_operation_registry
        marketing_operation_registry.clear_all_operations()
        
        self._marketing_operation_counter = 0  # For incremental MID generation
        
        # Initialize optimized prompts/hooks for each customer segment based on OPRO mode
        if use_opro2:
            # OPRO2: Start with magic strings containing default optimized hooks (no emojis)
            self.optimized_prompts = {
                CustomerSegment.HIGH_VALUE_CUSTOMERS.value: "OPRO2_HOOK:Exclusive VIP Preview: Your Premium Collection Awaits Inside",
                CustomerSegment.NEWCOMERS.value: "OPRO2_HOOK:Welcome! Your Personal Shopping Journey Starts Here",
                CustomerSegment.AT_RISK_CUSTOMERS.value: "OPRO2_HOOK:We Miss You! Here's What You've Been Missing + Special Return Offer",
                CustomerSegment.LOW_VALUE_CUSTOMERS.value: "OPRO2_HOOK:Special Savings Just for You! Don't Miss Out on These Deals",
                CustomerSegment.REGULAR_CUSTOMERS.value: "OPRO2_HOOK:Thank You for Being with Us! Enjoy Exclusive Deals Inside"
            }
        else:
            # Original OPRO: Start with LLM instruction prompts
            self.optimized_prompts = {
                CustomerSegment.HIGH_VALUE_CUSTOMERS.value: "Generate a single compelling email marketing hook for high-value loyal customers. Focus on exclusivity, premium quality, and VIP treatment. Use sophisticated language that reflects their status. Return only the hook text without any prefixes like 'Subject:' or 'Hook:'. Example format: 'Exclusive VIP Preview: Your Premium Collection Awaits Inside'",
                CustomerSegment.NEWCOMERS.value: "Generate a single welcoming email marketing hook for new customers. Emphasize discovery, first-time benefits, and introductory offers. Use friendly, encouraging language to build trust. Return only the hook text without any prefixes like 'Subject:' or 'Hook:'. Example format: 'Welcome! Your Personal Shopping Journey Starts Here'",
                CustomerSegment.AT_RISK_CUSTOMERS.value: "Generate a single re-engagement email marketing hook for customers who haven't purchased recently. Focus on 'we miss you' messaging, comeback offers, and highlighting what they've been missing. Return only the hook text without any prefixes like 'Subject:' or 'Hook:'. Example format: 'We Miss You! Here's What You've Been Missing + Special Return Offer'",
                CustomerSegment.LOW_VALUE_CUSTOMERS.value: "Generate a single promotional email marketing hook for price-sensitive customers. Highlight savings, discounts, limited-time offers, and value propositions. Use urgency and scarcity language. Return only the hook text without any prefixes like 'Subject:' or 'Hook:'. Example format: 'Great finds, even better value! Limited-time savings inside'",
                CustomerSegment.REGULAR_CUSTOMERS.value: "Generate a single engaging email marketing hook for regular customers. Focus on appreciation, new arrivals, and maintaining their shopping relationship. Use warm, familiar language that acknowledges their loyalty. Return only the hook text without any prefixes like 'Subject:' or 'Hook:'. Example format: 'Thank You for Being with Us! Discover What's New This Week'"
            }
        
        # Current hooks generated for each customer segment for this simulation day
        self.current_hooks: Dict[str, str] = {}
        
        # Pre-computed hook embeddings for efficiency (avoid recomputing for each customer)
        self.current_hook_embeddings: Dict[str, Any] = {}
        
        # Store previous iteration data for OPRO optimization
        self.previous_segment_scores: Dict[str, float] = {}
        self.previous_segment_prompts: Dict[str, str] = {}
        self.previous_segment_hooks: Dict[str, str] = {}
        
        # Store all OPRO optimization history for analysis
        self.opro_optimization_history: List[Dict[str, Any]] = []
        
        # Store previous day's accuracy for difference calculation
        self.previous_day_segment_accuracy: Dict[str, float] = {}
        
        # OPRO cycle tracking
        self.current_cycle_accuracies: List[Dict[str, float]] = []  # Store daily accuracies for current cycle
        self.previous_cycle_average_accuracy: Dict[str, float] = {}  # Average accuracy from previous cycle
        self.cycle_count = 0  # Track which cycle we're in

        # Running best: monotonically non-decreasing best accuracy per segment
        self.best_segment_accuracy: Dict[str, float] = {}
        # The prompt/hook that produced the best accuracy for each segment
        self.best_segment_prompts: Dict[str, str] = {}
        self.best_segment_hooks: Dict[str, str] = {}
    
    def reset(self):
        """Reset simulation state so the dashboard can be re-run cleanly."""
        self.current_simulation_day = 1
        self.current_hooks = {}
        self.current_hook_embeddings = {}
        self.previous_segment_scores = {}
        self.previous_segment_prompts = {}
        self.previous_segment_hooks = {}
        self.opro_optimization_history = []
        self.previous_day_segment_accuracy = {}
        self.current_cycle_accuracies = []
        self.previous_cycle_average_accuracy = {}
        self.cycle_count = 0
        self.best_segment_accuracy = {}
        self.best_segment_prompts = {}
        self.best_segment_hooks = {}
        # Clear shop visits and marketing operations from the previous run
        from models.models import MarketingOperation
        from models.registries import marketing_operation_registry
        try:
            MarketingOperation.delete().execute()
        except Exception:
            pass
        marketing_operation_registry.clear_all_operations()
        self._marketing_operation_counter = 0
        self.shop_component.reset()

    def runSimulation(self, progress_function):
        """Run the complete simulation."""
        while self.current_simulation_day <= self.simulation_duration:
            if self.logging_config.daily_progress:
                print(f"Running simulation day {self.current_simulation_day}/{self.simulation_duration}")
            
            # Capture the day BEFORE the step advances it
            day_just_simulated = self.current_simulation_day
            
            self.runSimulationStep()

            progress_function(self.snapshot_day(day_just_simulated))
    
    def _capture_final_day_opro_data(self):
        """
        Capture the final cycle's OPRO data if the simulation ends mid-cycle.
        """
        # If we have uncompleted cycle data, process it
        if self.current_cycle_accuracies:
            # Calculate partial cycle average accuracy differences
            final_cycle_differences = self.calculate_cycle_average_accuracy_differences()
            final_cycle_number = self.cycle_count + 1
            
            if self.logging_config.accuracy_cycle:
                print(f"🔄 Final incomplete cycle data (Cycle {final_cycle_number}):")
            
            for segment, difference in final_cycle_differences.items():
                if (segment in self.optimized_prompts and 
                    segment in self.current_hooks):
                    
                    # Only add if not already captured
                    already_captured = any(
                        entry['day'] == f"Cycle {final_cycle_number}" and entry['segment'] == segment 
                        for entry in self.opro_optimization_history
                    )
                    
                    if not already_captured:
                        entry = OptimizerEntry(
                            accuracy_score=difference,
                            prompt=self.optimized_prompts[segment],
                            hook=self.current_hooks[segment]
                        )
                        
                        history_entry = {
                            'day': f"Cycle {final_cycle_number} (incomplete)",
                            'segment': segment,
                            'optimizer_entry': entry
                        }
                        self.opro_optimization_history.append(history_entry)
    
    def runMarketing(self, client):
        """Run single marketing step for a specific client."""
        # Store old visit probability before marketing
        old_visit_probability = client.visit_probability
        
        # Run only one marketing step instead of multiple steps
        marketing_session = MarketingSession(
            client=client,
            item_catalogue=self.item_catalogue,
            marketing_agent=self.marketing_agent,
            transaction_registry=self.transaction_registry,
            k=1,  # Run only one step
            simulation_day=self.current_simulation_day,
            optimal_hooks=self.hook_evaluator.optimal_hooks,  # Pass optimal hooks from evaluator
            current_hooks=self.current_hooks,  # Pass pre-generated hooks for current day
            current_hook_embeddings=self.current_hook_embeddings,  # Pass pre-computed embeddings
            hook_evaluator=self.hook_evaluator  # Pass hook evaluator for optimized evaluation
        )
        
        session_history = marketing_session.run_session()
        
        # Get new visit probability after marketing
        new_visit_probability = client.visit_probability
        hook_score_delta = new_visit_probability - old_visit_probability
        
        # Print one line summary for this client
        if self.logging_config.marketing_individual:
            print(f"📧 {client.name}: Visit Prob: {new_visit_probability:.3f}, Hook Score Δ: {hook_score_delta:+.3f}")
        
        if session_history and len(session_history) > 0:
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
        
        marketing_email = session_result.get('marketing_email')
        
        if marketing_email:
            return MarketingOperation.create_from_marketing_email(marketing_email, client, session_result, mid, self.current_simulation_day)
        else:
            # Fallback in case there's no marketing_email
            return MarketingOperation(
                mid=mid,
                cid=client.cid,
                simulation_day=self.current_simulation_day,
                marketing_hook="",
                item_recommendations=[],
                entire_ad_message="",
                website_visited=False,
                client_response=""
            )
    
    def runSimulationStep(self):
        """Run a single step of the simulation."""
        if self.logging_config.daily_progress:
            print(f"\n🔄 === DAY {self.current_simulation_day} SIMULATION STEP ===")
        
        # Sync simulation day with shop component
        self.shop_component.set_simulation_day(self.current_simulation_day)
        
        # 1. Customer Interaction Cycle
        if self.logging_config.marketing_hooks:
            print(f"📧 Marketing Phase - Day {self.current_simulation_day}")
        
        # Get all clients and run marketing step for each
        all_clients = self.customer_registry.get_all_customers()

        # Generate hooks for each customer segment at the start of each day
        if self.logging_config.marketing_hooks:
            print(f"🎣 Generating hooks for customer segments:")
        self.current_hooks = self.marketing_agent.generate_hooks_for_segments(self.optimized_prompts)
        if self.logging_config.marketing_hooks:
            for segment, hook in self.current_hooks.items():
                print(f"   • {segment}: {hook}")
        
        # Pre-compute BERT embeddings for all hooks to avoid recomputing for each customer
        self.current_hook_embeddings = {}
        for segment, hook in self.current_hooks.items():
            try:
                embedding = self.hook_evaluator.encode_hook(hook)
                self.current_hook_embeddings[segment] = embedding
            except Exception as e:
                # Fallback: empty embedding will trigger individual computation later
                self.current_hook_embeddings[segment] = None
        
        marketing_results = []
        for i, client in enumerate(all_clients, 1):
            result = self.runMarketing(client)
            marketing_results.append(result)

        if self.logging_config.shopping_phase:
            print(f"🛒 Shopping Phase - Day {self.current_simulation_day}")
        self.shop_component.run_session()

        # Calculate absolute accuracy for current day and store for cycle tracking
        segment_accuracy = self.calculate_segment_hook_accuracy(self.current_simulation_day)
        if self.logging_config.accuracy_daily:
            print(f"📈 Segment Hook Absolute Accuracy for Day {self.current_simulation_day}:")
            for segment, accuracy in segment_accuracy.items():
                print(f"   • {segment}: {accuracy:.3f} ({accuracy*100:.1f}%)")

        # Add current day's accuracy to the current cycle
        self.current_cycle_accuracies.append(segment_accuracy.copy())

        # Check if we're at the end of an OPRO cycle
        days_in_current_cycle = len(self.current_cycle_accuracies)
        is_cycle_end = (days_in_current_cycle >= self.opro_cycle_days)
        
        if self.logging_config.accuracy_cycle:
            print(f"🔄 OPRO Cycle: Day {days_in_current_cycle}/{self.opro_cycle_days} of current cycle")

        # Apply OPRO optimization only at the end of each cycle
        if is_cycle_end:
            self.cycle_count += 1
            
            # Calculate average accuracy for this cycle
            cycle_average_accuracies = self.calculate_cycle_average_accuracy()
            
            if self.logging_config.opro_optimization:
                print(f"🧠 OPRO Optimization - End of Cycle {self.cycle_count}:")
            if self.logging_config.accuracy_cycle:
                print(f"   Cycle average accuracies:")
                for segment, accuracy in cycle_average_accuracies.items():
                    print(f"     • {segment}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Store optimization data in history before applying OPRO
            for segment, accuracy in cycle_average_accuracies.items():
                if (segment in self.optimized_prompts and 
                    segment in self.current_hooks):
                    
                    entry = OptimizerEntry(
                        accuracy_score=accuracy,
                        prompt=self.optimized_prompts[segment],
                        hook=self.current_hooks[segment]
                    )
                    
                    # Store with cycle and segment info
                    history_entry = {
                        'day': f"Cycle {self.cycle_count}",  # Current cycle's data
                        'segment': segment,
                        'optimizer_entry': entry
                    }
                    self.opro_optimization_history.append(history_entry)
            
            # Apply OPRO optimization only if we have previous cycle data
            if self.cycle_count > 1:
                # Call OPRO optimization from marketing agent using cycle accuracies
                new_optimized_prompts = self.marketing_agent.apply_opro_step(
                    previous_scores=cycle_average_accuracies,
                    previous_prompts=self.optimized_prompts,
                    previous_hooks=self.current_hooks,
                    k=3,  # Consider top 3 performing entries
                    logging_config=self.logging_config
                )
                
                # Only adopt a new prompt if this cycle's score beats the running best
                for segment, prompt in new_optimized_prompts.items():
                    current_score = cycle_average_accuracies.get(segment, 0.0)
                    if current_score >= self.best_segment_accuracy.get(segment, -1.0):
                        self.optimized_prompts[segment] = prompt
                        if self.logging_config.opro_prompts:
                            print(f"     • {segment}: adopted new prompt (score {current_score:.3f} >= best {self.best_segment_accuracy.get(segment, 0.0):.3f}): {prompt[:100]}...")
                    else:
                        # Revert to the prompt that achieved the best score so far
                        if segment in self.best_segment_prompts:
                            self.optimized_prompts[segment] = self.best_segment_prompts[segment]
                        if self.logging_config.opro_prompts:
                            print(f"     • {segment}: reverted to best prompt (score {current_score:.3f} < best {self.best_segment_accuracy.get(segment, 0.0):.3f})")
            else:
                if self.logging_config.opro_optimization:
                    print(f"ℹ️  Skipping OPRO optimization (Cycle {self.cycle_count} - need previous cycle data)")
            
            # Store current cycle's average as previous cycle for next iteration
            self.previous_cycle_average_accuracy = cycle_average_accuracies.copy()

            # Update running best (monotonically non-decreasing)
            for segment, accuracy in cycle_average_accuracies.items():
                if accuracy > self.best_segment_accuracy.get(segment, -1.0):
                    self.best_segment_accuracy[segment] = accuracy
                    self.best_segment_prompts[segment] = self.optimized_prompts[segment]
                    self.best_segment_hooks[segment] = self.current_hooks.get(segment, "")
            
            # Reset for next cycle
            self.current_cycle_accuracies = []
        else:
            if self.logging_config.accuracy_cycle:
                print(f"ℹ️  Continuing current OPRO cycle (Day {days_in_current_cycle}/{self.opro_cycle_days})")

        # End of simulation step - advance to next day
        self.advance_simulation_day()
        if self.logging_config.day_advancement:
            print(f"⏭️  Advanced to Day {self.current_simulation_day}")
    
    def advance_simulation_day(self):
        """Advance the simulation to the next day."""
        self.current_simulation_day += 1
        self.shop_component.advance_simulation_day()
    
    def get_current_simulation_day(self) -> int:
        """Get the current simulation day."""
        return self.current_simulation_day
    
    def calculate_segment_hook_accuracy(self, day_number: Optional[int] = None) -> Dict[str, float]:
        """
        For each customer segment, calculate what fraction of customers visited the shop today.

        A score of 1.0 means every customer in that segment visited.
        A score of 0.0 means nobody visited.
        This is used as a signal for how well the marketing hook worked for that segment.

        Args:
            day_number: Simulation day (defaults to current day)

        Returns:
            Dict[str, float]: e.g. {'H': 0.8, 'N': 0.4, ...}  (values between 0.0 and 1.0)
        """
        if day_number is None:
            day_number = self.current_simulation_day

        # Step 1 — load every customer and group them by their segment label (e.g. 'H', 'N', 'R')
        all_customers = self.customer_registry.get_all_customers()
        segment_counts = {}
        for customer in all_customers:
            segment = customer.segment
            if segment not in segment_counts:
                # First time we see this segment: start both counters at zero
                segment_counts[segment] = {'total': 0, 'visited': 0}
            segment_counts[segment]['total'] += 1

        # Step 2 — find out which customers actually visited the shop today
        shop_visits = self.shop_component.get_visits_for_day(day_number)
        visited_customer_ids = {visit.cid for visit in shop_visits}  # a fast lookup set

        # Step 3 — for each customer who visited, add 1 to their segment's visited counter
        for customer in all_customers:
            if customer.cid in visited_customer_ids:
                segment_counts[customer.segment]['visited'] += 1

        # Step 4 — turn the raw counts into a percentage (visited / total) for each segment
        segment_accuracy = {}
        for segment, counts in segment_counts.items():
            if counts['total'] > 0:
                segment_accuracy[segment] = counts['visited'] / counts['total']
            else:
                segment_accuracy[segment] = 0.0

        return segment_accuracy
    
    def debug_segment_accuracy(self, day_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Debug version of segment accuracy calculation with detailed output.
        
        Args:
            day_number: Simulation day (defaults to current day)
            
        Returns:
            Dict with debug information and calculated accuracy
        """
        if day_number is None:
            day_number = self.current_simulation_day
        
        print(f"\n🔍 DEBUG: Segment Accuracy Calculation for Day {day_number}")
        print("=" * 60)
        
        # Get all customers grouped by segment
        all_customers = self.customer_registry.get_all_customers()
        print(f"Total customers in registry: {len(all_customers)}")
        
        segment_counts = {}
        
        # Count total customers per segment
        for customer in all_customers:
            segment = customer.segment
            print(f"Customer {customer.cid} ({customer.name}): Segment = '{segment}'")
            if segment not in segment_counts:
                segment_counts[segment] = {'total': 0, 'visited': 0}
            segment_counts[segment]['total'] += 1
        
        print(f"\nSegment distribution:")
        for segment, counts in segment_counts.items():
            print(f"  {segment}: {counts['total']} customers")
        
        # Get all shop visits for the day
        shop_visits = self.shop_component.get_visits_for_day(day_number)
        print(f"\nShop visits for day {day_number}: {len(shop_visits)}")
        
        visited_customer_ids = set()
        for visit in shop_visits:
            print(f"  Visit: Customer {visit.cid} on day {visit.day_number}")
            visited_customer_ids.add(visit.cid)
        
        print(f"Visited customer IDs: {visited_customer_ids}")
        
        # Count customers who visited per segment
        print(f"\nMatching customers to visits:")
        for customer in all_customers:
            if customer.cid in visited_customer_ids:
                segment = customer.segment
                print(f"  ✅ {customer.name} (CID: {customer.cid}, Segment: {segment}) visited")
                if segment in segment_counts:
                    segment_counts[segment]['visited'] += 1
            else:
                print(f"  ❌ {customer.name} (CID: {customer.cid}, Segment: {customer.segment}) did NOT visit")
        
        # Calculate accuracy scores
        segment_accuracy = {}
        print(f"\nSegment accuracy calculation:")
        for segment, counts in segment_counts.items():
            if counts['total'] > 0:
                accuracy = counts['visited'] / counts['total']
                segment_accuracy[segment] = accuracy
                print(f"  {segment}: {counts['visited']}/{counts['total']} = {accuracy:.3f} ({accuracy*100:.1f}%)")
            else:
                segment_accuracy[segment] = 0.0
                print(f"  {segment}: 0/0 = 0.000 (0.0%)")
        
        print("=" * 60)
        
        debug_info = {
            'day_number': day_number,
            'total_customers': len(all_customers),
            'total_visits': len(shop_visits),
            'segment_counts': segment_counts,
            'visited_customer_ids': list(visited_customer_ids),
            'segment_accuracy': segment_accuracy,
            'shop_visits_raw': [{'cid': v.cid, 'day': v.day_number} for v in shop_visits]
        }
        
        return debug_info

    def calculate_segment_hook_accuracy_difference(self, day_number: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate the difference in hook accuracy for each customer segment compared to the previous day.
        
        This provides a more meaningful metric for OPRO optimization by measuring the actual improvement
        or degradation in hook performance rather than just absolute accuracy values.
        
        Accuracy difference = current_day_accuracy - previous_day_accuracy
        
        Args:
            day_number: Simulation day (defaults to current day)
            
        Returns:
            Dict[str, float]: Dictionary mapping segment to accuracy difference (-1.0 to 1.0)
                             Positive values indicate improvement, negative values indicate degradation
        """
        if day_number is None:
            day_number = self.current_simulation_day
        
        # Get current day's accuracy
        current_accuracy = self.calculate_segment_hook_accuracy(day_number)
        
        # Calculate differences compared to previous day
        segment_accuracy_differences = {}
        
        for segment, current_acc in current_accuracy.items():
            if segment in self.previous_day_segment_accuracy:
                # Calculate difference: positive = improvement, negative = degradation
                previous_acc = self.previous_day_segment_accuracy[segment]
                difference = current_acc - previous_acc
                segment_accuracy_differences[segment] = difference
            else:
                # First day - no previous data, return current accuracy as baseline
                segment_accuracy_differences[segment] = current_acc
        
        # Update previous day accuracy for next iteration
        self.previous_day_segment_accuracy = current_accuracy.copy()
        
        return segment_accuracy_differences

    def calculate_cycle_average_accuracy_differences(self) -> Dict[str, float]:
        """
        Calculate the average accuracy differences between the current cycle and the previous cycle.
        
        This computes: (average accuracy of current cycle) - (average accuracy of previous cycle)
        for each customer segment over the OPRO cycle period (D days).
        
        Returns:
            Dict[str, float]: Dictionary mapping segment to average accuracy difference
                             Positive values indicate cycle improvement, negative values indicate degradation
        """
        if not self.current_cycle_accuracies:
            return {}
        
        # Calculate average accuracy for current cycle
        current_cycle_average = {}
        
        # Initialize segment totals
        for day_accuracy in self.current_cycle_accuracies:
            for segment in day_accuracy:
                if segment not in current_cycle_average:
                    current_cycle_average[segment] = 0.0
        
        # Sum up accuracies across all days in current cycle
        for segment in current_cycle_average:
            total_accuracy = 0.0
            days_count = 0
            for day_accuracy in self.current_cycle_accuracies:
                if segment in day_accuracy:
                    total_accuracy += day_accuracy[segment]
                    days_count += 1
            
            if days_count > 0:
                current_cycle_average[segment] = total_accuracy / days_count
        
        # Calculate differences compared to previous cycle
        cycle_accuracy_differences = {}
        
        for segment, current_avg in current_cycle_average.items():
            if segment in self.previous_cycle_average_accuracy:
                # Calculate difference: current cycle avg - previous cycle avg
                previous_avg = self.previous_cycle_average_accuracy[segment]
                difference = current_avg - previous_avg
                cycle_accuracy_differences[segment] = difference
            else:
                # First cycle - use current average as baseline
                cycle_accuracy_differences[segment] = current_avg
        
        return cycle_accuracy_differences

    def calculate_cycle_average_accuracy(self) -> Dict[str, float]:
        """
        Calculate the average accuracy for the current cycle.
        
        This computes the average visit rate for each customer segment 
        over the current OPRO cycle period (D days).
        
        Returns:
            Dict[str, float]: Dictionary mapping segment to average accuracy (0.0-1.0)
        """
        if not self.current_cycle_accuracies:
            return {}
        
        # Calculate average accuracy for current cycle
        current_cycle_average = {}
        
        # Initialize segment totals
        for day_accuracy in self.current_cycle_accuracies:
            for segment in day_accuracy:
                if segment not in current_cycle_average:
                    current_cycle_average[segment] = 0.0
        
        # Sum up accuracies across all days in current cycle
        for segment in current_cycle_average:
            total_accuracy = 0.0
            days_count = 0
            for day_accuracy in self.current_cycle_accuracies:
                if segment in day_accuracy:
                    total_accuracy += day_accuracy[segment]
                    days_count += 1
            
            if days_count > 0:
                current_cycle_average[segment] = total_accuracy / days_count
        
        return current_cycle_average

    def print_opro_optimization_history(self):
        """
        Print all OPRO optimization history including prompt-accuracy score pairs.
        This shows the complete evolution of prompts and their performance throughout the simulation.
        Uses actual visit rate accuracies rather than differences between cycles.
        """
        if not self.logging_config.final_history:
            return
            
        print("\n" + "="*80)
        print("📊 OPRO OPTIMIZATION HISTORY - All Prompt-Accuracy Score Pairs")
        print(f"   (Cycle-based optimization every {self.opro_cycle_days} days)")
        print("   (Scores show average visit rate accuracies for each cycle: higher = better)")
        print("="*80)
        
        if not self.opro_optimization_history:
            print("No OPRO optimization data available.")
            return
        
        # Group by segment for better organization
        from collections import defaultdict
        history_by_segment = defaultdict(list)
        
        for entry in self.opro_optimization_history:
            segment = entry['segment']
            history_by_segment[segment].append(entry)
        
        # Print organized by segment
        for segment, entries in history_by_segment.items():
            print(f"\n🎯 SEGMENT: {segment}")
            print("-" * 60)
            
            # Sort by cycle (handle both day numbers and cycle strings)
            def sort_key(x):
                day_str = str(x['day'])
                if day_str.startswith('Cycle'):
                    # Extract cycle number from "Cycle X" format
                    try:
                        cycle_num = int(day_str.split()[1])
                        return cycle_num
                    except:
                        return 999  # Put incomplete cycles at end
                else:
                    return int(day_str)
            
            entries.sort(key=sort_key)
            
            for entry in entries:
                day = entry['day']
                optimizer_entry = entry['optimizer_entry']
                
                # Format the accuracy score
                score = optimizer_entry.accuracy_score
                
                print(f"  {day}:")
                print(f"    📈 Visit Rate Accuracy: {score:.3f} ({score*100:.1f}%)")
                print(f"    📝 Prompt: {optimizer_entry.prompt[:120]}...")
                print(f"    🎣 Generated Hook: {optimizer_entry.hook}")
                print()
        
        # Print summary statistics
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 40)
        
        all_scores = [entry['optimizer_entry'].accuracy_score for entry in self.opro_optimization_history]
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(all_scores)
            min_score = min(all_scores)
            high_performers = [s for s in all_scores if s > 0.5]  # Above 50% visit rate
            low_performers = [s for s in all_scores if s < 0.2]   # Below 20% visit rate
            
            print(f"  Total Entries: {len(self.opro_optimization_history)}")
            print(f"  Average Accuracy: {avg_score:.3f} ({avg_score*100:.1f}%)")
            print(f"  Best Accuracy: {max_score:.3f} ({max_score*100:.1f}%)")
            print(f"  Worst Accuracy: {min_score:.3f} ({min_score*100:.1f}%)")
            print(f"  High Performers (>50%): {len(high_performers)}/{len(all_scores)}")
            print(f"  Low Performers (<20%): {len(low_performers)}/{len(all_scores)}")
            
            # Find best performing prompt
            best_entry = max(self.opro_optimization_history, 
                           key=lambda x: x['optimizer_entry'].accuracy_score)
            print(f"  Best Performing:")
            print(f"    Day {best_entry['day']}, Segment: {best_entry['segment']}")
            print(f"    Accuracy: {best_entry['optimizer_entry'].accuracy_score:.3f} ({best_entry['optimizer_entry'].accuracy_score*100:.1f}%)")
            print(f"    Hook: {best_entry['optimizer_entry'].hook}")
        
        print("="*80)

    def get_daily_metrics(self, day_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Get daily metrics for analysis.
        
        Args:
            day_number: Simulation day (defaults to current day)
            
        Returns:
            dict: Daily metrics including visits, transactions, etc.
        """
        if day_number is None:
            day_number = self.current_simulation_day
        
        return {
            'day': day_number,
            'total_visits': self.shop_component.get_visit_count_for_day(day_number),
            'shop_visits': self.shop_component.get_visits_for_day(day_number),
            # Additional metrics can be added here
        }
    
    def snapshot_day(self, day_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a comprehensive snapshot of all metrics and data for a specific day.
        
        This function returns a JSON-like dictionary containing:
        - Day number
        - Segment accuracy scores for the day
        - Current hooks used for each segment
        - Historical hooks and their performance scores
        - Shopping metrics (visits, transactions, etc.)
        - OPRO optimization data
        
        Args:
            day_number: Simulation day to snapshot (defaults to current day - 1)
            
        Returns:
            Dict[str, Any]: Comprehensive daily snapshot containing all metrics
        """
        if day_number is None:
            day_number = self.current_simulation_day
        
        # Use actual per-day accuracy for day-by-day dashboard updates
        segment_accuracy = self.calculate_segment_hook_accuracy(day_number)
        
        # Get current hooks for each segment
        current_hooks_snapshot = self.current_hooks.copy() if self.current_hooks else {}
        
        # Get all customers for additional metrics
        all_customers = self.customer_registry.get_all_customers()
        total_customers = len(all_customers)
        
        # Calculate segment distribution
        segment_distribution = {}
        for customer in all_customers:
            segment = customer.segment
            if segment not in segment_distribution:
                segment_distribution[segment] = 0
            segment_distribution[segment] += 1
        
        # Get shop visits for the day
        shop_visits = self.shop_component.get_visits_for_day(day_number)
        total_visits = len(shop_visits)
        
        # Calculate visit rate
        visit_rate = (total_visits / total_customers) if total_customers > 0 else 0.0
        
        # Get visited customer IDs for segment-level visit analysis
        visited_customer_ids = {visit.cid for visit in shop_visits}
        
        # Calculate segment-level visit metrics
        segment_visit_metrics = {}
        for segment in segment_distribution.keys():
            segment_customers = [c for c in all_customers if c.segment == segment]
            segment_total = len(segment_customers)
            segment_visited = len([c for c in segment_customers if c.cid in visited_customer_ids])
            segment_visit_rate = (segment_visited / segment_total) if segment_total > 0 else 0.0
            
            segment_visit_metrics[segment] = {
                'total_customers': segment_total,
                'visited_customers': segment_visited,
                'visit_rate': segment_visit_rate,
                'accuracy': segment_accuracy.get(segment, 0.0)
            }
        
        # Prepare historical hooks data with scores
        historical_hooks = []
        for entry in self.opro_optimization_history:
            hook_data = {
                'day_or_cycle': entry['day'],
                'segment': entry['segment'],
                'hook': entry['optimizer_entry'].hook,
                'accuracy_score': entry['optimizer_entry'].accuracy_score,
                'prompt': entry['optimizer_entry'].prompt[:100] + "..." if len(entry['optimizer_entry'].prompt) > 100 else entry['optimizer_entry'].prompt
            }
            historical_hooks.append(hook_data)
        
        # Get current optimized prompts
        current_prompts = self.optimized_prompts.copy() if hasattr(self, 'optimized_prompts') else {}
        
        # Create comprehensive snapshot
        snapshot = {
            'day_number': day_number,
            'simulation_metadata': {
                'current_simulation_day': self.current_simulation_day,
                'simulation_duration': self.simulation_duration,
                'opro_cycle_days': self.opro_cycle_days,
                'cycle_count': self.cycle_count,
                'total_customers': total_customers
            },
            'daily_metrics': {
                'total_visits': total_visits,
                'total_customers': total_customers,
                'visit_rate': visit_rate,
                'visit_rate_percentage': visit_rate * 100
            },
            'segment_accuracy': segment_accuracy,
            'segment_visit_metrics': segment_visit_metrics,
            'segment_distribution': segment_distribution,
            'current_hooks': current_hooks_snapshot,
            'current_prompts': current_prompts,
            'historical_hooks_and_scores': historical_hooks,
            'best_segment_accuracy': self.best_segment_accuracy.copy(),
        'best_segment_hooks': self.best_segment_hooks.copy(),
        'opro_data': {
                'total_optimization_entries': len(self.opro_optimization_history),
                'current_cycle_accuracies_count': len(self.current_cycle_accuracies),
                'previous_cycle_average_accuracy': self.previous_cycle_average_accuracy.copy() if self.previous_cycle_average_accuracy else {}
            },
            'snapshot_timestamp': day_number,
            'data_summary': {
                'segments_tracked': list(segment_accuracy.keys()),
                'hooks_generated': len(current_hooks_snapshot),
                'historical_entries': len(historical_hooks),
                'best_performing_segment': max(segment_accuracy.items(), key=lambda x: x[1])[0] if segment_accuracy else None,
                'worst_performing_segment': min(segment_accuracy.items(), key=lambda x: x[1])[0] if segment_accuracy else None,
                'average_accuracy': sum(segment_accuracy.values()) / len(segment_accuracy) if segment_accuracy else 0.0
            }
        }
        
        return snapshot

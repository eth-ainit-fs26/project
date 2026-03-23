"""
Customer interaction utilities: visit probability updates and response generation.
"""
from models.models import Customer
from services.llm_service import LLMService
from agents.marketing_agent import MarketingEmail


class CustomerUpdater:
    def __init__(self):
        self.visit_probability_adjustment = 0.05

    def update_visit_probability(self, customer: Customer, new_probability: float) -> None:
        """Update the visit probability for a customer, clamped to [0, 1]."""
        customer.visit_probability = max(0.0, min(1.0, new_probability))


class CustomerResponse:
    def __init__(self, customer: Customer, llm_service: LLMService):
        self.customer = customer
        self.llm_service = llm_service

    def create_response(self, marketing_email: MarketingEmail) -> str:
        """Create a customer response based on marketing email."""
        # Placeholder — would be replaced with actual LLM-based response logic
        return "Thank you for your email. I'm interested in learning more."

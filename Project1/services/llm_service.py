from typing import Optional
from abc import ABC, abstractmethod
import os


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt."""
        pass


class LlamaFileProvider(LLMProvider):
    """LLM Provider that loads a local model file using llama_cpp."""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 8, n_batch: int = 256, verbose: bool = False):
        """
        Initialize the LlamaFileProvider with a model file path.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads to use
            n_batch: Batch size for generation
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.verbose = verbose
        self.llm = None
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Llama model."""
        try:
            from llama_cpp import Llama
            
            if self.llm is None:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_batch=self.n_batch,
                    verbose=self.verbose
                )
        except ImportError:
            raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt using the loaded model."""
        if self.llm is None:
            self._initialize_model()
        
        # Extract parameters with defaults
        max_tokens = kwargs.get('max_tokens', 200)
        temperature = kwargs.get('temperature', 0.7)
        
        # Create messages format for chat completion
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if self.llm is not None:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response["choices"][0]["message"]["content"].strip()
            else:
                return f"Error: LLM model not initialized"
        except Exception as e:
            return f"Error generating text: {str(e)}"


class LLMService:
    """
    LLM Service for handling language model operations.
    Designed to be provider-agnostic.
    """
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        """Initialize LLM service with optional provider."""
        self.provider = provider
    
    def set_provider(self, provider: LLMProvider):
        """Set the LLM provider."""
        self.provider = provider
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the configured provider."""
        if not self.provider:
            return "LLM provider not configured"
        return self.provider.generate_text(prompt, **kwargs)
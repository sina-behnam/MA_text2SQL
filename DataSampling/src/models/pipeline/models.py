import os
from typing import Dict, List, Tuple, Any, Optional, Union
import openai
import torch
# Optional import for Anthropic API
try:
    from anthropic import Anthropic, NOT_GIVEN
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Optional imports for local models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Set your API key
API_KEY = 'your_api_key_here'
os.environ["TOGETHER_API_KEY"] = API_KEY

class ModelProvider:
    """Base class for model providers (API or local)"""
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response from the model.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        raise NotImplementedError("Subclasses must implement this method")

class TogetherAIProvider(ModelProvider):
    """Provider for Together.ai API-based models"""
    
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize the Together.ai provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for Together.ai
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", API_KEY)
        
        # Initialize OpenAI client with Together.ai API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the Together.ai API.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        
        return response.choices[0].message.content

class OpenAIProvider(ModelProvider):
    """Provider for OpenAI API-based models"""
    
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for OpenAI
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key
        )
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        
        return response.choices[0].message.content
    
class AnthropicProvider(ModelProvider):
    """Provider for Anthropic API models (Claude)"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None, max_tokens: int = 1024, extended_thinking: bool = False):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_name: Name of the Claude model to use (e.g., "claude-3-opus-20240229")
            api_key: API key for Anthropic
            max_tokens: Maximum number of tokens to generate
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "To use Anthropic models, you need to install the anthropic package: "
                "pip install anthropic"
            )
            
        self.model_name = model_name
        self.api_key = api_key 
        self.max_tokens = max_tokens
        self.extended_thinking = extended_thinking
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Please provide it as a parameter "
                "or set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)

    @staticmethod
    def get_output_response(response):
        thinking_messages = None
        response_messages = None

        for block in response.content:
            if block.type == "thinking":
                thinking_messages = block.thinking
            elif block.type == "redacted_thinking":
                thinking_messages = 'IT IS REDACTED'
            elif block.type == "text":
                response_messages = block.text

        message = f'<think>\n{thinking_messages}\n</think>\n\n' if thinking_messages else ''
        message += response_messages

        return message
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the Anthropic API.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        try:
            # Create message using Anthropic API format
            response = self.client.messages.create(
                model=self.model_name,
                system=system_message,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000,
                } if self.extended_thinking else NOT_GIVEN,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                max_tokens= 4000 if self.extended_thinking else 1024, # It always should be higher than the budget tokens for thinking
            )
            
            # Extract the response message
            return AnthropicProvider.get_output_response(response)
    
        except Exception as e:
            # Handle API errors
            error_message = f"Anthropic API error: {str(e)}"
            print(error_message)
            return error_message

class LocalHuggingFaceProvider(ModelProvider):
    """Provider for local HuggingFace models"""
    
    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 512, trust_remote_code: bool = True):
        """
        Initialize the local HuggingFace provider.
        
        Args:
            model_path: Path or name of the model to load from HuggingFace
            device: Device to use ("cpu", "cuda", "auto")
            max_new_tokens: Maximum number of tokens to generate
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "To use local models, you need to install transformers: "
                "pip install transformers torch"
            )
        
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model {model_path} on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=trust_remote_code
        )
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the local HuggingFace model.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        # Format the prompt based on model architecture
        # We'll use a generic format, but this might need customization for specific models
        if "llama" in self.model_path.lower():
            # Llama format
            prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]"
        elif "mistral" in self.model_path.lower():
            # Mistral format
            prompt = f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
        elif "gemma" in self.model_path.lower():
            # Gemma format
            prompt = f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model"
        else:
            # Generic format
            prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
        
        # Generate text
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (removing the prompt)
        if "llama" in self.model_path.lower() or "mistral" in self.model_path.lower():
            # For Llama and Mistral format
            response = response.split("[/INST]")[-1].strip()
        elif "gemma" in self.model_path.lower():
            # For Gemma format
            response = response.split("<start_of_turn>model")[-1].strip()
        else:
            # For generic format
            response = response.split("Assistant:")[-1].strip()
        
        return response
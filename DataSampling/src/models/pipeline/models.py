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
    
    def __init__(self, model_name: str, api_key: str = None, max_tokens: int = 1024):
        """
        Initialize the Together.ai provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for Together.ai
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", API_KEY)
        self.max_tokens = max_tokens
        self.total_limit = 8193  # Actual Together.ai limit
        
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
        input_text = system_message + user_message
        estimated_input_tokens = len(input_text) // 4  # rough estimate
        
        # Leave buffer and adjust max_tokens if needed
        available_tokens = 8100 - estimated_input_tokens - 100  # 100 token buffer
        actual_max_tokens = min(self.max_tokens, available_tokens)
        actual_max_tokens = max(actual_max_tokens, 50)  # Minimum 50
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens= actual_max_tokens,
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
    
class ConversationalModelProvider:
    """Wrapper to add conversation context to model providers with proper token management"""
    
    def __init__(self, base_provider: ModelProvider):
        self.base_provider = base_provider
        self.conversation_history = []
        
    def start_new_conversation(self):
        """Start a new conversation by clearing history"""
        self.conversation_history = []
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def calculate_available_tokens(self, messages: List[Dict], provider_type: str) -> int:
        """Calculate available tokens for response based on provider limits"""
        total_input = ""
        for msg in messages:
            total_input += msg["content"] + " "
        
        input_tokens = self.estimate_tokens(total_input)
        
        # Provider-specific limits
        if isinstance(self.base_provider, TogetherAIProvider):
            max_context = 8193
            buffer = 100
        elif isinstance(self.base_provider, OpenAIProvider):
            # Most OpenAI models have higher limits, but being conservative
            max_context = 8000
            buffer = 200
        elif isinstance(self.base_provider, AnthropicProvider):
            # Claude has very high context limits
            max_context = 100000
            buffer = 1000
        else:
            # Local models - conservative estimate
            max_context = 4000
            buffer = 200
        
        available = max_context - input_tokens - buffer
        return max(available, 50)  # Minimum 50 tokens
    
    def truncate_conversation_if_needed(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Truncate conversation history if it's too long"""
        total_tokens = sum(self.estimate_tokens(msg["content"]) for msg in messages)
        
        if total_tokens <= max_tokens:
            return messages
        
        # Keep system message and recent messages
        system_msg = [msg for msg in messages if msg["role"] == "system"]
        user_assistant_msgs = [msg for msg in messages if msg["role"] in ["user", "assistant"]]
        
        # Keep the most recent messages
        truncated = system_msg.copy()
        current_tokens = sum(self.estimate_tokens(msg["content"]) for msg in system_msg)
        
        # Add messages from most recent backwards
        for msg in reversed(user_assistant_msgs):
            msg_tokens = self.estimate_tokens(msg["content"])
            if current_tokens + msg_tokens <= max_tokens:
                truncated.insert(-len([m for m in truncated if m["role"] != "system"]), msg)
                current_tokens += msg_tokens
            else:
                break
        
        return truncated
    
    def generate_with_context(self, system_message: str, user_message: str) -> str:
        """Generate response with conversation context and proper token management"""
        # Build messages normally
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Handle different provider types
        if isinstance(self.base_provider, AnthropicProvider):
            return self._generate_anthropic(system_message, user_message)
        
        elif isinstance(self.base_provider, (TogetherAIProvider, OpenAIProvider)):
            return self._generate_openai_format(system_message, user_message)
        
        elif isinstance(self.base_provider, LocalHuggingFaceProvider):
            return self._generate_local(system_message, user_message)
        
        else:
            # Fallback to base provider
            return self.base_provider.generate(system_message, user_message)
    
    def _generate_anthropic(self, system_message: str, user_message: str) -> str:
        """Handle Anthropic (Claude) models"""
        try:
            # Build messages (Anthropic doesn't include system in messages array)
            messages = []
            
            # Add conversation history
            for msg in self.conversation_history:
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Truncate if needed (Anthropic has high limits, so this is rarely needed)
            messages = self.truncate_conversation_if_needed(messages, 19000)
            
            # Generate response
            response = self.base_provider.client.messages.create(
                model=self.base_provider.model_name,
                system=system_message,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000,
                } if self.base_provider.extended_thinking else NOT_GIVEN,
                messages=messages,
                max_tokens=4000 if self.base_provider.extended_thinking else self.base_provider.max_tokens,
            )
            
            assistant_response = AnthropicProvider.get_output_response(response)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            # Fallback to base provider
            print(f"Anthropic conversation error: {e}")
            return self.base_provider.generate(system_message, user_message)
    
    def _generate_openai_format(self, system_message: str, user_message: str) -> str:
        """Handle OpenAI and Together.ai models"""
        try:
            # Build messages
            messages = [{"role": "system", "content": system_message}]
            
            # Add conversation history
            for msg in self.conversation_history:
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Calculate available tokens and truncate if needed
            if isinstance(self.base_provider, TogetherAIProvider):
                max_context_tokens = 6000  # Conservative for Together.ai
            else:
                max_context_tokens = 6000  # Conservative for OpenAI
                
            messages = self.truncate_conversation_if_needed(messages, max_context_tokens)
            
            # Instead of calling client directly, use base provider's generate method
            # This ensures TogetherAIProvider's token management is always used
            
            # Extract system message (first message)
            final_system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    final_system_message = msg["content"]
                elif msg["role"] in ["user", "assistant"]:
                    user_messages.append(f"{msg['role'].title()}: {msg['content']}")
            
            # Combine conversation history and current message
            if user_messages:
                # Include conversation context in user message
                final_user_message = "\n\n".join(user_messages)
            else:
                final_user_message = user_message
            
            # Use base provider's generate method (this handles token limits properly)
            assistant_response = self.base_provider.generate(final_system_message, final_user_message)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            # Fallback to base provider
            print(f"OpenAI format conversation error: {e}")
            return self.base_provider.generate(system_message, user_message)
    
    def _generate_local(self, system_message: str, user_message: str) -> str:
        """Handle local HuggingFace models (limited conversation support)"""
        try:
            # For local models, we'll concatenate the conversation history
            # but keep it limited due to context constraints
            
            conversation_text = ""
            
            # Add recent conversation history (keep it short)
            recent_history = self.conversation_history[-4:]  # Last 2 exchanges
            for msg in recent_history:
                if msg["role"] == "user":
                    conversation_text += f"User: {msg['content']}\n"
                else:
                    conversation_text += f"Assistant: {msg['content']}\n"
            
            # Combine with current message
            if conversation_text:
                enhanced_user_message = f"Previous conversation:\n{conversation_text}\nCurrent question: {user_message}"
            else:
                enhanced_user_message = user_message
            
            # Generate response
            assistant_response = self.base_provider.generate(system_message, enhanced_user_message)
            
            # Update conversation history (keep it limited)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Keep only recent history for local models
            if len(self.conversation_history) > 8:  # 4 exchanges
                self.conversation_history = self.conversation_history[-8:]
            
            return assistant_response
            
        except Exception as e:
            # Fallback to base provider
            print(f"Local model conversation error: {e}")
            return self.base_provider.generate(system_message, user_message)
    
    def get_conversation_length(self) -> int:
        """Get the current conversation length"""
        return len(self.conversation_history)
    
    def get_conversation_tokens(self) -> int:
        """Estimate total tokens in conversation history"""
        total_text = ""
        for msg in self.conversation_history:
            total_text += msg["content"] + " "
        return self.estimate_tokens(total_text)
    
class ConversationalModelProvider2:
    """
    Model provider that uses server-side conversation threads where possible,
    eliminating the need for client-side conversation history management.
    """
    
    def __init__(self, base_provider: ModelProvider):
        self.base_provider = base_provider
        self.thread_id = None
        self.assistant_id = None
        
    def start_new_conversation(self):
        """Start a new conversation - creates new thread for OpenAI, resets others"""
        if isinstance(self.base_provider, OpenAIProvider):
            try:
                # Create new thread on OpenAI's servers
                thread = self.base_provider.client.beta.threads.create()
                self.thread_id = thread.id
                
                # Create or get assistant (you might want to create this once and reuse)
                assistant = self.base_provider.client.beta.assistants.create(
                    name="SQL Expert",
                    instructions="You are a database expert specializing in SQL query generation.",
                    model=self.base_provider.model_name,
                    tools=[]
                )
                self.assistant_id = assistant.id
                
                print(f"Created OpenAI thread: {self.thread_id}")
                
            except Exception as e:
                print(f"Failed to create OpenAI thread: {e}. Falling back to regular chat.")
                self.thread_id = None
                self.assistant_id = None
        else:
            # For non-OpenAI providers, just reset
            self.thread_id = None
            self.assistant_id = None
    
    def generate_with_context(self, system_message: str, user_message: str) -> str:
        """
        Generate response using server-side threads where available,
        otherwise use stateless calls.
        """
        
        if isinstance(self.base_provider, OpenAIProvider) and self.thread_id:
            return self._generate_openai_assistant(system_message, user_message)
        
        elif isinstance(self.base_provider, AnthropicProvider):
            return self._generate_anthropic_stateless(system_message, user_message)
        
        elif isinstance(self.base_provider, TogetherAIProvider):
            return self._generate_together_stateless(system_message, user_message)
        
        elif isinstance(self.base_provider, LocalHuggingFaceProvider):
            return self._generate_local_stateless(system_message, user_message)
        
        else:
            # Fallback to base provider
            return self.base_provider.generate(system_message, user_message)
    
    def _generate_openai_assistant(self, system_message: str, user_message: str) -> str:
        """Use OpenAI Assistants API with server-side conversation threads"""
        try:
            # Add message to the thread (OpenAI maintains history)
            message = self.base_provider.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=user_message
            )
            
            # Run the assistant
            run = self.base_provider.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                additional_instructions=system_message
            )
            
            # Wait for completion
            import time
            while run.status in ['queued', 'in_progress']:
                time.sleep(1)
                run = self.base_provider.client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the assistant's response
                messages = self.base_provider.client.beta.threads.messages.list(
                    thread_id=self.thread_id
                )
                
                # Get the latest assistant message
                for msg in messages.data:
                    if msg.role == 'assistant':
                        return msg.content[0].text.value
                
                return "No response from assistant"
            
            else:
                print(f"Assistant run failed with status: {run.status}")
                # Fallback to regular chat
                return self._generate_openai_fallback(system_message, user_message)
                
        except Exception as e:
            print(f"OpenAI Assistant error: {e}. Falling back to regular chat.")
            return self._generate_openai_fallback(system_message, user_message)
    
    def _generate_openai_fallback(self, system_message: str, user_message: str) -> str:
        """Fallback to regular OpenAI chat API (stateless)"""
        response = self.base_provider.client.chat.completions.create(
            model=self.base_provider.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=getattr(self.base_provider, 'max_tokens', 1024)
        )
        return response.choices[0].message.content
    
    def _generate_anthropic_stateless(self, system_message: str, user_message: str) -> str:
        """Anthropic doesn't have server-side threads, use stateless calls"""
        try:
            response = self.base_provider.client.messages.create(
                model=self.base_provider.model_name,
                system=system_message,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000,
                } if self.base_provider.extended_thinking else NOT_GIVEN,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=4000 if self.base_provider.extended_thinking else self.base_provider.max_tokens,
            )
            
            return AnthropicProvider.get_output_response(response)
            
        except Exception as e:
            print(f"Anthropic error: {e}")
            return self.base_provider.generate(system_message, user_message)
    
    def _generate_together_stateless(self, system_message: str, user_message: str) -> str:
        """Together.ai doesn't have server-side threads, use stateless calls"""
        try:
            # Calculate safe max_tokens to avoid 422 error
            input_text = system_message + user_message
            estimated_input_tokens = len(input_text) // 4
            available_tokens = 8193 - estimated_input_tokens - 100  # Buffer
            max_tokens = min(getattr(self.base_provider, 'max_tokens', 1024), available_tokens)
            max_tokens = max(max_tokens, 50)  # Minimum 50 tokens
            
            response = self.base_provider.client.chat.completions.create(
                model=self.base_provider.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Together.ai error: {e}")
            return self.base_provider.generate(system_message, user_message)
    
    def _generate_local_stateless(self, system_message: str, user_message: str) -> str:
        """Local models are inherently stateless"""
        return self.base_provider.generate(system_message, user_message)
    
    def get_conversation_length(self) -> int:
        """
        Get conversation length from server-side thread if available
        """
        if isinstance(self.base_provider, OpenAIProvider) and self.thread_id:
            try:
                messages = self.base_provider.client.beta.threads.messages.list(
                    thread_id=self.thread_id
                )
                return len(messages.data)
            except:
                return 0
        else:
            return 0
    
    def get_conversation_tokens(self) -> int:
        """
        Estimate conversation tokens from server-side thread if available
        """
        if isinstance(self.base_provider, OpenAIProvider) and self.thread_id:
            try:
                messages = self.base_provider.client.beta.threads.messages.list(
                    thread_id=self.thread_id
                )
                total_text = ""
                for msg in messages.data:
                    total_text += msg.content[0].text.value + " "
                return len(total_text) // 4  # Rough estimation
            except:
                return 0
        else:
            return 0
    
    def cleanup_thread(self):
        """Clean up server-side resources"""
        if isinstance(self.base_provider, OpenAIProvider) and self.thread_id:
            try:
                # Delete the thread to clean up
                self.base_provider.client.beta.threads.delete(self.thread_id)
                if self.assistant_id:
                    self.base_provider.client.beta.assistants.delete(self.assistant_id)
                print(f"Cleaned up OpenAI thread: {self.thread_id}")
            except Exception as e:
                print(f"Error cleaning up thread: {e}")
            finally:
                self.thread_id = None
                self.assistant_id = None
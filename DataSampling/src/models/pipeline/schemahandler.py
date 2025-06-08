from typing import Dict, List, Tuple
# from models import ConversationalModelProvider, ModelProvider
import spacy

class SequentialSchemaHandler:
    """
    Handles large database schemas by sending them in sequential chunks
    to avoid token limit issues with language models.
    """
    
    def __init__(self, model_provider, max_tokens_per_chunk: int = 4000, token_threshold: int = 6000,
                 nlp_model="en_core_web_sm"):
        """
        Initialize the sequential schema handler
        
        Args:
            model_provider: The conversational model provider
            max_tokens_per_chunk: Maximum tokens per schema chunk
            token_threshold: Token threshold to trigger chunking
        """
        self.model_provider = model_provider
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.token_threshold = token_threshold
        # NLP model for token estimation (if needed)
        self.nlp_model = spacy.load(nlp_model) if nlp_model else None

    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def estimate_tokens_spacy(self, text: str) -> int:
        """
        Estimate tokens using spaCy for more accurate tokenization
        
        Args:
            text: Input text to estimate tokens
            
        Returns:
            Estimated number of tokens
        """
        if self.nlp_model is not None:
            doc = self.nlp_model(text)
            return len(doc)
        else:
            # Fallback to simple estimation if spaCy model is not available
            return self.estimate_tokens(text)

    def is_schema_large(self, schema_info: Dict) -> bool:
        """
        Check if schema needs to be sent in chunks
        
        Args:
            schema_info: Database schema information
            
        Returns:
            True if schema is large and needs chunking
        """
        # Estimate total schema size
        total_text = f"Database: {schema_info['database']['name']}\n"
        
        for schema in schema_info.get('schemas', []):
            table_text = f"Table: {schema['table_name']}\n{schema['DDL']}\n"
            if schema.get('description'):
                table_text += f"Description: {schema['description']}\n"
            total_text += table_text
        
        estimated_tokens = self.estimate_tokens_spacy(total_text)
        return estimated_tokens > self.token_threshold
    
    def chunk_schema(self, schema_info: Dict) -> List[Dict]:
        """
        Break schema into manageable chunks
        
        Args:
            schema_info: Database schema information
            
        Returns:
            List of schema chunks
        """
        schemas = schema_info.get('schemas', [])
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Base info tokens
        base_info = f"Database: {schema_info['database']['name']}\n"
        base_tokens = self.estimate_tokens_spacy(base_info)
        
        for schema in schemas:
            # Calculate tokens for this table
            table_text = f"Table: {schema['table_name']}\n{schema['DDL']}\n"
            if schema.get('description'):
                table_text += f"Description: {schema['description']}\n"
            
            table_tokens = self.estimate_tokens_spacy(table_text)
            
            # Check if adding this table would exceed chunk limit
            if current_tokens + table_tokens + base_tokens > self.max_tokens_per_chunk and current_chunk:
                # Save current chunk and start new one
                chunks.append({
                    'database': schema_info['database'],
                    'schemas': current_chunk.copy(),
                    'chunk_info': {
                        'chunk_number': len(chunks) + 1,
                        'is_continuation': len(chunks) > 0
                    }
                })
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(schema)
            current_tokens += table_tokens
        
        # Add remaining schemas as final chunk
        if current_chunk:
            chunks.append({
                'database': schema_info['database'],
                'schemas': current_chunk,
                'chunk_info': {
                    'chunk_number': len(chunks) + 1,
                    'is_continuation': len(chunks) > 0
                }
            })
        
        return chunks
    
    def create_chunk_prompt(self, chunk: Dict, total_chunks: int) -> str:
        """
        Create prompt for a schema chunk
        
        Args:
            chunk: Schema chunk information
            total_chunks: Total number of chunks
            
        Returns:
            Formatted prompt for this chunk
        """
        db_name = chunk['database']['name']
        db_type = chunk['database'].get('type', 'sqlite').upper()
        chunk_num = chunk['chunk_info']['chunk_number']
        is_continuation = chunk['chunk_info']['is_continuation']
        
        if is_continuation:
            prompt = f"""This is part {chunk_num} of {total_chunks} of the "{db_name}" database schema (Database Type: {db_type}).

**IMPORTANT: This continues the same database schema from the previous parts. Please add these tables to your understanding of the "{db_name}" database.**

Here are additional tables from the same database:

"""
        else:
            prompt = f"""You are working with the "{db_name}" database (Database Type: {db_type}).

**IMPORTANT: This database schema is large and will be sent in {total_chunks} parts. This is part {chunk_num} of {total_chunks}.**

Here's the database schema (part {chunk_num}):

"""
        
        # Add table definitions
        for schema in chunk['schemas']:
            prompt += f"## Table: {schema['table_name']}\n"
            if schema.get('description'):
                prompt += f"Description: {schema['description']}\n"
            prompt += f"```sql\n{schema['DDL']}\n```\n\n"
        
        if chunk_num == total_chunks:
            # Final chunk
            prompt += f"""**This completes the "{db_name}" database schema. You now have the complete database structure with all {sum(len(c['schemas']) for c in self.current_chunks)} tables.**

Please confirm that you understand the complete database schema and are ready to answer questions about it."""
        else:
            # Intermediate chunk
            prompt += f"""**More tables from this database will follow in the next part. Please acknowledge that you've understood these tables and are ready for part {chunk_num + 1}.**"""
        
        return prompt
    
    def send_schema_sequentially(self, schema_info: Dict, system_message: str) -> str:
        """
        Send large schema in sequential chunks
        
        Args:
            schema_info: Database schema information
            system_message: System message for the model
            
        Returns:
            Final response after all chunks are sent
        """
        # Break schema into chunks
        chunks = self.chunk_schema(schema_info)
        self.current_chunks = chunks  # Store for prompt creation
        total_chunks = len(chunks)
        
        print(f"Large schema detected. Sending in {total_chunks} chunks...")
        
        responses = []
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = self.create_chunk_prompt(chunk, total_chunks)

            # Send chunk to model
            if i == 0:
                # First chunk includes system message
                response = self.model_provider.generate_with_context(system_message, chunk_prompt)
            else:
                # Subsequent chunks are user messages only
                response = self.model_provider.generate_with_context("", chunk_prompt)
            
            responses.append(response)
            print(f"Sent schema chunk {i + 1}/{total_chunks}")
        
        # Return the final response (after all chunks are sent)
        return responses[-1]
    
    def handle_large_schema(self, schema_info: Dict, system_message: str) -> Tuple[bool, str]:
        """
        Main method to handle schema - checks if large and processes accordingly
        
        Args:
            schema_info: Database schema information
            system_message: System message for the model
            
        Returns:
            Tuple of (was_chunked, final_response)
        """
        if self.is_schema_large(schema_info):
            # Schema is large, send in chunks
            final_response = self.send_schema_sequentially(schema_info, system_message)
            return True, final_response
        else:
            # Schema is small enough, return None to indicate normal processing
            return False, ""
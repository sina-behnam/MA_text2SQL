
from typing import Dict, List, Tuple, Optional, Union, Any
import sqlparse

class SchemaAdapter:
    """
    Adapter class to standardize schema access across different dataset formats.
    """
    @staticmethod
    def adapt_schema(schema: Dict, dataset_name: str) -> Dict:
        """
        Adapt a schema to a standardized format based on dataset type.
        
        Args:
            schema: The original schema
            dataset_name: Name of the dataset
            
        Returns:
            Standardized schema dictionary
        """
        if dataset_name.lower() == 'spider':
            return SpiderSchemaAdapter.adapt(schema)
        elif dataset_name.lower() == 'spider2':
            return SpiderSchemaAdapter.adapt(schema)
        elif dataset_name.lower() == 'bird':
            return BirdSchemaAdapter.adapt(schema)
        else:
            # Default - return schema as is
            return schema

class SpiderSchemaAdapter:
    """Spider-specific schema adapter"""
    
    @staticmethod
    def adapt(schema: Dict) -> Dict:
        """
        Adapt Spider schema to standardized format.
        
        Args:
            schema: The Spider schema
            
        Returns:
            Standardized schema dictionary
        """
        # Spider schemas are already in the expected format for most fields
        # Just ensure all expected fields are present
        adapted_schema = schema.copy()
        
        # Ensure all necessary fields are present
        if 'tables' not in adapted_schema:
            adapted_schema['tables'] = []
            
        if 'columns' not in adapted_schema:
            adapted_schema['columns'] = []
            
        if 'table_to_columns' not in adapted_schema:
            # Create table_to_columns mapping from columns if not present
            table_to_columns = {}
            for col in adapted_schema.get('columns', []):
                table_idx = col.get('table_idx', -1)
                if table_idx >= 0:
                    if table_idx not in table_to_columns:
                        table_to_columns[table_idx] = []
                    table_to_columns[table_idx].append(col['id'])
            
            adapted_schema['table_to_columns'] = table_to_columns
            
        if 'foreign_keys' not in adapted_schema:
            adapted_schema['foreign_keys'] = []
            
        if 'primary_keys' not in adapted_schema:
            adapted_schema['primary_keys'] = []
            
        return adapted_schema

class Spider2SchemaAdapter:
    """Spider2-specific schema adapter"""
    
    @staticmethod
    def adapt(schema: Dict) -> Dict:
        """
        Adapt Spider2 schema to standardized format.
        
        Args:
            schema: The Spider2 schema
            
        Returns:
            Standardized schema dictionary
        """
        # Spider2 schemas are similar to Spider, but may have some differences
        adapted_schema = SpiderSchemaAdapter.adapt(schema)
        
        # Additional processing for Spider2 if needed
        
        return adapted_schema

class BirdSchemaAdapter:
    """Bird-specific schema adapter"""
    
    @staticmethod
    def adapt(schema: Dict) -> Dict:
        """
        Adapt Bird schema to standardized format.
        
        Args:
            schema: The Bird schema
            
        Returns:
            Standardized schema dictionary
        """
        # Bird schemas are already close to the expected format
        # Just ensure all expected fields are present
        adapted_schema = schema.copy()
        
        # Ensure all necessary fields are present
        if 'tables' not in adapted_schema:
            adapted_schema['tables'] = []
            
        if 'columns' not in adapted_schema:
            adapted_schema['columns'] = []
            
        if 'table_to_columns' not in adapted_schema:
            # Create table_to_columns mapping from columns if not present
            table_to_columns = {}
            for col in adapted_schema.get('columns', []):
                table_idx = col.get('table_idx', -1)
                if table_idx >= 0:
                    if table_idx not in table_to_columns:
                        table_to_columns[table_idx] = []
                    table_to_columns[table_idx].append(col['id'])
            
            adapted_schema['table_to_columns'] = table_to_columns
            
        if 'foreign_keys' not in adapted_schema:
            adapted_schema['foreign_keys'] = []
            
        if 'primary_keys' not in adapted_schema:
            adapted_schema['primary_keys'] = []
            
        return adapted_schema


class DataAdapter:
    """
    Abstract adapter class to standardize dataset-specific fields
    for the DataProcessor.
    """
    
    def get_db_id(self, instance: Dict) -> str:
        """Extract database ID from instance."""
        return instance.get('db_id', '')
    
    def get_question(self, instance: Dict) -> str:
        """Extract question text from instance."""
        return instance.get('question', '')
    
    def get_sql(self, instance: Dict) -> str:
        """Extract SQL query from instance."""
        pass  # To be implemented by subclasses
    
    def get_evidence(self, instance: Dict) -> str:
        """Extract evidence text from instance (if available)."""
        return ''  # Default implementation returns empty string
    
    def get_question_id(self, instance: Dict) -> str:
        """Extract question ID from instance."""
        pass  # To be implemented by subclasses
    
    def create_standardized_instance(self, instance: Dict) -> Dict:
        """Convert dataset-specific instance to standardized format."""
        pass  # To be implemented by subclasses


class BirdDataAdapter(DataAdapter):
    """Adapter for BIRD dataset."""
    
    def get_sql(self, instance: Dict) -> str:
        return instance.get('SQL', '')
    
    def get_evidence(self, instance: Dict) -> str:
        return instance.get('evidence', '')
    
    def get_question_id(self, instance: Dict) -> str:
        return instance.get('question_id', '')
    
    def create_standardized_instance(self, instance: Dict) -> Dict:
        """Convert BIRD instance to standardized format."""
        return {
            'db_id': self.get_db_id(instance),
            'question': self.get_question(instance),
            'sql': self.get_sql(instance),
            'evidence': self.get_evidence(instance),
            'question_id': self.get_question_id(instance),
            'difficulty': instance.get('difficulty', 'unknown'),
            'orig_instance': instance  # Keep original for reference
        }

class SpiderDataAdapter(DataAdapter):
    """Adapter for Spider dataset."""
    
    def get_sql(self, instance: Dict) -> str:
        return instance.get('query', '')  # Different field name in Spider
    
    def get_evidence(self, instance: Dict) -> str:
        # Spider doesn't have evidence
        return ''
    
    def get_question_id(self, instance: Dict) -> str:
        # Spider might have different ID field or format
        if 'id' in instance:
            return str(instance.get('id', ''))
        else:
            return str(instance.get('question_index', ''))
    
    def create_standardized_instance(self, instance: Dict) -> Dict:
        """Convert Spider instance to standardized format."""
        return {
            'db_id': self.get_db_id(instance),
            'question': self.get_question(instance),
            'sql': self.get_sql(instance),
            'evidence': self.get_evidence(instance),
            'question_id': self.get_question_id(instance),
            'difficulty': 'unknown',  # Spider doesn't have difficulty
            'orig_instance': instance  # Keep original for reference
        }

class Spider2DataAdapter(DataAdapter):
    """Adapter for Spider2 dataset."""

    def get_db_id(self, instance: Dict) -> str:
        return instance.get('db', '')

    def get_sql(self, instance: Dict) -> str:
        return instance.get('sql', '')  # Different field name in Spider2
    
    def get_question_id(self, instance: Dict) -> str:
        if 'instance_id' in instance:
            return str(instance.get('instance_id', ''))
        
    def get_external_knowledge(self, instance: Dict) -> str:
        # Spider2 may have external knowledge
        return instance.get('external_knowledge', '')
    
    def get_difficulty(self, instance: Dict) -> str:
        # if the the number of sql tokens are higher than 50, then it is moderate, and if the number of sql tokens are higher than 100, then it is hard
        sql = instance.get('sql', '')
        # Parse the SQL and get all non-whitespace tokens
        sql_tokens = []
        for statement in sqlparse.parse(sql):
            sql_tokens.extend([token for token in statement.flatten() if not token.is_whitespace])
        if len(sql_tokens) > 160:
            return 'challenging'
        elif len(sql_tokens) > 80:
            return 'moderate'
        else:
            return 'simple'
    
    def create_standardized_instance(self, instance: Dict) -> Dict:
        """Convert Spider2 instance to standardized format."""
        return {
            'db_id': self.get_db_id(instance),
            'question': self.get_question(instance),
            'sql': self.get_sql(instance),
            'evidence': self.get_external_knowledge(instance),
            'difficulty': self.get_difficulty(instance),
            'question_id': self.get_question_id(instance),
            'orig_instance': instance  # Keep original for reference
        }

class AdapterFactory:
    """Factory for creating dataset-specific adapters."""
    
    @staticmethod
    def get_adapter(dataset_name: str) -> DataAdapter:
        """Get adapter for specified dataset."""
        if dataset_name.lower() == 'bird':
            return BirdDataAdapter()
        elif dataset_name.lower() == 'spider':
            return SpiderDataAdapter()
        elif dataset_name.lower() == 'spider2':
            return Spider2DataAdapter()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
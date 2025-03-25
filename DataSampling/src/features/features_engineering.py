from typing import Dict, List


class FeatureExtractor:
    """
    Extract features from Text2SQL dataset for model input.
    """
    
    def __init__(self, dataset):
        """
        Initialize the feature extractor.
        
        Args:
            dataset: Dataset object
        """
        self.dataset = dataset
    
    def extract_features(self, idx: int) -> Dict:
        """
        Extract features for a specific example.
        
        Args:
            idx: Index of the example
        
        Returns:
            Dictionary of features
        """
        example = self.dataset.get_example_by_idx(idx)
        db_id = example['db_id']
        schema = self.dataset.get_schema_by_db_name(db_id)
        
        # Extract schema features
        table_names = [table['original_name'] for table in schema['tables']]
        
        # Create column lists per table
        table_columns = {}
        for table_idx, table in enumerate(schema['tables']):
            table_name = table['original_name']
            table_columns[table_name] = []
            
            # Get columns for this table
            for col_idx in schema['table_to_columns'][table_idx]:
                col = schema['columns'][col_idx]
                col_name = col['original_name']
                col_type = col['type']
                
                # Skip the id column if it's the first one
                if col_idx > 0 or col_name.lower() != 'id':
                    table_columns[table_name].append({
                        'name': col_name,
                        'type': col_type
                    })
        
        # Format question
        question = example['question']
        
        # Prepare the feature dictionary
        features = {
            'question': question,
            'db_id': db_id,
            'tables': table_names,
            'table_columns': table_columns,
            'foreign_keys': self._format_foreign_keys(schema),
            'primary_keys': self._format_primary_keys(schema),
            'target_query': example['query']
        }
        
        return features
    
    def _format_foreign_keys(self, schema: Dict) -> List[Dict]:
        """
        Format foreign keys into a readable format.
        
        Args:
            schema: Database schema
        
        Returns:
            List of foreign key relationships
        """
        foreign_keys = []
        
        for fk_pair in schema.get('foreign_keys', []):
            source_col_idx, target_col_idx = fk_pair
            
            source_col = schema['columns'][source_col_idx]
            target_col = schema['columns'][target_col_idx]
            
            source_table_idx = source_col['table_idx']
            target_table_idx = target_col['table_idx']
            
            source_table = schema['tables'][source_table_idx]['original_name']
            target_table = schema['tables'][target_table_idx]['original_name']
            
            foreign_keys.append({
                'source_table': source_table,
                'source_column': source_col['original_name'],
                'target_table': target_table,
                'target_column': target_col['original_name']
            })
        
        return foreign_keys
    
    def _format_primary_keys(self, schema: Dict) -> Dict[str, List[str]]:
        """
        Format primary keys by table.
        
        Args:
            schema: Database schema
        
        Returns:
            Dictionary mapping table names to primary key columns
        """
        primary_keys = {}
        
        for pk_idx in schema.get('primary_keys', []):
            col = schema['columns'][pk_idx]
            table_idx = col['table_idx']
            
            if table_idx >= 0:
                table_name = schema['tables'][table_idx]['original_name']
                if table_name not in primary_keys:
                    primary_keys[table_name] = []
                
                primary_keys[table_name].append(col['original_name'])
        
        return primary_keys
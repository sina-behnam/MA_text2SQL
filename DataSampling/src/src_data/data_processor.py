import os
import json
import re
import sqlparse
import spacy
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

from tqdm.auto import tqdm

# Load spaCy model for NLP analysis - same as before
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


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


class AdapterFactory:
    """Factory for creating dataset-specific adapters."""
    
    @staticmethod
    def get_adapter(dataset_name: str) -> DataAdapter:
        """Get adapter for specified dataset."""
        if dataset_name.lower() == 'bird':
            return BirdDataAdapter()
        elif dataset_name.lower() == 'spider':
            return SpiderDataAdapter()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


class DataProcessor:
    """
    Generalized processor for Text2SQL dataset instances.
    """
    
    def __init__(self, dataset, dataset_name=None):
        """
        Initialize data processor.
        
        Args:
            dataset: A dataset instance (e.g., BirdDataset)
            dataset_name: Name of the dataset (optional if can be inferred from dataset)
        """
        self.dataset = dataset
        
        # Try to infer dataset_name if not provided
        if dataset_name is None:
            if hasattr(dataset, 'dataset_name'):
                dataset_name = dataset.dataset_name
            else:
                raise ValueError("Dataset name must be provided if not available in dataset object")
        
        self.dataset_name = dataset_name
        self.adapter = AdapterFactory.get_adapter(dataset_name)
        
        # Dictionary to store schema analysis by db_id
        self.schema_analyses = {}
        
    def analyze_question(self, question_text, schema=None):
        """
        Analyze natural language question.
        
        Args:
            question_text: The question text to analyze
            schema: Optional schema to compute overlap metrics
            
        Returns:
            Dictionary with question analysis
        """
        # Process text with spaCy
        doc = nlp(question_text)
        
        # Basic statistics
        char_length = len(question_text)
        word_length = len([token for token in doc if not token.is_punct and not token.is_space])
        
        # Entity presence using NER
        entities = []
        entity_types = set()
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            entity_types.add(ent.label_)
        
        # Number presence
        numbers = []
        for token in doc:
            if token.like_num:
                numbers.append(token.text)
        
        # Negation presence
        has_negation = any(token.dep_ == 'neg' for token in doc)
        negation_words = [token.text for token in doc if token.dep_ == 'neg']
        
        # Comparatives/superlatives
        comparatives = []
        superlatives = []
        for token in doc:
            if token.tag_ == 'JJR' or token.tag_ == 'RBR':
                comparatives.append(token.text)
            elif token.tag_ == 'JJS' or token.tag_ == 'RBS':
                superlatives.append(token.text)
        
        # Prepare result
        result = {
            'char_length': char_length,
            'word_length': word_length,
            'entities': entities,
            'entity_types': list(entity_types),
            'has_entities': len(entities) > 0,
            'numbers': numbers,
            'has_numbers': len(numbers) > 0,
            'has_negation': has_negation,
            'negation_words': negation_words,
            'comparatives': comparatives,
            'has_comparatives': len(comparatives) > 0,
            'superlatives': superlatives,
            'has_superlatives': len(superlatives) > 0
        }
        
        # If schema is provided, compute overlap
        if schema:
            schema_overlap = self._compute_schema_overlap(doc, schema)
            result.update(schema_overlap)
            
        return result
    
    def _compute_schema_overlap(self, doc, schema):
        """
        Compute overlap between question and schema.
        
        Args:
            doc: spaCy document
            schema: Database schema
            
        Returns:
            Dictionary with overlap metrics
        """
        # Apply schema adapter to ensure consistent structure
        adapted_schema = SchemaAdapter.adapt_schema(schema, self.dataset_name)
        
        # Extract schema elements
        tables = adapted_schema.get('tables', [])
        columns = adapted_schema.get('columns', [])
        
        # Prepare sets of schema terms (normalize to lowercase)
        table_names = {table.get('original_name', '').lower() for table in tables}
        
        column_names = {col.get('original_name', '').lower() for col in columns}
        
        # Get descriptions if available
        column_descriptions = set()
        value_descriptions = set()
        
        for col in columns:
            if 'description' in col and col['description']:
                # Add individual words from description
                desc_doc = nlp(col['description'])
                for token in desc_doc:
                    if not token.is_stop and not token.is_punct and token.is_alpha:
                        column_descriptions.add(token.lemma_.lower())
            
            if 'value_description' in col and col['value_description']:
                # Add individual words from value description
                val_desc_doc = nlp(col['value_description'])
                for token in val_desc_doc:
                    if not token.is_stop and not token.is_punct and token.is_alpha:
                        value_descriptions.add(token.lemma_.lower())
        
        # Extract question terms (lowercased lemmas for better matching)
        question_terms = set()
        question_lemmas = set()
        
        for token in doc:
            if not token.is_stop and not token.is_punct and token.is_alpha:
                question_terms.add(token.text.lower())
                question_lemmas.add(token.lemma_.lower())
        
        # Compute direct term overlap
        table_overlap = {table: table.lower() in question_terms 
                         for table in table_names if table}
        table_overlap_lemma = {table: table.lower() in question_lemmas 
                               for table in table_names if table}
        
        column_overlap = {col: col.lower() in question_terms 
                          for col in column_names if col}
        column_overlap_lemma = {col: col.lower() in question_lemmas 
                                for col in column_names if col}
        
        # Count overlaps
        table_overlap_count = sum(1 for match in table_overlap.values() if match)
        table_overlap_lemma_count = sum(1 for match in table_overlap_lemma.values() if match)
        
        column_overlap_count = sum(1 for match in column_overlap.values() if match)
        column_overlap_lemma_count = sum(1 for match in column_overlap_lemma.values() if match)
        
        # Compute description overlap if we have descriptions
        desc_overlap_count = 0
        
        for term in question_terms.union(question_lemmas):
            if term in column_descriptions or term in value_descriptions:
                desc_overlap_count += 1
        
        # Compute semantic similarity using spaCy vectors
        semantic_similarity = {}
        for table_name in table_names:
            if not table_name:
                continue
            table_doc = nlp(table_name)
            if table_doc.vector_norm and doc.vector_norm:
                similarity = table_doc.similarity(doc)
                semantic_similarity[f"table_{table_name}"] = similarity
        
        for col_name in column_names:
            if not col_name:
                continue
            col_doc = nlp(col_name)
            if col_doc.vector_norm and doc.vector_norm:
                similarity = col_doc.similarity(doc)
                semantic_similarity[f"column_{col_name}"] = similarity
        
        # Prepare result
        table_sim_values = [similarity for name, similarity in semantic_similarity.items() 
                          if name.startswith('table_')]
        column_sim_values = [similarity for name, similarity in semantic_similarity.items() 
                           if name.startswith('column_')]
                           
        result = {
            'table_overlap_count': table_overlap_count,
            'table_overlap_lemma_count': table_overlap_lemma_count,
            'column_overlap_count': column_overlap_count,
            'column_overlap_lemma_count': column_overlap_lemma_count,
            'description_overlap_count': desc_overlap_count,
            'avg_table_similarity': sum(table_sim_values) / max(1, len(table_sim_values)) if table_sim_values else 0,
            'avg_column_similarity': sum(column_sim_values) / max(1, len(column_sim_values)) if column_sim_values else 0,
            'max_table_similarity': max(table_sim_values, default=0),
            'max_column_similarity': max(column_sim_values, default=0)
        }
        
        return result
    
    def analyze_sql(self, sql_query):
        """
        Analyze SQL query.
        
        Args:
            sql_query: The SQL query to analyze
            
        Returns:
            Dictionary with SQL analysis
        """
        # Basic statistics
        char_length = len(sql_query)
        
        # Parse SQL using sqlparse
        parsed = sqlparse.parse(sql_query)
        
        if not parsed:
            return {
                'char_length': char_length,
                'error': 'Failed to parse SQL query'
            }
        
        # Get the first statement
        stmt = parsed[0]
        
        # Extract tables from FROM and JOIN clauses
        from_pattern = r'\bFROM\s+([^\s,]+)'
        join_pattern = r'\bJOIN\s+([^\s]+)'
        
        tables_from = re.findall(from_pattern, sql_query, re.IGNORECASE)
        tables_join = re.findall(join_pattern, sql_query, re.IGNORECASE)
        
        # Clean up table names (remove aliases and quotes)
        def clean_table_name(name):
            # Remove AS alias
            name = re.sub(r'\s+AS\s+\w+', '', name, flags=re.IGNORECASE)
            # Remove quotes
            return name.strip('`"\'[]')
        
        tables_from = [clean_table_name(t) for t in tables_from]
        tables_join = [clean_table_name(t) for t in tables_join]
        
        # Combine tables from both sources
        tables = list(set(tables_from + tables_join))
        
        # Count joins
        join_count = sql_query.upper().count('JOIN')
        
        # Count WHERE conditions
        where_conditions = 0
        if 'WHERE' in sql_query.upper():
            # Extract WHERE clause
            where_clause = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|HAVING|$)', 
                                    sql_query, re.IGNORECASE | re.DOTALL)
            if where_clause:
                where_text = where_clause.group(1).strip()
                # Count AND/OR operators
                where_conditions = where_text.upper().count(' AND ') + where_text.upper().count(' OR ') + 1
        
        # Count subqueries (approximation)
        subquery_count = sql_query.upper().count('SELECT') - 1
        if subquery_count < 0:
            subquery_count = 0
        
        # Count clauses
        clauses = 0
        clause_types = []
        
        if 'GROUP BY' in sql_query.upper():
            clauses += 1
            clause_types.append('GROUP BY')
            
        if 'HAVING' in sql_query.upper():
            clauses += 1
            clause_types.append('HAVING')
            
        if 'ORDER BY' in sql_query.upper():
            clauses += 1
            clause_types.append('ORDER BY')
            
        if 'LIMIT' in sql_query.upper():
            clauses += 1
            clause_types.append('LIMIT')
        
        # Count aggregation functions
        agg_functions = ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX']
        agg_function_count = 0
        found_agg_functions = []
        
        for func in agg_functions:
            # Look for the function in SQL
            pattern = r'\b' + func + r'\s*\('
            matches = re.findall(pattern, sql_query, re.IGNORECASE)
            if matches:
                agg_function_count += len(matches)
                found_agg_functions.append(func)
        
        # Count selected columns
        selected_columns = 0
        
        # Find the SELECT clause
        select_clause = ""
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
        
        if select_match:
            select_clause = select_match.group(1).strip()
            
            # Handle SELECT *
            if select_clause == '*':
                selected_columns = 1  # Count * as one column
            else:
                # Handle functions and complex expressions
                # This is a simplification - won't work for all cases
                in_function = 0
                parts = []
                current_part = ""
                
                for char in select_clause:
                    if char == '(':
                        in_function += 1
                        current_part += char
                    elif char == ')':
                        in_function -= 1
                        current_part += char
                    elif char == ',' and in_function == 0:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part:
                    parts.append(current_part.strip())
                
                selected_columns = len(parts)
        
        # Prepare result
        result = {
            'char_length': char_length,
            'tables_count': len(tables),
            'tables': tables,
            'join_count': join_count,
            'where_conditions': where_conditions,
            'subquery_count': subquery_count,
            'clauses_count': clauses,
            'clause_types': clause_types,
            'aggregation_function_count': agg_function_count,
            'aggregation_functions': found_agg_functions,
            'select_columns': selected_columns
        }
        
        return result
    
    def analyze_schema(self, db_name):
        """
        Analyze database schema.
        
        Args:
            db_name: Name of the database to analyze
            
        Returns:
            Dictionary with schema analysis
        """
        try:
            # Get schema information
            schema = self.dataset.get_schema_by_db_name(db_name)
            
            # Apply schema adapter to ensure consistent structure
            adapted_schema = SchemaAdapter.adapt_schema(schema, self.dataset_name)
            
            # Basic statistics
            tables = adapted_schema.get('tables', [])
            columns = adapted_schema.get('columns', [])
            table_to_columns = adapted_schema.get('table_to_columns', {})
            foreign_keys = adapted_schema.get('foreign_keys', [])
            primary_keys = adapted_schema.get('primary_keys', [])
            
            # Count tables
            table_count = len(tables)
            
            # Process each table
            table_analysis = []
            
            for table in tables:
                table_id = table.get('id')
                table_name = table.get('original_name', '')
                
                # Get columns for this table
                table_columns = []
                column_datatypes = {}
                
                for col_idx in table_to_columns.get(table_id, []):
                    if col_idx >= len(columns):
                        continue
                        
                    col = columns[col_idx]
                    col_name = col.get('original_name', '')
                    col_type = col.get('type', 'unknown').upper()
                    
                    is_pk = col_idx in primary_keys
                    
                    # Find foreign keys for this column
                    fk_references = []
                    for fk in foreign_keys:
                        if len(fk) >= 2 and fk[0] == col_idx:
                            # This column references another
                            if fk[1] < len(columns):
                                referenced_col = columns[fk[1]]
                                referenced_table_idx = referenced_col.get('table_idx')
                                
                                if referenced_table_idx is not None and referenced_table_idx < len(tables):
                                    referenced_table = tables[referenced_table_idx].get('original_name', '')
                                    fk_references.append({
                                        'table': referenced_table,
                                        'column': referenced_col.get('original_name', '')
                                    })
                    
                    # Add to column list
                    table_columns.append({
                        'name': col_name,
                        'type': col_type,
                        'is_primary_key': is_pk,
                        'foreign_key_references': fk_references
                    })
                    
                    # Count datatype
                    if col_type in column_datatypes:
                        column_datatypes[col_type] += 1
                    else:
                        column_datatypes[col_type] = 1
                
                # Create table analysis
                table_analysis.append({
                    'name': table_name,
                    'columns_count': len(table_columns),
                    'columns': table_columns,
                    'datatype_counts': column_datatypes,
                    'primary_key_count': sum(1 for col in table_columns if col['is_primary_key']),
                    'foreign_key_count': sum(len(col['foreign_key_references']) for col in table_columns)
                })
            
            # Prepare result
            result = {
                'db_name': db_name,
                'table_count': table_count,
                'table_names': [table.get('original_name', '') for table in tables],
                'column_count': len(columns),
                'primary_key_count': len(primary_keys),
                'foreign_key_count': len(foreign_keys),
                'tables': table_analysis
            }
            
            return result
            
        except Exception as e:
            return {
                'db_name': db_name,
                'error': str(e)
            }
    
    def process_dataset_instance(self, instance):
        """
        Process a single dataset instance.
        
        Args:
            instance: The dataset instance to process
            
        Returns:
            Dictionary with complete analysis (excluding schema analysis)
        """
        # Standardize the instance using the appropriate adapter
        std_instance = self.adapter.create_standardized_instance(instance)
        
        # Extract standardized fields
        db_id = std_instance['db_id']
        question = std_instance['question']
        sql = std_instance['sql']
        evidence = std_instance['evidence']
        question_id = std_instance['question_id']
        difficulty = std_instance['difficulty']
        
        # Load schema
        schema = self.dataset.get_schema_by_db_name(db_id)
        
        # Check if we already analyzed this schema
        if db_id not in self.schema_analyses:
            self.schema_analyses[db_id] = self.analyze_schema(db_id)
        
        # Analyze components (excluding schema analysis)
        question_analysis = self.analyze_question(question, schema)
        sql_analysis = self.analyze_sql(sql)
        
        # Combine into a complete analysis
        analysis = {
            'question_id': question_id,
            'db_id': db_id,
            'question': question,
            'difficulty': difficulty,
            'sql': sql,
            'evidence': evidence,
            'question_analysis': question_analysis,
            'sql_analysis': sql_analysis,
            'dataset': self.dataset_name
        }
        
        return analysis
    
    def batch_process(self, limit=None):
        """
        Process multiple dataset instances.
        
        Args:
            limit: Maximum number of instances to process
            
        Returns:
            List of processed instances and dictionary of schema analyses
        """
        results = []
        
        # Get all instances
        instances = self.dataset.load_data()
        
        # Apply limit if specified
        if limit is not None:
            instances = instances[:limit]
        
        # Process each instance
        for instance in tqdm(instances, desc="Processing instances", unit="instance"):
            result = self.process_dataset_instance(instance)
            results.append(result)
        
        return results
    
    def save_processed_data(self, output_path, results):
        """
        Save processed data to a file.
        
        Args:
            output_path: Path to save the results
            results: The results to save
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create output structure with instances and schemas
        output_data = {
            'instances': results,
            'schemas': self.schema_analyses
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved processed data to {output_path}")
        print(f"- {len(results)} instances")
        print(f"- {len(self.schema_analyses)} database schemas")


# Example usage for testing
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process Text2SQL datasets for analysis')
    parser.add_argument('--dataset', type=str, choices=['bird', 'spider'], required=True,
                        help='Dataset type to process (bird or spider)')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing the dataset')
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'],
                        help='Dataset split to process (train/dev/test)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for processed data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of instances to process (optional)')
    
    args = parser.parse_args()
    
    # Load the dataset based on command-line arguments
    dataset = DataLoader.get_dataset(args.dataset, base_dir=args.base_dir, split=args.split)
    
    # Create processor
    processor = DataProcessor(dataset, args.dataset)
    
    # Process instances
    print(f"Processing {args.dataset.upper()} dataset ({args.split} split)...")
    results = processor.batch_process(limit=args.limit)
    
    # Save results
    processor.save_processed_data(args.output, results)
    print(f"Processing complete! Saved {len(results)} instances to {args.output}")
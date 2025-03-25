import os
import json
import pandas as pd
from collections import defaultdict
from ..features.features_engineering import FeatureExtractor


class DataProcessor:
    """
    Data processor for Text2SQL datasets.
    Converts raw data into processed formats and generates statistics.
    """
    
    def __init__(self, dataset, output_dir: str = None):
        """
        Initialize the data processor.
        
        Args:
            dataset: Dataset object
            output_dir: Directory to save processed data
        """
        self.dataset = dataset
        self.output_dir = output_dir or os.path.join(
            'data', 'processed', dataset.dataset_name, dataset.split
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'statistics'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
        
    def process_dataset(self):
        """
        Process the entire dataset and generate statistics.
        """
        # Ensure data is loaded
        data = self.dataset.load_data()
        schemas = self.dataset.load_schemas()
        
        # Generate statistics
        self._generate_dataset_statistics()
        self._generate_schema_statistics()
        self._generate_question_statistics()
        self._generate_sql_statistics()
        
        # Generate sample files
        self._generate_sample_files()
        
        print(f"Dataset processing complete. Results saved to {self.output_dir}")
    
    def _generate_dataset_statistics(self):
        """
        Generate general statistics about the dataset.
        """
        data = self.dataset.data
        
        # Count examples per database
        db_counts = defaultdict(int)
        for example in data:
            db_counts[example['db_id']] += 1
        
        # Convert to DataFrame and sort
        db_stats = pd.DataFrame({
            'database': list(db_counts.keys()),
            'count': list(db_counts.values())
        }).sort_values('count', ascending=False)
        
        # Save statistics
        db_stats.to_csv(
            os.path.join(self.output_dir, 'statistics', 'database_distribution.csv'),
            index=False
        )
        
        # Generate overall statistics
        stats = {
            'total_examples': len(data),
            'total_databases': len(db_counts),
            'avg_examples_per_db': len(data) / len(db_counts) if db_counts else 0,
            'max_examples_per_db': max(db_counts.values()) if db_counts else 0,
            'min_examples_per_db': min(db_counts.values()) if db_counts else 0
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'statistics', 'general_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _generate_schema_statistics(self):
        """
        Generate statistics about database schemas.
        """
        schemas = self.dataset.db_schemas
        
        # Calculate statistics for each schema
        schema_stats = []
        for db_id, schema in schemas.items():
            num_tables = len(schema['tables'])
            num_columns = len(schema['columns'])
            num_foreign_keys = len(schema.get('foreign_keys', []))
            num_primary_keys = len(schema.get('primary_keys', []))
            
            schema_stats.append({
                'database': db_id,
                'tables': num_tables,
                'columns': num_columns,
                'foreign_keys': num_foreign_keys,
                'primary_keys': num_primary_keys,
                'avg_columns_per_table': num_columns / num_tables if num_tables else 0
            })
        
        # Convert to DataFrame
        schema_df = pd.DataFrame(schema_stats)
        
        # Save statistics
        schema_df.to_csv(
            os.path.join(self.output_dir, 'statistics', 'schema_statistics.csv'),
            index=False
        )
        
        # Calculate aggregate statistics
        agg_stats = {
            'avg_tables_per_db': schema_df['tables'].mean(),
            'avg_columns_per_db': schema_df['columns'].mean(),
            'avg_foreign_keys_per_db': schema_df['foreign_keys'].mean(),
            'avg_primary_keys_per_db': schema_df['primary_keys'].mean(),
            'max_tables_in_db': schema_df['tables'].max(),
            'max_columns_in_db': schema_df['columns'].max()
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'statistics', 'schema_agg_stats.json'), 'w') as f:
            json.dump(agg_stats, f, indent=2)
    
    def _generate_question_statistics(self):
        """
        Generate statistics about questions in the dataset.
        """
        data = self.dataset.data
        
        # Analyze questions
        question_lengths = []
        question_tokens = []
        
        for example in data:
            question = example['question']
            question_lengths.append(len(question))
            question_tokens.append(len(question.split()))
        
        # Create histogram data
        length_bins = list(range(0, max(question_lengths) + 10, 10))
        token_bins = list(range(0, max(question_tokens) + 5, 5))
        
        length_hist = pd.cut(question_lengths, bins=length_bins).value_counts().sort_index()
        token_hist = pd.cut(question_tokens, bins=token_bins).value_counts().sort_index()
        
        # Save histograms
        length_hist.to_csv(
            os.path.join(self.output_dir, 'statistics', 'question_length_hist.csv')
        )
        token_hist.to_csv(
            os.path.join(self.output_dir, 'statistics', 'question_token_hist.csv')
        )
        
        # Calculate statistics
        stats = {
            'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'avg_question_tokens': sum(question_tokens) / len(question_tokens) if question_tokens else 0,
            'min_question_length': min(question_lengths) if question_lengths else 0,
            'max_question_length': max(question_lengths) if question_lengths else 0,
            'min_question_tokens': min(question_tokens) if question_tokens else 0,
            'max_question_tokens': max(question_tokens) if question_tokens else 0
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'statistics', 'question_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _generate_sql_statistics(self):
        """
        Generate statistics about SQL queries in the dataset.
        """
        data = self.dataset.data
        
        # Analyze SQL queries
        sql_lengths = []
        sql_tokens = []
        keywords = defaultdict(int)
        
        # Common SQL keywords to track
        common_keywords = ['select', 'from', 'where', 'group', 'order', 'having', 
                          'join', 'left', 'right', 'inner', 'outer', 'limit',
                          'union', 'intersect', 'except', 'distinct', 'count', 
                          'sum', 'avg', 'min', 'max']
        
        for example in data:
            sql = example['query']
            sql_lengths.append(len(sql))
            tokens = sql.lower().split()
            sql_tokens.append(len(tokens))
            
            # Count keywords
            for keyword in common_keywords:
                if keyword in tokens:
                    keywords[keyword] += 1
        
        # Create keyword statistics
        keyword_stats = pd.DataFrame({
            'keyword': list(keywords.keys()),
            'count': list(keywords.values()),
            'percentage': [count / len(data) * 100 for count in keywords.values()]
        }).sort_values('count', ascending=False)
        
        # Save keyword statistics
        keyword_stats.to_csv(
            os.path.join(self.output_dir, 'statistics', 'sql_keywords.csv'),
            index=False
        )
        
        # Create histogram data
        length_bins = list(range(0, max(sql_lengths) + 20, 20))
        token_bins = list(range(0, max(sql_tokens) + 10, 10))
        
        length_hist = pd.cut(sql_lengths, bins=length_bins).value_counts().sort_index()
        token_hist = pd.cut(sql_tokens, bins=token_bins).value_counts().sort_index()
        
        # Save histograms
        length_hist.to_csv(
            os.path.join(self.output_dir, 'statistics', 'sql_length_hist.csv')
        )
        token_hist.to_csv(
            os.path.join(self.output_dir, 'statistics', 'sql_token_hist.csv')
        )
        
        # Calculate statistics
        stats = {
            'avg_sql_length': sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0,
            'avg_sql_tokens': sum(sql_tokens) / len(sql_tokens) if sql_tokens else 0,
            'min_sql_length': min(sql_lengths) if sql_lengths else 0,
            'max_sql_length': max(sql_lengths) if sql_lengths else 0,
            'min_sql_tokens': min(sql_tokens) if sql_tokens else 0,
            'max_sql_tokens': max(sql_tokens) if sql_tokens else 0
        }
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'statistics', 'sql_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _generate_sample_files(self):
        """
        Generate sample files for manual inspection.
        """
        data = self.dataset.data
        
        # Sample size (adjust as needed)
        sample_size = min(10, len(data))
        
        # Generate detailed samples
        for i in range(sample_size):
            example = data[i]
            db_id = example['db_id']
            
            # Get schema information
            schema_text = self.dataset.get_table_schemas_text(db_id)
            
            # Create a detailed sample
            sample = {
                'example_id': i,
                'db_id': db_id,
                'question': example['question'],
                'query': example['query'],
                'schema': schema_text
            }
            
            # Save as JSON
            with open(os.path.join(self.output_dir, 'samples', f'sample_{i}.json'), 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Create a summary file with all samples
        summary = []
        for i in range(min(50, len(data))):
            example = data[i]
            summary.append({
                'example_id': i,
                'db_id': example['db_id'],
                'question': example['question'],
                'query': example['query']
            })
        
        # Save summary
        with open(os.path.join(self.output_dir, 'samples', 'samples_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


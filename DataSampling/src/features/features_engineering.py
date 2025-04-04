import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import spacy
import os
import json
import gzip
import pickle
from datetime import datetime
from pathlib import Path

class FeatureEngineering:
    """
    Feature engineering for Text2SQL tasks.
    Works with processed data from any dataset implementation.
    Optimized for large datasets with chunking and compression options.
    """
    
    def __init__(self, processed_data=None, chunk_size=1000, use_compression=True):
        """
        Initialize feature engineering.
        
        Args:
            processed_data: Optional pre-processed data
            chunk_size: Size of chunks when processing large datasets
            use_compression: Whether to use compression for output files
        """
        self.processed_data = processed_data
        self.features = {}
        self.datasets_present = set()
        self.chunk_size = chunk_size
        self.use_compression = use_compression
        
        # Load spaCy model for NLP features
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
        
        # If data is provided, identify which datasets are present
        if self.processed_data:
            self._identify_datasets()
    
    def _identify_datasets(self):
        """
        Identify which datasets are present in the processed data.
        """
        self.datasets_present = set()
        for item in self.processed_data:
            if 'dataset' in item and 'error' not in item:
                self.datasets_present.add(item['dataset'].lower())
        
        print(f"Identified datasets: {', '.join(self.datasets_present)}")
    
    def load_processed_data(self, file_paths, limit=None):
        """
        Load processed data from files. Can handle single file or multiple files.
        
        Args:
            file_paths: Path to processed data file(s). Can be:
                        - A single file path as string
                        - A list of file paths
                        - A directory path containing multiple JSON files
            limit: Optional limit on the number of instances to load per file
        """
        self.processed_data = []
        
        # Convert single path to list
        if isinstance(file_paths, str):
            # Check if it's a directory
            if os.path.isdir(file_paths):
                # Find all JSON files in directory
                file_list = [os.path.join(file_paths, f) for f in os.listdir(file_paths) 
                           if f.endswith('.json') and os.path.isfile(os.path.join(file_paths, f))]
            else:
                # Single file
                file_list = [file_paths]
        else:
            # Already a list of paths
            file_list = file_paths
            
        # Process each file
        for file_path in file_list:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            print(f"Loading data from {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if limit:
                        # Parse file line by line for large files with a limit
                        data = []
                        for i, line in enumerate(f):
                            if i == 0 and line.startswith('['):
                                # Handle array format - read first [
                                continue
                            if i > limit:
                                break
                            
                            # Handle commas at the end of objects in array
                            if line.rstrip().endswith(','):
                                line = line[:-1]
                            
                            try:
                                if line.strip() and not line.strip().startswith(']'):
                                    item = json.loads(line)
                                    data.append(item)
                            except json.JSONDecodeError:
                                continue
                        
                        # Add to main data list
                        self.processed_data.extend(data)
                        print(f"  Loaded {len(data)} items from {file_path}")
                    else:
                        # Load the entire file
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            self.processed_data.extend(file_data)
                            print(f"  Loaded {len(file_data)} items from {file_path}")
                        else:
                            # Single item file
                            self.processed_data.append(file_data)
                            print(f"  Loaded 1 item from {file_path}")
            except json.JSONDecodeError as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error loading {file_path}: {str(e)}")
                continue
            
        # Identify which datasets are present
        self._identify_datasets()
        print(f"Total loaded data: {len(self.processed_data)} items from {len(self.datasets_present)} datasets: {', '.join(self.datasets_present)}")
    
    def extract_question_features(self):
        """
        Extract features from question analysis.
        
        Returns:
            DataFrame with question features
        """
        if not self.processed_data:
            raise ValueError("No processed data available")
        
        features = []
        
        for item in self.processed_data:
            if 'error' in item:
                continue
                
            q_analysis = item.get('question_analysis', {})
            
            # Extract basic features
            feature = {
                'question_id': item.get('question_id'),
                'db_id': item.get('db_id'),
                'dataset': item.get('dataset', 'unknown'),
                'difficulty': item.get('difficulty', 'unknown'),  # Added difficulty feature
                'q_char_length': q_analysis.get('char_length', 0),
                'q_word_length': q_analysis.get('word_length', 0),
                'q_has_entities': q_analysis.get('has_entities', False),
                'q_entity_count': len(q_analysis.get('entities', [])),
                'q_has_numbers': q_analysis.get('has_numbers', False),
                'q_number_count': len(q_analysis.get('numbers', [])),
                'q_has_negation': q_analysis.get('has_negation', False),
                'q_negation_count': len(q_analysis.get('negation_words', [])),
                'q_has_comparatives': q_analysis.get('has_comparatives', False),
                'q_comparative_count': len(q_analysis.get('comparatives', [])),
                'q_has_superlatives': q_analysis.get('has_superlatives', False),
                'q_superlative_count': len(q_analysis.get('superlatives', [])),
            }
            
            # Add entity type features
            entity_types = q_analysis.get('entity_types', [])
            for ent_type in ['DATE', 'TIME', 'PERSON', 'ORG', 'GPE', 'LOC', 'MONEY', 'PERCENT', 'CARDINAL']:
                feature[f'q_has_{ent_type.lower()}'] = ent_type in entity_types
            
            # Add schema overlap features
            if 'table_overlap_count' in q_analysis:
                feature.update({
                    'q_table_overlap_count': q_analysis.get('table_overlap_count', 0),
                    'q_table_overlap_lemma_count': q_analysis.get('table_overlap_lemma_count', 0),
                    'q_column_overlap_count': q_analysis.get('column_overlap_count', 0),
                    'q_column_overlap_lemma_count': q_analysis.get('column_overlap_lemma_count', 0),
                    'q_description_overlap_count': q_analysis.get('description_overlap_count', 0),
                    'q_avg_table_similarity': q_analysis.get('avg_table_similarity', 0),
                    'q_avg_column_similarity': q_analysis.get('avg_column_similarity', 0),
                    'q_max_table_similarity': q_analysis.get('max_table_similarity', 0),
                    'q_max_column_similarity': q_analysis.get('max_column_similarity', 0)
                })
            
            # Add any evidence-related features if the dataset supports it
            if 'evidence' in item and item['evidence']:
                feature['q_has_evidence'] = True
                feature['q_evidence_length'] = len(item['evidence'])
            else:
                feature['q_has_evidence'] = False
                feature['q_evidence_length'] = 0
            
            features.append(feature)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Optimize memory usage by downcasting numeric types
        df = self._optimize_dataframe(df)
        
        # Cache features
        self.features['question'] = df
        
        return df
    
    def extract_sql_features(self):
        """
        Extract features from SQL analysis.
        
        Returns:
            DataFrame with SQL features
        """
        if not self.processed_data:
            raise ValueError("No processed data available")
        
        features = []
        
        for item in self.processed_data:
            if 'error' in item:
                continue
                
            sql_analysis = item.get('sql_analysis', {})
            
            # Extract features
            feature = {
                'question_id': item.get('question_id'),
                'db_id': item.get('db_id'),
                'dataset': item.get('dataset', 'unknown'),
                'difficulty': item.get('difficulty', 'unknown'),  # Added difficulty feature
                'sql_char_length': sql_analysis.get('char_length', 0),
                'sql_tables_count': sql_analysis.get('tables_count', 0),
                'sql_join_count': sql_analysis.get('join_count', 0),
                'sql_where_conditions': sql_analysis.get('where_conditions', 0),
                'sql_subquery_count': sql_analysis.get('subquery_count', 0),
                'sql_clauses_count': sql_analysis.get('clauses_count', 0),
                'sql_has_group_by': 'GROUP BY' in sql_analysis.get('clause_types', []),
                'sql_has_having': 'HAVING' in sql_analysis.get('clause_types', []),
                'sql_has_order_by': 'ORDER BY' in sql_analysis.get('clause_types', []),
                'sql_has_limit': 'LIMIT' in sql_analysis.get('clause_types', []),
                'sql_agg_function_count': sql_analysis.get('aggregation_function_count', 0),
                'sql_select_columns': sql_analysis.get('select_columns', 0)
            }
            
            # Add individual aggregation function features
            for func in ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX']:
                feature[f'sql_has_{func.lower()}'] = func in sql_analysis.get('aggregation_functions', [])
            
            features.append(feature)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Optimize memory usage
        df = self._optimize_dataframe(df)
        
        # Cache features
        self.features['sql'] = df
        
        return df
    
    def extract_schema_features(self):
        """
        Extract features from schema analysis.
        
        Returns:
            DataFrame with schema features
        """
        if not self.processed_data:
            raise ValueError("No processed data available")
        
        features = []
        
        # Extract unique db_ids
        db_ids = set()
        for item in self.processed_data:
            if 'error' not in item and 'db_id' in item:
                db_ids.add((item.get('db_id'), item.get('dataset', 'unknown')))
        
        for db_id, dataset in db_ids:
            # Find the first instance with this db_id
            instance = next((item for item in self.processed_data 
                            if item.get('db_id') == db_id and 
                            item.get('dataset') == dataset and 
                            'error' not in item), None)
            
            if not instance:
                continue
                
            schema_analysis = instance.get('schema_analysis', {})
            
            # Extract basic features
            feature = {
                'db_id': db_id,
                'dataset': dataset,
                'schema_table_count': schema_analysis.get('table_count', 0),
                'schema_column_count': schema_analysis.get('column_count', 0),
                'schema_primary_key_count': schema_analysis.get('primary_key_count', 0),
                'schema_foreign_key_count': schema_analysis.get('foreign_key_count', 0)
            }
            
            # # Add table-specific features
            # tables = schema_analysis.get('tables', [])
            # for idx, table in enumerate(tables[:10]):  # Limit to first 10 tables for large schemas
            #     prefix = f'table_{idx+1}'
            #     feature[f'{prefix}_name'] = table.get('name', '')
            #     feature[f'{prefix}_columns_count'] = table.get('columns_count', 0)
            #     feature[f'{prefix}_pk_count'] = table.get('primary_key_count', 0)
            #     feature[f'{prefix}_fk_count'] = table.get('foreign_key_count', 0)
                
            #     # Add datatype counts
            #     datatype_counts = table.get('datatype_counts', {})
            #     for dtype in ['TEXT', 'INTEGER', 'REAL', 'NUMERIC', 'BLOB', 'DATE', 'DATETIME', 'BOOLEAN', 'NUMBER', 'OTHERS']:
            #         feature[f'{prefix}_{dtype.lower()}_count'] = datatype_counts.get(dtype, 0)
            
            features.append(feature)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Optimize memory usage
        df = self._optimize_dataframe(df)
        
        # Cache features
        self.features['schema'] = df
        
        return df
    
    def combine_features(self, optimize_memory=True):
        """
        Combine all features into a single DataFrame.
        
        Args:
            optimize_memory: Whether to optimize memory usage
            
        Returns:
            DataFrame with all features
        """
        # Make sure all feature types are extracted
        if 'question' not in self.features:
            self.extract_question_features()
        
        if 'sql' not in self.features:
            self.extract_sql_features()
        
        if 'schema' not in self.features:
            self.extract_schema_features()
        
        # Combine features
        question_df = self.features['question']
        sql_df = self.features['sql']
        schema_df = self.features['schema']
        
        # Merge question and SQL features (one-to-one)
        combined_df = pd.merge(question_df, sql_df, on=['question_id', 'db_id', 'dataset', 'difficulty'], how='left')
        
        # Merge with schema features (many-to-one)
        combined_df = pd.merge(combined_df, schema_df, on=['db_id', 'dataset'], how='left')
        
        # Optimize memory usage if requested
        if optimize_memory:
            combined_df = self._optimize_dataframe(combined_df)
        
        return combined_df
    

    
    def extract_dataset_specific_features(self, dataset_name):
        """
        Extract features that are specific to a particular dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with dataset-specific features
        """
        if not self.processed_data:
            raise ValueError("No processed data available")
        
        features = []
        
        # Filter processed data for this dataset
        dataset_data = [item for item in self.processed_data 
                        if item.get('dataset', '').lower() == dataset_name.lower() 
                        and 'error' not in item]
        
        if not dataset_data:
            return pd.DataFrame()  # Empty DataFrame if no data for this dataset
        
        for item in dataset_data:
            # Initialize with common identifiers
            feature = {
                'question_id': item.get('question_id'),
                'db_id': item.get('db_id'),
                'dataset': dataset_name,
                'difficulty': item.get('difficulty', 'unknown'),  # Add difficulty
            }
            
            # Add dataset-specific features
            if dataset_name.lower() == 'bird':
                # BIRD-specific features (e.g., evidence-related)
                feature.update({
                    'has_evidence': bool(item.get('evidence', '')),
                    'evidence_length': len(item.get('evidence', '')),
                    # Add more BIRD-specific features as needed
                })
            
            elif dataset_name.lower() == 'spider':
                # Spider-specific features
                # For example, difficulty level if available
                if 'orig_instance' in item and 'difficulty' in item['orig_instance']:
                    feature['orig_difficulty'] = item['orig_instance']['difficulty']
                # Add more Spider-specific features as needed
            
            # Add other dataset-specific features as needed
            
            features.append(feature)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Optimize memory usage
        df = self._optimize_dataframe(df)
        
        # Cache features with dataset prefix
        self.features[f'{dataset_name.lower()}_specific'] = df
        
        return df
    
    def analyze_difficulty_distribution(self):
        """
        Analyze distribution of question difficulties across datasets.
        Note: Spider dataset doesn't have difficulty labels, while BIRD does.
        
        Returns:
            DataFrame with difficulty distribution
        """
        if 'question' not in self.features:
            self.extract_question_features()
            
        question_df = self.features['question']
        
        # Get difficulty distribution
        difficulty_df = question_df.groupby(['dataset', 'difficulty']).size().reset_index(name='count')
        
        # Calculate percentages within each dataset
        difficulty_df['percentage'] = difficulty_df.groupby('dataset')['count'].transform(
            lambda x: 100 * x / x.sum()
        ).round(2)
        
        return difficulty_df
    
    def _optimize_dataframe(self, df):
        """
        Optimize memory usage of DataFrame by downcasting numeric types.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Downcast numeric columns
        for col in result.select_dtypes(include=['int']).columns:
            result[col] = pd.to_numeric(result[col], downcast='integer')
            
        for col in result.select_dtypes(include=['float']).columns:
            result[col] = pd.to_numeric(result[col], downcast='float')
        
        # Convert object columns with few unique values to category
        for col in result.select_dtypes(include=['object']).columns:
            if result[col].nunique() < 0.5 * len(result):
                result[col] = result[col].astype('category')
        
        return result
    
    def _get_output_path(self, base_path, feature_type, chunk_idx=None, fmt='csv', compressed=False):
        """
        Get output path for a feature file.
        
        Args:
            base_path: Base path for output file
            feature_type: Type of features
            chunk_idx: Optional chunk index for large files
            fmt: Format of the file ('csv' or 'pickle')
            compressed: Whether to use compression
            
        Returns:
            Output path
        """
        # Make path object
        path = Path(base_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove extension if present
        stem = path.stem
        
        # Add feature type
        if feature_type != 'all':
            stem = f"{stem}_{feature_type}"
        
        # Add chunk index if specified
        if chunk_idx is not None:
            stem = f"{stem}_chunk{chunk_idx}"
        
        # Add extension
        if fmt == 'pickle':
            stem = f"{stem}.pkl"
        else:  # default to csv
            stem = f"{stem}.csv"
        
        # Add compression extension if requested
        if compressed:
            stem = f"{stem}.gz"
        
        # Combine with parent directory
        return path.parent / stem
    
    def save_features(self, output_path, feature_type='all', max_rows_per_file=100000, fmt='csv'):
        """
        Save features to a file.
        
        Args:
            output_path: Path to save the features
            feature_type: Type of features to save ('all', 'question', 'sql', 'schema', 'combined', 
                                                  or a specific dataset name for dataset-specific features)
            max_rows_per_file: Maximum rows per output file (for chunking large files)
            fmt: Format to save ('csv' or 'pickle')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Track all saved files
        saved_files = []
        
        if feature_type == 'all':
            # Save all feature types
            for ft in ['question', 'sql', 'schema', 'combined']:
                if ft == 'combined':
                    df = self.combine_features()
                elif ft in self.features:
                    df = self.features[ft]
                else:
                    if ft == 'question':
                        df = self.extract_question_features()
                    elif ft == 'sql':
                        df = self.extract_sql_features()
                    elif ft == 'schema':
                        df = self.extract_schema_features()
                
                # Save with chunking if needed
                self._save_dataframe(df, output_path, ft, max_rows_per_file, fmt)
                
                # Add to saved files list
                ft_path = self._get_output_path(output_path, ft, fmt=fmt, compressed=self.use_compression)
                saved_files.append(str(ft_path))
                
                print(f"Saved {ft} features with {len(df)} rows")
                
            # Save dataset-specific features for each dataset
            for dataset in self.datasets_present:
                df = self.extract_dataset_specific_features(dataset)
                if not df.empty:
                    ds_type = f"{dataset}_specific"
                    
                    # Save with chunking if needed
                    self._save_dataframe(df, output_path, ds_type, max_rows_per_file, fmt)
                    
                    # Add to saved files list
                    ds_path = self._get_output_path(output_path, ds_type, fmt=fmt, compressed=self.use_compression)
                    saved_files.append(str(ds_path))
                    
                    print(f"Saved {dataset}-specific features with {len(df)} rows")
                    
            # Only save difficulty distribution for datasets that have it (like BIRD)
            bird_data = [item for item in self.processed_data 
                       if item.get('dataset', '').lower() == 'bird' 
                       and 'error' not in item]
            
            if bird_data:
                difficulty_df = self.analyze_difficulty_distribution()
                diff_type = "difficulty_distribution"
                
                # Save difficulty distribution (should be small)
                diff_path = self._get_output_path(output_path, diff_type, fmt=fmt, compressed=self.use_compression)
                if fmt == 'pickle':
                    if self.use_compression:
                        with gzip.open(diff_path, 'wb') as f:
                            pickle.dump(difficulty_df, f)
                    else:
                        with open(diff_path, 'wb') as f:
                            pickle.dump(difficulty_df, f)
                else:
                    if self.use_compression:
                        difficulty_df.to_csv(diff_path, index=False, compression='gzip')
                    else:
                        difficulty_df.to_csv(diff_path, index=False)
                
                saved_files.append(str(diff_path))
                print(f"Saved difficulty distribution with {len(difficulty_df)} rows")
            
            # Save a manifest file with metadata about all saved files
            manifest = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_types': list(set([p.split('_')[-1].split('.')[0] for p in saved_files if not p.endswith('_manifest.json')])),
                'datasets': list(self.datasets_present),
                'total_instances': len(self.processed_data) if self.processed_data else 0,
                'files': saved_files,
                'compression': self.use_compression,
                'format': fmt
            }
            
            manifest_path = os.path.join(os.path.dirname(output_path), 'feature_extraction_manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"Saved feature extraction manifest to {manifest_path}")
            
        else:
            # Save specific feature type
            if feature_type == 'combined':
                df = self.combine_features()
            elif '_specific' in feature_type:
                # Extract dataset name from feature_type (e.g., 'bird_specific' -> 'bird')
                dataset_name = feature_type.split('_')[0]
                df = self.extract_dataset_specific_features(dataset_name)
            elif feature_type in self.features:
                df = self.features[feature_type]
            else:
                if feature_type == 'question':
                    df = self.extract_question_features()
                elif feature_type == 'sql':
                    df = self.extract_sql_features()
                elif feature_type == 'schema':
                    df = self.extract_schema_features()
                elif feature_type == 'difficulty_distribution':
                    df = self.analyze_difficulty_distribution()
                else:
                    # Check if it's a dataset name
                    if feature_type.lower() in self.datasets_present:
                        df = self.extract_dataset_specific_features(feature_type)
                    else:
                        raise ValueError(f"Unknown feature type: {feature_type}")
            
            # Save with chunking if needed
            self._save_dataframe(df, output_path, feature_type, max_rows_per_file, fmt)
            print(f"Saved {feature_type} features with {len(df)} rows")
    
    def _save_dataframe(self, df, base_path, feature_type, max_rows_per_file, fmt='csv'):
        """
        Save DataFrame with chunking for large files.
        
        Args:
            df: DataFrame to save
            base_path: Base path for the output file
            feature_type: Type of features
            max_rows_per_file: Maximum rows per file
            fmt: Format to save ('csv' or 'pickle')
        """
        # Check if chunking is needed
        if len(df) > max_rows_per_file:
            # Calculate number of chunks
            num_chunks = (len(df) + max_rows_per_file - 1) // max_rows_per_file
            
            # Save in chunks
            for i in range(num_chunks):
                start_idx = i * max_rows_per_file
                end_idx = min((i + 1) * max_rows_per_file, len(df))
                
                chunk_df = df.iloc[start_idx:end_idx]
                chunk_path = self._get_output_path(base_path, feature_type, i, fmt, self.use_compression)
                
                if fmt == 'pickle':
                    if self.use_compression:
                        with gzip.open(chunk_path, 'wb') as f:
                            pickle.dump(chunk_df, f)
                    else:
                        with open(chunk_path, 'wb') as f:
                            pickle.dump(chunk_df, f)
                else:  # csv format
                    if self.use_compression:
                        chunk_df.to_csv(chunk_path, index=False, compression='gzip')
                    else:
                        chunk_df.to_csv(chunk_path, index=False)
                
                print(f"  Saved chunk {i+1}/{num_chunks} with {len(chunk_df)} rows to {chunk_path}")
        else:
            # Save as a single file
            output_path = self._get_output_path(base_path, feature_type, fmt=fmt, compressed=self.use_compression)
            
            if fmt == 'pickle':
                if self.use_compression:
                    with gzip.open(output_path, 'wb') as f:
                        pickle.dump(df, f)
                else:
                    with open(output_path, 'wb') as f:
                        pickle.dump(df, f)
            else:  # csv format
                if self.use_compression:
                    df.to_csv(output_path, index=False, compression='gzip')
                else:
                    df.to_csv(output_path, index=False)


# Example usage
if __name__ == "__main__":
    # Example 1: Load from separate dataset JSON files
    bird_data_path = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/outputs/test_multi/processed_bird_data_test.json"
    spider_data_path = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/outputs/test_multi/processed_spider_data_test.json"
    
    # Create feature engineering object with compression enabled
    feature_eng = FeatureEngineering(chunk_size=500, use_compression=True)
    
    # Load processed data from multiple files
    feature_eng.load_processed_data([bird_data_path, spider_data_path])
    
    # Extract features
    question_features = feature_eng.extract_question_features()
    sql_features = feature_eng.extract_sql_features()
    schema_features = feature_eng.extract_schema_features()
    
    print(f"Extracted {len(question_features)} question features")
    print(f"Extracted {len(sql_features)} SQL features")
    print(f"Extracted {len(schema_features)} schema features")
    
    # Save all features with chunking and compression
    output_dir = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/outputs/test_multi/features_test"
    feature_eng.save_features(
        os.path.join(output_dir, "all_datasets_features"),
        feature_type='all',
        max_rows_per_file=50000,  # Split large files
        fmt='csv'  # Use CSV format
    )
    
    # Example 2: Load from a directory containing multiple JSON files
    # data_dir = "outputs/test_multi"
    # feature_eng2 = FeatureEngineering()
    # feature_eng2.load_processed_data(data_dir)
    # feature_eng2.save_features(
    #     os.path.join(output_dir, "directory_load_features"),
    #     feature_type='combined'
    # )
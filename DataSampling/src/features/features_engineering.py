import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import spacy
import os
import json

class FeatureEngineering:
    """
    Feature engineering for Text2SQL tasks.
    Works with processed data from any dataset implementation.
    """
    
    def __init__(self, processed_data=None):
        """
        Initialize feature engineering.
        
        Args:
            processed_data: Optional pre-processed data
        """
        self.processed_data = processed_data
        self.features = {}
        self.datasets_present = set()
        
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
            if 'dataset' in item and not 'error' in item:
                self.datasets_present.add(item['dataset'].lower())
    
    def load_processed_data(self, file_path):
        """
        Load processed data from a file.
        
        Args:
            file_path: Path to the processed data file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)
            
        # Identify which datasets are present
        self._identify_datasets()
        print(f"Loaded data from {len(self.datasets_present)} datasets: {', '.join(self.datasets_present)}")
    
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
            
            # Add table-specific features
            tables = schema_analysis.get('tables', [])
            for idx, table in enumerate(tables[:10]):  # Limit to first 10 tables for large schemas
                prefix = f'table_{idx+1}'
                feature[f'{prefix}_name'] = table.get('name', '')
                feature[f'{prefix}_columns_count'] = table.get('columns_count', 0)
                feature[f'{prefix}_pk_count'] = table.get('primary_key_count', 0)
                feature[f'{prefix}_fk_count'] = table.get('foreign_key_count', 0)
                
                # Add datatype counts
                datatype_counts = table.get('datatype_counts', {})
                for dtype in ['TEXT', 'INTEGER', 'REAL', 'NUMERIC', 'BLOB', 'DATE', 'DATETIME', 'BOOLEAN']:
                    feature[f'{prefix}_{dtype.lower()}_count'] = datatype_counts.get(dtype, 0)
            
            features.append(feature)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Cache features
        self.features['schema'] = df
        
        return df
    
    def combine_features(self):
        """
        Combine all features into a single DataFrame.
        
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
        combined_df = pd.merge(question_df, sql_df, on=['question_id', 'db_id', 'dataset'], how='left')
        
        # Merge with schema features (many-to-one)
        combined_df = pd.merge(combined_df, schema_df, on=['db_id', 'dataset'], how='left')
        
        return combined_df
    
    def compute_complexity_score(self):
        """
        Compute a complexity score for each question.
        
        Returns:
            DataFrame with complexity scores
        """
        if 'question' not in self.features or 'sql' not in self.features:
            if not self.processed_data:
                raise ValueError("No processed data available")
            self.extract_question_features()
            self.extract_sql_features()
        
        question_df = self.features['question']
        sql_df = self.features['sql']
        
        # Merge dataframes
        merged_df = pd.merge(question_df, sql_df, on=['question_id', 'db_id', 'dataset'], how='left')
        
        # Define weights for complexity factors
        weights = {
            'q_word_length': 0.05,
            'q_entity_count': 0.05,
            'q_number_count': 0.05,
            'q_has_negation': 0.1,
            'q_has_comparatives': 0.1,
            'q_has_superlatives': 0.1,
            'sql_tables_count': 0.15,
            'sql_join_count': 0.1,
            'sql_where_conditions': 0.1,
            'sql_subquery_count': 0.15,
            'sql_clauses_count': 0.05,
            'sql_agg_function_count': 0.1
        }
        
        # Normalize the factors
        for factor in weights.keys():
            if factor in merged_df.columns:
                if factor.startswith('q_has_') and merged_df[factor].dtype == bool:
                    merged_df[factor] = merged_df[factor].astype(int)
                
                max_val = merged_df[factor].max()
                if max_val > 0:  # Avoid division by zero
                    merged_df[f'norm_{factor}'] = merged_df[factor] / max_val
                else:
                    merged_df[f'norm_{factor}'] = 0
        
        # Compute complexity score
        merged_df['complexity_score'] = 0
        for factor, weight in weights.items():
            if f'norm_{factor}' in merged_df.columns:
                merged_df['complexity_score'] += merged_df[f'norm_{factor}'] * weight
        
        # Scale to 0-10 range
        merged_df['complexity_score'] = merged_df['complexity_score'] * 10
        
        # Round to 2 decimal places
        merged_df['complexity_score'] = merged_df['complexity_score'].round(2)
        
        # Add dataset-specific complexity adjustments
        # This allows for dataset-specific calibration of complexity scores
        for dataset in self.datasets_present:
            dataset_mask = merged_df['dataset'] == dataset
            if dataset_mask.sum() > 0:
                # Optionally adjust scores for specific datasets
                # For example, BIRD might have more complex questions on average
                if dataset.lower() == 'bird':
                    # Apply a small multiplier for BIRD to account for its generally higher complexity
                    # This is just an example - you would calibrate based on your analysis
                    merged_df.loc[dataset_mask, 'complexity_score'] = merged_df.loc[dataset_mask, 'complexity_score'] * 1.05
                    merged_df.loc[dataset_mask, 'complexity_score'] = merged_df.loc[dataset_mask, 'complexity_score'].round(2)
                    
                # Cap at 10
                merged_df.loc[merged_df['complexity_score'] > 10, 'complexity_score'] = 10
        
        return merged_df[['question_id', 'db_id', 'dataset', 'complexity_score']]
    
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
                'dataset': dataset_name
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
                    feature['difficulty'] = item['orig_instance']['difficulty']
                # Add more Spider-specific features as needed
            
            # Add other dataset-specific features as needed
            
            features.append(feature)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Cache features with dataset prefix
        self.features[f'{dataset_name.lower()}_specific'] = df
        
        return df
    
    def save_features(self, output_path, feature_type='all'):
        """
        Save features to a file.
        
        Args:
            output_path: Path to save the features
            feature_type: Type of features to save ('all', 'question', 'sql', 'schema', 'combined', 
                                                  or a specific dataset name for dataset-specific features)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if feature_type == 'all':
            # Save all feature types
            for ft in ['question', 'sql', 'schema', 'combined', 'complexity']:
                if ft == 'combined':
                    df = self.combine_features()
                elif ft == 'complexity':
                    df = self.compute_complexity_score()
                elif ft in self.features:
                    df = self.features[ft]
                else:
                    if ft == 'question':
                        df = self.extract_question_features()
                    elif ft == 'sql':
                        df = self.extract_sql_features()
                    elif ft == 'schema':
                        df = self.extract_schema_features()
                
                # Save to CSV
                ft_path = output_path.replace('.csv', f'_{ft}.csv')
                df.to_csv(ft_path, index=False)
                print(f"Saved {ft} features to {ft_path}")
                
            # Save dataset-specific features for each dataset
            for dataset in self.datasets_present:
                df = self.extract_dataset_specific_features(dataset)
                if not df.empty:
                    ds_path = output_path.replace('.csv', f'_{dataset}_specific.csv')
                    df.to_csv(ds_path, index=False)
                    print(f"Saved {dataset}-specific features to {ds_path}")
        else:
            # Save specific feature type
            if feature_type == 'combined':
                df = self.combine_features()
            elif feature_type == 'complexity':
                df = self.compute_complexity_score()
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
                else:
                    # Check if it's a dataset name
                    if feature_type.lower() in self.datasets_present:
                        df = self.extract_dataset_specific_features(feature_type)
                    else:
                        raise ValueError(f"Unknown feature type: {feature_type}")
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Saved {feature_type} features to {output_path}")


# Example usage
if __name__ == "__main__":
    # Path to processed data that contains multiple datasets
    processed_data_path = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/outputs/test_multi/processed_combined_data.json"
    
    # Create feature engineering object
    feature_eng = FeatureEngineering()
    
    # Load processed data
    feature_eng.load_processed_data(processed_data_path)
    
    # Extract features
    question_features = feature_eng.extract_question_features()
    sql_features = feature_eng.extract_sql_features()
    schema_features = feature_eng.extract_schema_features()
    
    print(f"Extracted {len(question_features)} question features")
    print(f"Extracted {len(sql_features)} SQL features")
    print(f"Extracted {len(schema_features)} schema features")
    
    # Combine features
    combined_features = feature_eng.combine_features()
    print(f"Combined into {len(combined_features)} feature sets")
    
    # Compute complexity score
    complexity_scores = feature_eng.compute_complexity_score()
    print(f"Computed complexity scores")
    
    # Extract dataset-specific features
    for dataset in feature_eng.datasets_present:
        dataset_features = feature_eng.extract_dataset_specific_features(dataset)
        print(f"Extracted {len(dataset_features)} {dataset}-specific features")
    
    # Save all features
    output_dir = "outputs/features"
    feature_eng.save_features(os.path.join(output_dir, "all_datasets_features.csv"), feature_type='all')
import os
import json
import argparse
import gzip
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SubsetGenerator")


class Text2SQLSubsetGenerator:
    """
    Generates a balanced subset of a Text2SQL dataset while preserving
    the distribution of important features.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the subset generator.
        
        Args:
            random_seed: Seed for random operations to ensure reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def load_data(self, filepath):
        """
        Load data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            The loaded data
        """
        logger.info(f"Loading data from: {filepath}")
        try:
            # Check if file is gzipped
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Verify data structure
            if 'instances' not in data or not isinstance(data['instances'], list):
                raise ValueError("Invalid data format: expected 'instances' list")
            
            logger.info(f"Loaded {len(data['instances'])} instances")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def extract_features(self, data):
        """
        Extract features from the dataset for stratification.
        
        Args:
            data: The loaded dataset
            
        Returns:
            A list of dictionaries with extracted features for each instance
        """
        logger.info("Extracting features for stratification...")
        
        features = []
        schemas = data.get('schemas', {})
        
        for instance in tqdm(data['instances'], desc="Extracting Features"):
            # Get basic information
            db_id = instance.get('db_id')
            sql_analysis = instance.get('sql_analysis', {})
            question_analysis = instance.get('question_analysis', {})
            
            # Extract key features
            feature = {
                'instance_index': len(features),  # Keep track of original index
                'db_id': db_id,
                'dataset': instance.get('dataset', 'unknown'),
                'difficulty': instance.get('difficulty', 'unknown'),
                
                # Question complexity
                'question_length': len(instance.get('question', '')),
                'has_numbers': question_analysis.get('has_numbers', False),
                'has_entities': question_analysis.get('has_entities', False),
                
                # SQL complexity
                'query_type': self._get_query_type(sql_analysis),
                'tables_count': sql_analysis.get('tables_count', 1),
                'join_count': sql_analysis.get('join_count', 0),
                'has_aggregation': bool(sql_analysis.get('aggregation_functions', [])),
                'aggregation_type': next(iter(sql_analysis.get('aggregation_functions', [])), 'none'),
                
                # Schema complexity
                'schema_size': self._get_schema_size(db_id, schemas)
            }
            
            features.append(feature)
        
        logger.info(f"Extracted features for {len(features)} instances")
        return features
    
    def _get_query_type(self, sql_analysis):
        """
        Determine the query type based on SQL analysis.
        
        Args:
            sql_analysis: The SQL analysis dictionary
            
        Returns:
            A string representing the query type
        """
        if sql_analysis.get('subquery_count', 0) > 0:
            if 'GROUP BY' in sql_analysis.get('clause_types', []):
                return 'nested_with_groupby'
            return 'nested'
        
        if 'GROUP BY' in sql_analysis.get('clause_types', []):
            if 'HAVING' in sql_analysis.get('clause_types', []):
                return 'groupby_having'
            return 'groupby'
        
        if sql_analysis.get('join_count', 0) > 0:
            return 'join'
        
        if 'ORDER BY' in sql_analysis.get('clause_types', []):
            return 'orderby'
        
        return 'simple'
    
    def _get_schema_size(self, db_id, schemas):
        """
        Categorize the schema size.
        
        Args:
            db_id: Database ID
            schemas: Dictionary of schemas
            
        Returns:
            Schema size category
        """
        if db_id not in schemas:
            return 'unknown'
        
        table_count = schemas[db_id].get('table_count', 0)
        
        if table_count <= 3:
            return 'small'
        elif table_count <= 7:
            return 'medium'
        else:
            return 'large'
    
    def create_stratification_variables(self, features):
        """
        Create stratification variables for balanced sampling.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            List of stratification variables
        """
        logger.info("Creating stratification variables...")
        
        # Convert features to DataFrame for easier manipulation
        df = pd.DataFrame(features)
        
        # Define key stratification dimensions
        strat_vars = []
        
        for _, row in df.iterrows():
            # Create stratification string without dataset (since all instances in a file belong to one dataset)
            strat_var = f"{row['query_type']}_{row['schema_size']}"
            
            # Add aggregation presence as a stratification dimension
            if row['has_aggregation']:
                strat_var += f"_agg_{row['aggregation_type']}"
            
            # Add question complexity as a stratification dimension
            if row['question_length'] < 30:
                q_length = 'short'
            elif row['question_length'] < 70:
                q_length = 'medium'
            else:
                q_length = 'long'
            
            strat_var += f"_{q_length}"
            
            strat_vars.append(strat_var)
        
        logger.info(f"Created {len(set(strat_vars))} unique stratification groups")
        return strat_vars
    
    def create_subset(self, data, subset_size, train_test_split_ratio=None):
        """
        Create a balanced subset of the dataset.
        
        Args:
            data: The original dataset
            subset_size: Desired size of the subset
            train_test_split_ratio: Optional ratio for train/test split (e.g., 0.2 for 20% test)
            
        Returns:
            The subset dataset
        """
        logger.info(f"Creating subset of size {subset_size}...")
        
        # Extract features
        features = self.extract_features(data)
        
        # Create stratification variables
        strat_vars = self.create_stratification_variables(features)

        # Ensure subset size is not larger than original dataset
        subset_size = min(subset_size, len(features))
        
        # Create indices for stratified sampling
        if train_test_split_ratio is not None:
            # Stratified train-test split
            train_indices, test_indices = train_test_split(
                range(len(features)),
                test_size=train_test_split_ratio,
                stratify=strat_vars,
                random_state=self.random_seed
            )
            
            # Calculate proportional sizes
            train_size = int(subset_size * (1 - train_test_split_ratio))
            test_size = subset_size - train_size
            
            # Sample from train and test indices
            train_strat_vars = [strat_vars[i] for i in train_indices]
            test_strat_vars = [strat_vars[i] for i in test_indices]
            
            # Use sklearn's stratified shuffle split for sampling
            train_subset_indices, _ = train_test_split(
                train_indices,
                test_size=len(train_indices) - train_size,
                stratify=train_strat_vars,
                random_state=self.random_seed
            )
            
            test_subset_indices, _ = train_test_split(
                test_indices,
                test_size=len(test_indices) - test_size,
                stratify=test_strat_vars,
                random_state=self.random_seed
            )
            
            # Combine indices
            subset_indices = sorted(list(train_subset_indices) + list(test_subset_indices))
            
            # Create train and test subset data
            train_subset = self._create_subset_from_indices(data, [features[i]['instance_index'] for i in train_subset_indices])
            test_subset = self._create_subset_from_indices(data, [features[i]['instance_index'] for i in test_subset_indices])
            
            logger.info(f"Created train subset with {len(train_subset['instances'])} instances")
            logger.info(f"Created test subset with {len(test_subset['instances'])} instances")
            
            return {
                'train': train_subset,
                'test': test_subset,
                'combined': self._create_subset_from_indices(data, subset_indices)
            }
        else:
            # Simple stratified sampling without train/test split
            subset_indices, _ = train_test_split(
                range(len(features)),
                test_size=len(features) - subset_size,
                stratify=strat_vars,
                random_state=self.random_seed
            )
            
            # Create subset using selected indices
            subset_data = self._create_subset_from_indices(data, [features[i]['instance_index'] for i in subset_indices])
            
            logger.info(f"Created subset with {len(subset_data['instances'])} instances")
            
            return subset_data
    
    def _create_subset_from_indices(self, data, indices):
        """
        Create a subset of the dataset using selected indices.
        
        Args:
            data: The original dataset
            indices: List of instance indices to include
            
        Returns:
            The subset dataset
        """
        # Create a new dataset with the same structure
        subset = {
            'instances': [data['instances'][i] for i in indices],
            'schemas': {}
        }
        
        # Add only the schemas that are referenced in the subset
        referenced_dbs = set(instance.get('db_id') for instance in subset['instances'])
        
        for db_id in referenced_dbs:
            if db_id in data.get('schemas', {}):
                subset['schemas'][db_id] = data['schemas'][db_id]
        
        return subset
    
    def analyze_distribution(self, original_data, subset_data):
        """
        Analyze the distribution of features between original and subset.
        
        Args:
            original_data: The original dataset
            subset_data: The subset dataset
            
        Returns:
            Dictionary of distribution comparisons
        """
        logger.info("Analyzing feature distribution in subset vs original...")
        
        analysis = {}
        
        # Extract features for both datasets
        original_features = self.extract_features(original_data)
        subset_features = self.extract_features(subset_data)
        
        # Convert to DataFrames
        original_df = pd.DataFrame(original_features)
        subset_df = pd.DataFrame(subset_features)
        
        # Analyze key distributions
        
        # 1. Query types
        original_query_types = original_df['query_type'].value_counts(normalize=True)
        subset_query_types = subset_df['query_type'].value_counts(normalize=True)
        
        analysis['query_types'] = {
            'original': original_query_types.to_dict(),
            'subset': subset_query_types.to_dict()
        }
        
        # 2. Schema sizes
        original_schema_sizes = original_df['schema_size'].value_counts(normalize=True)
        subset_schema_sizes = subset_df['schema_size'].value_counts(normalize=True)
        
        analysis['schema_sizes'] = {
            'original': original_schema_sizes.to_dict(),
            'subset': subset_schema_sizes.to_dict()
        }
        
        # 3. Aggregation types
        original_agg_types = original_df['aggregation_type'].value_counts(normalize=True)
        subset_agg_types = subset_df['aggregation_type'].value_counts(normalize=True)
        
        analysis['aggregation_types'] = {
            'original': original_agg_types.to_dict(),
            'subset': subset_agg_types.to_dict()
        }
        
        # 4. Question complexity
        analysis['question_complexity'] = {
            'original': {
                'avg_length': original_df['question_length'].mean(),
                'has_numbers': original_df['has_numbers'].mean(),
                'has_entities': original_df['has_entities'].mean()
            },
            'subset': {
                'avg_length': subset_df['question_length'].mean(),
                'has_numbers': subset_df['has_numbers'].mean(),
                'has_entities': subset_df['has_entities'].mean()
            }
        }
        
        # 5. Database distribution
        original_dbs = original_df['db_id'].value_counts(normalize=True)
        subset_dbs = subset_df['db_id'].value_counts(normalize=True)
        
        # Just keep top 10 most common DBs to avoid too much detail
        analysis['databases'] = {
            'original': original_dbs.head(10).to_dict(),
            'subset': subset_dbs.head(10).to_dict()
        }
        
        logger.info("Distribution analysis complete")
        
        return analysis
    
    def generate_distribution_report(self, analysis, output_dir):
        """
        Generate visualizations of the distribution analysis.
        
        Args:
            analysis: Distribution analysis dictionary
            output_dir: Directory to save visualizations
        """
        logger.info(f"Generating distribution report in {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Query types comparison
        plt.figure(figsize=(12, 6))
        query_types = sorted(set(list(analysis['query_types']['original'].keys()) + 
                             list(analysis['query_types']['subset'].keys())))
        
        original_values = [analysis['query_types']['original'].get(qt, 0) for qt in query_types]
        subset_values = [analysis['query_types']['subset'].get(qt, 0) for qt in query_types]
        
        x = np.arange(len(query_types))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Original')
        plt.bar(x + width/2, subset_values, width, label='Subset')
        
        plt.ylabel('Proportion')
        plt.title('Query Type Distribution Comparison')
        plt.xticks(x, query_types, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'query_types_comparison.png'))
        plt.close()
        
        # 2. Schema sizes comparison
        plt.figure(figsize=(10, 6))
        schema_sizes = sorted(set(list(analysis['schema_sizes']['original'].keys()) + 
                              list(analysis['schema_sizes']['subset'].keys())))
        
        original_values = [analysis['schema_sizes']['original'].get(ss, 0) for ss in schema_sizes]
        subset_values = [analysis['schema_sizes']['subset'].get(ss, 0) for ss in schema_sizes]
        
        x = np.arange(len(schema_sizes))
        
        plt.bar(x - width/2, original_values, width, label='Original')
        plt.bar(x + width/2, subset_values, width, label='Subset')
        
        plt.ylabel('Proportion')
        plt.title('Schema Size Distribution Comparison')
        plt.xticks(x, schema_sizes)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'schema_sizes_comparison.png'))
        plt.close()
        
        # 3. Aggregation types comparison
        plt.figure(figsize=(12, 6))
        agg_types = sorted(set(list(analysis['aggregation_types']['original'].keys()) + 
                           list(analysis['aggregation_types']['subset'].keys())))
        
        original_values = [analysis['aggregation_types']['original'].get(at, 0) for at in agg_types]
        subset_values = [analysis['aggregation_types']['subset'].get(at, 0) for at in agg_types]
        
        x = np.arange(len(agg_types))
        
        plt.bar(x - width/2, original_values, width, label='Original')
        plt.bar(x + width/2, subset_values, width, label='Subset')
        
        plt.ylabel('Proportion')
        plt.title('Aggregation Type Distribution Comparison')
        plt.xticks(x, agg_types, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aggregation_types_comparison.png'))
        plt.close()
        
        # 4. Question complexity comparison
        plt.figure(figsize=(10, 6))
        complexity_metrics = ['avg_length', 'has_numbers', 'has_entities']
        
        # Normalize avg_length to be comparable with boolean metrics
        avg_length_original = analysis['question_complexity']['original']['avg_length']
        avg_length_subset = analysis['question_complexity']['subset']['avg_length']
        max_avg_length = max(avg_length_original, avg_length_subset)
        
        # Normalized metrics
        original_values = [
            analysis['question_complexity']['original']['avg_length'] / max_avg_length,
            analysis['question_complexity']['original']['has_numbers'],
            analysis['question_complexity']['original']['has_entities']
        ]
        
        subset_values = [
            analysis['question_complexity']['subset']['avg_length'] / max_avg_length,
            analysis['question_complexity']['subset']['has_numbers'],
            analysis['question_complexity']['subset']['has_entities']
        ]
        
        x = np.arange(len(complexity_metrics))
        
        plt.bar(x - width/2, original_values, width, label='Original')
        plt.bar(x + width/2, subset_values, width, label='Subset')
        
        plt.ylabel('Normalized Value')
        plt.title('Question Complexity Comparison')
        plt.xticks(x, complexity_metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'question_complexity_comparison.png'))
        plt.close()
        
        # 5. Create a summary report
        with open(os.path.join(output_dir, 'distribution_report.txt'), 'w') as f:
            f.write("DISTRIBUTION COMPARISON REPORT\n")
            f.write("=============================\n\n")
            
            # Question complexity
            f.write("Question Complexity:\n")
            f.write(f"  Average Length: {avg_length_original:.2f} (original) vs {avg_length_subset:.2f} (subset)\n")
            f.write(f"  Has Numbers: {analysis['question_complexity']['original']['has_numbers']:.2%} (original) vs {analysis['question_complexity']['subset']['has_numbers']:.2%} (subset)\n")
            f.write(f"  Has Entities: {analysis['question_complexity']['original']['has_entities']:.2%} (original) vs {analysis['question_complexity']['subset']['has_entities']:.2%} (subset)\n\n")
            
            # Query types
            f.write("Query Types Distribution:\n")
            for qt in query_types:
                original_pct = analysis['query_types']['original'].get(qt, 0) * 100
                subset_pct = analysis['query_types']['subset'].get(qt, 0) * 100
                f.write(f"  {qt}: {original_pct:.2f}% (original) vs {subset_pct:.2f}% (subset)\n")
            f.write("\n")
            
            # Schema sizes
            f.write("Schema Size Distribution:\n")
            for ss in schema_sizes:
                original_pct = analysis['schema_sizes']['original'].get(ss, 0) * 100
                subset_pct = analysis['schema_sizes']['subset'].get(ss, 0) * 100
                f.write(f"  {ss}: {original_pct:.2f}% (original) vs {subset_pct:.2f}% (subset)\n")
            f.write("\n")
            
            # Aggregation types
            f.write("Aggregation Types Distribution:\n")
            for at in agg_types:
                original_pct = analysis['aggregation_types']['original'].get(at, 0) * 100
                subset_pct = analysis['aggregation_types']['subset'].get(at, 0) * 100
                f.write(f"  {at}: {original_pct:.2f}% (original) vs {subset_pct:.2f}% (subset)\n")
        
        logger.info(f"Distribution report saved to {output_dir}")
    
    def save_subset(self, subset_data, output_path, compress=False):
        """
        Save the subset to a file.
        
        Args:
            subset_data: The subset dataset
            output_path: Path to save the subset
            compress: Whether to compress with gzip
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if compress or output_path.endswith('.gz'):
            logger.info(f"Saving compressed subset to: {output_path}")
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(subset_data, f)
        else:
            logger.info(f"Saving subset to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(subset_data, f)


def main():
    parser = argparse.ArgumentParser(description='Text2SQL Dataset Subset Generator')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--subset-size', type=int, required=True, help='Size of the subset to create')
    parser.add_argument('--train-test-split', type=float, help='Train-test split ratio (e.g., 0.2 for 20% test)')
    parser.add_argument('--report-dir', type=str, default='subset_report', help='Directory for distribution report')
    parser.add_argument('--compress', action='store_true', help='Compress output with gzip')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create subset generator
    generator = Text2SQLSubsetGenerator(random_seed=args.seed)
    
    # Load data
    data = generator.load_data(args.input)
    
    # Create subset
    if args.train_test_split:
        subset_data = generator.create_subset(data, args.subset_size, args.train_test_split)
        
        # Save train and test subsets
        base_path, ext = os.path.splitext(args.output)
        
        train_path = f"{base_path}_train{ext}"
        generator.save_subset(subset_data['train'], train_path, args.compress)
        
        test_path = f"{base_path}_test{ext}"
        generator.save_subset(subset_data['test'], test_path, args.compress)
        
        # Use combined subset for distribution analysis
        subset_for_analysis = subset_data['combined']
    else:
        subset_data = generator.create_subset(data, args.subset_size)
        generator.save_subset(subset_data, args.output, args.compress)
        subset_for_analysis = subset_data
    
    # Analyze distribution
    analysis = generator.analyze_distribution(data, subset_for_analysis)
    
    # Generate distribution report
    generator.generate_distribution_report(analysis, args.report_dir)
    
    logger.info("Subset generation complete!")
    
    # Print summary
    print("\nSubset Generation Summary:")
    print(f"Original Dataset: {len(data['instances'])} instances")
    
    if args.train_test_split:
        print(f"Train Subset: {len(subset_data['train']['instances'])} instances")
        print(f"Test Subset: {len(subset_data['test']['instances'])} instances")
        print(f"Total Subset: {len(subset_data['combined']['instances'])} instances")
    else:
        print(f"Subset: {len(subset_data['instances'])} instances")
    
    print(f"Distribution Report: {args.report_dir}")


if __name__ == "__main__":
    main()
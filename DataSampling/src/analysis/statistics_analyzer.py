import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path
import os
import sqlparse
import glob
import argparse
from tqdm.auto import tqdm

def created_difficulty(sql: str):
    """
    Calculate difficulty level based on SQL query complexity.
    
    Args:
        sql: SQL query string
    
    Returns:
        Difficulty level: 'simple', 'moderate', or 'challenging'
    """
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


def load_processed_data(processed_dir: str):
    """
    Load data from the processed directory structure that data_processor.py creates.
    
    Args:
        processed_dir: Directory containing processed data
        
    Returns:
        Dict containing loaded instances and schemas
    """
    instances_dir = os.path.join(processed_dir, "instances")
    schemas_dir = os.path.join(processed_dir, "schemas")
    
    print(f"Loading instances from {instances_dir}")
    print(f"Loading schemas from {schemas_dir}")
    
    # Check if directories exist
    if not os.path.exists(instances_dir):
        raise FileNotFoundError(f"Instances directory not found: {instances_dir}")
    if not os.path.exists(schemas_dir):
        raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")
    
    # Load metadata if exists
    metadata_path = os.path.join(processed_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata.get('dataset_name', 'unknown')} dataset with {metadata.get('total_instances', 'unknown')} instances")
    
    # Load instances
    instances = []
    instance_files = glob.glob(os.path.join(instances_dir, "*.json"))
    
    for file_path in tqdm(instance_files, desc="Loading instances"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                instance = json.load(f)
                instances.append(instance)
        except Exception as e:
            print(f"Error loading instance file {file_path}: {e}")
    
    # Load schemas
    schemas = {}
    schema_files = glob.glob(os.path.join(schemas_dir, "*.json"))
    
    for file_path in tqdm(schema_files, desc="Loading schemas"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                db_name = schema.get('db_name', os.path.basename(file_path).replace('.json', ''))
                schemas[db_name] = schema
        except Exception as e:
            print(f"Error loading schema file {file_path}: {e}")
    
    print(f"Loaded {len(instances)} instances and {len(schemas)} schemas")
    
    return {
        "instances": instances,
        "schemas": schemas,
        "metadata": metadata
    }


def extract_metrics(data):
    """
    Extract metrics from processed data for analysis.
    
    Args:
        data: Dict containing instances and schemas from processed data
        
    Returns:
        Dict containing metrics for analysis
    """
    instances = data.get('instances', [])
    schemas = data.get('schemas', {})
    
    # Initialize metrics 
    metrics = {
        'instances': instances,
        'schemas': schemas,
        'difficulties': [],
        'question_lengths': [],
        'word_counts': [],
        'sql_lengths': [],
        'entity_presence': [],
        'entity_counts': [],
        'entity_types': [],
        'superlative_presence': [],
        'comparative_presence': [],
        'negation_presence': [],
        'join_counts': [],
        'where_conditions': [],
        'subquery_counts': [],
        'aggregation_counts': [],
        'aggregation_types_occurrences': {},
        'clause_types_occurrences': {},
        'table_counts': [],
        'column_counts': [],
        'fk_counts': [],
        'pk_counts': [],
        'table_overlaps': [],
        'column_overlaps': [],
        'table_lemma_overlaps': [],
        'column_lemma_overlaps': [],
        'sql_table_occurance': [],
        'has_evidence': [],
        'db_ids': []
    }
    
    # Extract metrics from each instance
    for item in instances:
        # Skip items with errors
        if 'error' in item:
            continue
            
        # Database ID
        metrics['db_ids'].append(item.get('db_id', ''))
        
        # Difficulty
        # Use the provided difficulty or calculate it
        if 'difficulty' not in item or item['difficulty'] == 'unknown':
            metrics['difficulties'].append(created_difficulty(item['sql']))
        else:
            metrics['difficulties'].append(item.get('difficulty', 'unknown'))
        
        # Question features
        q_analysis = item.get('question_analysis', {})
        metrics['question_lengths'].append(q_analysis.get('char_length', 0))
        metrics['word_counts'].append(q_analysis.get('word_length', 0))
        
        # Entity information
        metrics['entity_presence'].append(q_analysis.get('has_entities', False))
        metrics['entity_counts'].append(len(q_analysis.get('entities', [])))
        
        for ent_type in q_analysis.get('entity_types', []):
            metrics['entity_types'].append(ent_type)

        # Linguistic features
        metrics['superlative_presence'].append(q_analysis.get('has_superlatives', False))
        metrics['comparative_presence'].append(q_analysis.get('has_comparatives', False))
        metrics['negation_presence'].append(q_analysis.get('has_negation', False))
        
        # Schema overlaps
        metrics['table_overlaps'].append(q_analysis.get('table_overlap_count', 0))
        metrics['column_overlaps'].append(q_analysis.get('column_overlap_count', 0))
        metrics['table_lemma_overlaps'].append(q_analysis.get('table_overlap_lemma_count', 0))
        metrics['column_lemma_overlaps'].append(q_analysis.get('column_overlap_lemma_count', 0))
        
        # SQL features
        sql_analysis = item.get('sql_analysis', {})
        metrics['sql_lengths'].append(sql_analysis.get('char_length', 0))
        metrics['sql_table_occurance'].append(sql_analysis.get('tables_count', 0))
        metrics['join_counts'].append(sql_analysis.get('join_count', 0))
        metrics['where_conditions'].append(sql_analysis.get('where_conditions', 0))
        metrics['subquery_counts'].append(sql_analysis.get('subquery_count', 0))
        metrics['aggregation_counts'].append(sql_analysis.get('aggregation_function_count', 0))

        if 'aggregation_functions' in sql_analysis:
            for agg_func in sql_analysis['aggregation_functions']:
                metrics['aggregation_types_occurrences'][agg_func] = metrics['aggregation_types_occurrences'].get(agg_func, 0) + 1
        
        if 'clause_types' in sql_analysis:
            for clause in sql_analysis['clause_types']:
                metrics['clause_types_occurrences'][clause] = metrics['clause_types_occurrences'].get(clause, 0) + 1
        
        # Evidence features
        evidence = item.get('evidence', '')
        metrics['has_evidence'].append(bool(evidence))
    
    # Schema statistics
    for db_id, schema in schemas.items():
        # Use table_count directly if available, otherwise count tables
        table_count = schema.get('table_count', len(schema.get('tables', [])))
        metrics['table_counts'].append(table_count)
        
        # Use column_count directly if available, otherwise count columns
        column_count = schema.get('column_count', len(schema.get('columns', [])))
        metrics['column_counts'].append(column_count)
        
        # Get primary and foreign key counts
        metrics['pk_counts'].append(schema.get('primary_key_count', 0))
        metrics['fk_counts'].append(schema.get('foreign_key_count', 0))

    return metrics


def analyze_metrics(metrics):
    """
    Analyze the collected metrics and generate statistics.

    Args:
        metrics: Dictionary containing all collected metrics
    Returns:
        analysis: Dictionary containing analysis results
    """
    analysis = {
        'general': {
            'total_examples': len(metrics['instances']),
            'total_databases': len(metrics['schemas']),
        },
        'difficulty': {},
        'question_features': {},
        'sql_features': {},
        'schema_features': {},
        'linguistic_features': {},
        'evidence_features': {},
    }
    
    # Calculate statistics
    analysis['difficulty'] = dict(Counter(metrics['difficulties']))
    
    # DB statistics
    analysis['database_stats'] = {
        'unique_dbs': len(set(metrics['db_ids'])),
        'db_use_frequency': dict(Counter(metrics['db_ids']).most_common(10))  # Top 10 most used DBs
    }
    
    # Question features
    analysis['question_features'] = {
        'length_stats': {
            'char_mean': np.mean(metrics['question_lengths']) if metrics['question_lengths'] else 0,
            'char_std': np.std(metrics['question_lengths']) if metrics['question_lengths'] else 0,
            'word_mean': np.mean(metrics['word_counts']) if metrics['word_counts'] else 0,
            'word_std': np.std(metrics['word_counts']) if metrics['word_counts'] else 0,
        },
        'entity_stats': {
            'has_entities_percent': sum(metrics['entity_presence']) / len(metrics['entity_presence']) * 100 if metrics['entity_presence'] else 0,
            'entity_count_avg': np.mean(metrics['entity_counts']) if metrics['entity_counts'] else 0,
            'entity_types': dict(Counter(metrics['entity_types']))
        },
        'linguistic_features': {
            'has_superlatives_percent': sum(metrics['superlative_presence']) / len(metrics['superlative_presence']) * 100 if metrics['superlative_presence'] else 0,
            'has_comparatives_percent': sum(metrics['comparative_presence']) / len(metrics['comparative_presence']) * 100 if metrics['comparative_presence'] else 0,
            'has_negation_percent': sum(metrics['negation_presence']) / len(metrics['negation_presence']) * 100 if metrics['negation_presence'] else 0
        },
        'schema_overlap_avg': {
            'table_overlap_count_avg': np.mean(metrics['table_overlaps']) if metrics['table_overlaps'] else 0,
            'column_overlap_count_avg': np.mean(metrics['column_overlaps']) if metrics['column_overlaps'] else 0,
            'table_overlap_lemma_count_avg': np.mean(metrics['table_lemma_overlaps']) if metrics['table_lemma_overlaps'] else 0,
            'column_overlap_lemma_count_avg': np.mean(metrics['column_lemma_overlaps']) if metrics['column_lemma_overlaps'] else 0,
        }
    }
    
    # SQL features
    analysis['sql_features'] = {
        'length_stats': {
            'mean': np.mean(metrics['sql_lengths']) if metrics['sql_lengths'] else 0,
            'std': np.std(metrics['sql_lengths']) if metrics['sql_lengths'] else 0,
        },
        'sql_table_occurrence': dict(Counter(metrics['sql_table_occurance'])),
        'where_conditions': dict(Counter(metrics['where_conditions'])),
        'subquery_counts': dict(Counter(metrics['subquery_counts'])),
        'aggregation_stats': {
            'aggregation_usage_percent': sum(1 for count in metrics['aggregation_counts'] if count > 0) / len(metrics['aggregation_counts']) * 100 if metrics['aggregation_counts'] else 0,
            'aggregation_types': metrics['aggregation_types_occurrences'],
        },
        'clause_types': metrics['clause_types_occurrences'],
        'join_stats': {
            'join_counts': dict(Counter(metrics['join_counts'])),
            'no_join_percent': sum(1 for count in metrics['join_counts'] if count == 0) / len(metrics['join_counts']) * 100 if metrics['join_counts'] else 0,
            'has_join_percent': sum(1 for count in metrics['join_counts'] if count > 0) / len(metrics['join_counts']) * 100 if metrics['join_counts'] else 0,
            'multi_join_percent': sum(1 for count in metrics['join_counts'] if count > 1) / len(metrics['join_counts']) * 100 if metrics['join_counts'] else 0
        }
    }
    
    # Schema features
    analysis['schema_features'] = {
        'table_count_stats': {
            'min': min(metrics['table_counts']) if metrics['table_counts'] else 0,
            'max': max(metrics['table_counts']) if metrics['table_counts'] else 0,
            'mean': np.mean(metrics['table_counts']) if metrics['table_counts'] else 0,
        },
        'column_count_stats': {
            'min': min(metrics['column_counts']) if metrics['column_counts'] else 0,
            'max': max(metrics['column_counts']) if metrics['column_counts'] else 0,
            'mean': np.mean(metrics['column_counts']) if metrics['column_counts'] else 0,
        },
        'keys_stats': {
            'avg_primary_keys': np.mean(metrics['pk_counts']) if metrics['pk_counts'] else 0,
            'avg_foreign_keys': np.mean(metrics['fk_counts']) if metrics['fk_counts'] else 0
        }
    }
     
    # Evidence features
    analysis['evidence_features'] = {
        'has_evidence_percent': sum(metrics['has_evidence']) / len(metrics['has_evidence']) * 100 if metrics['has_evidence'] else 0,
    }
    
    return analysis


def create_visualizations(analysis, output_dir='visualizations'):
    """
    Create visualizations from the analysis results
    
    Args:
        analysis: Dictionary containing analysis results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("colorblind")
    
    # Difficulty distribution
    if analysis['difficulty']:
        difficulties = analysis['difficulty']
        plt.figure(figsize=(10, 6))
        plt.bar(difficulties.keys(), difficulties.values())
        plt.title('Query Difficulty Distribution', fontsize=14)
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'difficulty_distribution.png'), dpi=300)
        plt.close()
    
    # Question and SQL length distribution
    question_lengths = analysis['question_features']['length_stats']
    sql_features = analysis['sql_features']
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Character Length', 'Word Count'], 
            [question_lengths['char_mean'], question_lengths['word_mean']])
    plt.title('Average Question Length', fontsize=14)
    plt.ylabel('Average Count', fontsize=12)
    
    plt.subplot(1, 2, 2)
    plt.bar(['SQL Length'], [sql_features['length_stats']['mean']])
    plt.title('Average SQL Length', fontsize=14)
    plt.ylabel('Character Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_stats.png'), dpi=300)
    plt.close()
    
    # Join distribution
    join_stats = sql_features['join_stats']
    plt.figure(figsize=(10, 8))
    labels = ['No Joins', 'Has Joins', 'Multi Joins']
    values = [join_stats['no_join_percent'], 
              join_stats['has_join_percent'], 
              join_stats['multi_join_percent']]
    
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('SQL Join Distribution', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'join_distribution.png'), dpi=300)
    plt.close()
    
    # Schema features
    schema_features = analysis['schema_features']
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    table_stats = schema_features['table_count_stats']
    plt.bar(['Minimum', 'Maximum', 'Average'], 
            [table_stats['min'], table_stats['max'], table_stats['mean']])
    plt.title('Table Count Statistics', fontsize=14)
    plt.ylabel('Number of Tables', fontsize=12)
    
    plt.subplot(1, 2, 2)
    key_stats = schema_features['keys_stats']
    plt.bar(['Primary Keys', 'Foreign Keys'], 
            [key_stats['avg_primary_keys'], key_stats['avg_foreign_keys']])
    plt.title('Average Keys per Database', fontsize=14)
    plt.ylabel('Average Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'schema_stats.png'), dpi=300)
    plt.close()
    
    # Linguistic features
    linguistic = analysis['question_features']['linguistic_features']
    plt.figure(figsize=(10, 6))
    features = ['Superlatives', 'Comparatives', 'Negations']
    percentages = [linguistic['has_superlatives_percent'],
                  linguistic['has_comparatives_percent'],
                  linguistic['has_negation_percent']]
    
    plt.bar(features, percentages, color=sns.color_palette("colorblind")[:3])
    plt.title('Linguistic Features in Questions', fontsize=14)
    plt.ylabel('Percentage of Questions', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(percentages):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linguistic_features.png'), dpi=300)
    plt.close()
    
    # Aggregation types
    agg_types = analysis['sql_features']['aggregation_stats']['aggregation_types']
    if agg_types:
        plt.figure(figsize=(10, 6))
        plt.bar(agg_types.keys(), agg_types.values())
        plt.title('Aggregation Function Usage', fontsize=14)
        plt.xlabel('Aggregation Function', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aggregation_types.png'), dpi=300)
        plt.close()
    
    # Table and column overlaps
    overlaps = analysis['question_features']['schema_overlap_avg']
    plt.figure(figsize=(10, 6))
    overlap_data = [
        overlaps['table_overlap_count_avg'],
        overlaps['column_overlap_count_avg'],
        overlaps['table_overlap_lemma_count_avg'],
        overlaps['column_overlap_lemma_count_avg']
    ]
    overlap_labels = [
        'Table Exact Match', 
        'Column Exact Match',
        'Table Lemma Match',
        'Column Lemma Match'
    ]
    
    plt.bar(overlap_labels, overlap_data)
    plt.title('Schema Term Overlap in Questions', fontsize=14)
    plt.ylabel('Average Count per Question', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'schema_overlap.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def analyze_text2sql_dataset(processed_dir, output_file='analysis_results.json', visualize=True, viz_dir='visualizations'):
    """
    Complete workflow to analyze a Text2SQL dataset from processed directory
    
    Args:
        processed_dir: Directory containing processed data
        output_file: Path to save analysis results
        visualize: Whether to generate visualizations
        viz_dir: Directory to save visualizations
        
    Returns:
        analysis: Dictionary containing complete analysis
    """
    # Load data from processed directory
    data = load_processed_data(processed_dir)
    
    # Extract metrics
    metrics = extract_metrics(data)
    
    # Analyze the metrics
    analysis = analyze_metrics(metrics)
    
    # Add metadata
    analysis['metadata'] = data.get('metadata', {})
    
    # Save analysis results
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis results saved to {output_file}")
    
    # Generate visualizations if requested
    if visualize:
        create_visualizations(analysis, viz_dir)
    
    return analysis


def combine_analyses(analysis_files, output_file='combined_analysis.json'):
    """
    Combine multiple analysis results into one.
    
    Args:
        analysis_files: List of analysis result files
        output_file: Path to save combined analysis
        
    Returns:
        combined: Dictionary containing combined analysis
    """
    # Initialize combined analysis
    combined = {
        'general': {
            'total_examples': 0,
            'total_databases': 0,
            'datasets': []
        },
        'difficulty': {},
        'datasets': {}
    }
    
    # Load and combine analyses
    for file_path in analysis_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            # Extract dataset name
            dataset_name = analysis.get('metadata', {}).get('dataset_name', os.path.basename(file_path))
            
            # Add to datasets list
            combined['general']['datasets'].append(dataset_name)
            
            # Update example and database counts
            combined['general']['total_examples'] += analysis['general']['total_examples']
            combined['general']['total_databases'] += analysis['general']['total_databases']
            
            # Combine difficulty distributions
            for diff, count in analysis.get('difficulty', {}).items():
                if diff in combined['difficulty']:
                    combined['difficulty'][diff] += count
                else:
                    combined['difficulty'][diff] = count
            
            # Store individual dataset analysis
            combined['datasets'][dataset_name] = analysis
            
        except Exception as e:
            print(f"Error combining analysis from {file_path}: {e}")
    
    # Save combined analysis
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"Combined analysis saved to {output_file}")
    
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Text2SQL processed data')
    parser.add_argument('--input', type=str, required=True, 
                        help='Directory containing processed data (instances and schemas subdirectories)')
    parser.add_argument('--output', type=str, default='analysis_results.json', 
                        help='Path to save analysis results')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualizations')
    parser.add_argument('--viz-dir', type=str, default='visualizations', 
                        help='Directory to save visualizations')
    parser.add_argument('--combine', nargs='+', 
                        help='List of analysis files to combine (optional)')
    parser.add_argument('--combined-output', type=str, default='combined_analysis.json',
                        help='Path to save combined analysis results')
    
    args = parser.parse_args()
    
    if args.combine:
        # Combine multiple analyses
        combine_analyses(args.combine, args.combined_output)
    else:
        # Run analysis on processed data
        analyze_text2sql_dataset(args.input, args.output, args.visualize, args.viz_dir)
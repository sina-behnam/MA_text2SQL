import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path
import os
import sqlparse

def created_difficulty(sql : str):

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


def load_data_processor(json_file_path):
    """
    Analyze Datasets on Data Proccossor output.
    
    Args:
        json_file_path: Path to the processed JSON file
        
    Returns:
        Dict containing metrics.
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
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
    
    # Extract metrics directly from the processed JSON
    for item in instances:
        # Skip items with errors
        if 'error' in item:
            continue
            
        # Database ID
        metrics['db_ids'].append(item.get('db_id', ''))
        
        # Difficulty (BIRD-specific)
        # ! check if the difficulty is defined or not in get
        if item['difficulty'] == 'unknown':
            metrics['difficulties'].append(created_difficulty(item['sql']))
        else:
            metrics['difficulties'].append(item.get('difficulty', 'unknown'))
        
        # Question features
        q_analysis = item.get('question_analysis', {})
        metrics['question_lengths'].append(q_analysis.get('char_length', 0))
        metrics['word_counts'].append(q_analysis.get('word_length', 0))
        
        # Entity information (Semantic parsing on Question)
        metrics['entity_presence'].append(q_analysis.get('has_entities', False))
        metrics['entity_counts'].append(len(q_analysis.get('entities', [])))
        
        for ent_type in q_analysis.get('entity_types', []):
            metrics['entity_types'].append(ent_type)

        ## Linguistic features (Semantic parsing on Question)
        metrics['superlative_presence'].append(q_analysis.get('has_superlatives', False))
        metrics['comparative_presence'].append(q_analysis.get('has_comparatives', False))
        metrics['negation_presence'].append(q_analysis.get('has_negation', False))
        
        # Schema overlaps (Semantic parsing on Question)
        metrics['table_overlaps'].append(q_analysis.get('table_overlap_count', 0))
        metrics['column_overlaps'].append(q_analysis.get('column_overlap_count', 0))
        metrics['table_lemma_overlaps'].append(q_analysis.get('table_overlap_lemma_count', 0))
        metrics['column_lemma_overlaps'].append(q_analysis.get('column_overlap_lemma_count', 0))
        
        # SQL features
        sql_analysis = item.get('sql_analysis', {})
        metrics['sql_lengths'].append(sql_analysis.get('char_length', 0))
        metrics['sql_table_occurance'].append(sql_analysis.get('tables_count', 0))
        # table_names = sql_analysis.get('table_names', [])
        # for table_name in table_names:
        #     table_names_occurrences[table_name] = table_names_occurrences.get(table_name, 0) + 1
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
        metrics['table_counts'].append(schema.get('table_count', 0))
        metrics['column_counts'].append(schema.get('column_count', 0))
        metrics['pk_counts'].append(schema.get('primary_key_count', 0))
        metrics['fk_counts'].append(schema.get('foreign_key_count', 0))

    return metrics

def analyzer(metrics):
    """
    Analyze the collected metrics and generate statistics.

    Args:
        metrics: Dictionary containing all collected metrics
    Returns:
        analysis: Dictionary containing analysis results
    """
    import numpy as np
    from collections import Counter

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
    
    # Calculate statistics (BIRD-specific)
    analysis['difficulty'] = dict(Counter(metrics['difficulties']))
    
    # DB statistics
    analysis['database_stats'] = {
        'unique_dbs': len(set(metrics['db_ids'])),
        'db_use_frequency': dict(Counter(metrics['db_ids']).most_common(10))  # Top 10 most used DBs
    }
    
    # Question features
    analysis['question_features'] = {
        'length_stats': {
            'char_mean': sum(metrics['question_lengths']) / len(metrics['question_lengths']) if metrics['question_lengths'] else 0,
            'char_std': np.std(metrics['question_lengths']) if metrics['question_lengths'] else 0,
            'word_mean': sum(metrics['word_counts']) / len(metrics['word_counts']) if metrics['word_counts'] else 0,
            'word_std': np.std(metrics['word_counts']) if metrics['word_counts'] else 0,
        },
        'entity_stats': {
            'has_entities_percent': sum(metrics['entity_presence']) / len(metrics['entity_presence']) * 100 if metrics['entity_presence'] else 0,
            'entity_count_avg': sum(metrics['entity_counts']) / len(metrics['entity_counts']) if metrics['entity_counts'] else 0,
            'entity_types': dict(Counter(metrics['entity_types']))
        },
        'linguistic_features': {
            'has_superlatives_percent': sum(metrics['superlative_presence']) / len(metrics['superlative_presence']) * 100 if metrics['superlative_presence'] else 0,
            'has_comparatives_percent': sum(metrics['comparative_presence']) / len(metrics['comparative_presence']) * 100 if metrics['comparative_presence'] else 0,
            'has_negation_percent': sum(metrics['negation_presence']) / len(metrics['negation_presence']) * 100 if metrics['negation_presence'] else 0
        },
        'schema_overlap_avg': {
            'table_overlap_count_avg': sum(metrics['table_overlaps']) / len(metrics['table_overlaps']) if metrics['table_overlaps'] else 0,
            'column_overlap_count_avg': sum(metrics['column_overlaps']) / len(metrics['column_overlaps']) if metrics['column_overlaps'] else 0,
            'table_overlap_lemma_count_avg': sum(metrics['table_lemma_overlaps']) / len(metrics['table_lemma_overlaps']) if metrics['table_lemma_overlaps'] else 0,
            'column_overlap_lemma_count_avg': sum(metrics['column_lemma_overlaps']) / len(metrics['column_lemma_overlaps']) if metrics['column_lemma_overlaps'] else 0,
        }
    }
    
    # SQL features
    analysis['sql_features'] = {
        'length_stats': {
            'mean': sum(metrics['sql_lengths']) / len(metrics['sql_lengths']) if metrics['sql_lengths'] else 0,
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
            'mean': sum(metrics['table_counts']) / len(metrics['table_counts']) if metrics['table_counts'] else 0,
        },
        'column_count_stats': {
            'min': min(metrics['column_counts']) if metrics['column_counts'] else 0,
            'max': max(metrics['column_counts']) if metrics['column_counts'] else 0,
            'mean': sum(metrics['column_counts']) / len(metrics['column_counts']) if metrics['column_counts'] else 0,
        },
        'keys_stats': {
            'avg_primary_keys': sum(metrics['pk_counts']) / len(metrics['pk_counts']) if metrics['pk_counts'] else 0,
            'avg_foreign_keys': sum(metrics['fk_counts']) / len(metrics['fk_counts']) if metrics['fk_counts'] else 0
        }
    }
     
    # Evidence features (BIRD-specific)
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
    
    # Difficulty distribution (for BIRD)
    if analysis['difficulty']:
        difficulties = analysis['difficulty']
        plt.figure(figsize=(10, 6))
        plt.bar(difficulties.keys(), difficulties.values())
        plt.title('Query Difficulty Distribution')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'difficulty_distribution.png'))
        plt.close()
    
    # Question length distribution
    question_lengths = analysis['question_features']['length_stats']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Character Length', 'Word Count'], 
            [question_lengths['char_mean'], question_lengths['word_mean']])
    plt.title('Average Question Length')
    plt.ylabel('Average Count')
    
    # SQL features
    sql_features = analysis['sql_features']
    plt.subplot(1, 2, 2)
    plt.bar(['SQL Length'], [sql_features['length_stats']['mean']])
    plt.title('Average SQL Length')
    plt.ylabel('Character Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_stats.png'))
    plt.close()
    
    # Join distribution
    join_stats = sql_features['join_stats']
    plt.figure(figsize=(10, 6))
    labels = ['No Joins', 'Has Joins', 'Multi Joins']
    values = [join_stats['no_join_percent'], 
              join_stats['has_join_percent'], 
              join_stats['multi_join_percent']]
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('SQL Join Distribution')
    plt.savefig(os.path.join(output_dir, 'join_distribution.png'))
    plt.close()
    
    # Schema features
    schema_features = analysis['schema_features']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    table_stats = schema_features['table_count_stats']
    plt.bar(['Minimum', 'Maximum', 'Average'], 
            [table_stats['min'], table_stats['max'], table_stats['mean']])
    plt.title('Table Count Statistics')
    plt.ylabel('Number of Tables')
    
    plt.subplot(1, 2, 2)
    key_stats = schema_features['keys_stats']
    plt.bar(['Primary Keys', 'Foreign Keys'], 
            [key_stats['avg_primary_keys'], key_stats['avg_foreign_keys']])
    plt.title('Average Keys per Database')
    plt.ylabel('Average Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'schema_stats.png'))
    plt.close()
    
    # Linguistic features
    linguistic = analysis['question_features']['linguistic_features']
    plt.figure(figsize=(10, 6))
    features = ['Superlatives', 'Comparatives', 'Negations']
    percentages = [linguistic['has_superlatives_percent'],
                  linguistic['has_comparatives_percent'],
                  linguistic['has_negation_percent']]
    plt.bar(features, percentages)
    plt.title('Linguistic Features in Questions')
    plt.ylabel('Percentage of Questions')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'linguistic_features.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def analyze_text2sql_dataset(json_file_path, visualize=True, output_dir='visualizations'):
    """
    Complete workflow to analyze a Text2SQL dataset from processed JSON
    
    Args:
        json_file_path: Path to the processed JSON file
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations
        
    Returns:
        analysis: Dictionary containing complete analysis
    """
    # Load and process the data
    metrics = load_data_processor(json_file_path)
    
    # Analyze the metrics
    analysis = analyzer(metrics)
    
    # Generate visualizations if requested
    if visualize:
        create_visualizations(analysis, output_dir)
    
    return analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Text2SQL dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to processed JSON file')
    parser.add_argument('--output', type=str, default='analysis_results.json', help='Path to save analysis results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--viz-dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Run analysis
    analysis = analyze_text2sql_dataset(args.input, args.visualize, args.viz_dir)
    
    # Save analysis results
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis completed and saved to {args.output}")
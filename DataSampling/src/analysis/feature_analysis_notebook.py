import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path

def analyze_bird_dataset(json_file_path):
    """
    Analyze BIRD dataset features based solely on metrics in the processed JSON file.
    
    Args:
        json_file_path: Path to the processed BIRD JSON file
        
    Returns:
        Dict containing analysis results
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    instances = data.get('instances', [])
    schemas = data.get('schemas', {})
    
    analysis = {
        'general': {
            'total_examples': len(instances),
            'total_databases': len(schemas),
        },
        'difficulty': {},
        'question_features': {},
        'sql_features': {},
        'schema_features': {},
        'linguistic_features': {},
        'evidence_features': {},
    }
    
    # Collect all metrics
    difficulties = []
    question_lengths = []
    sql_lengths = []
    entity_presence = []
    entity_counts = []
    entity_types = []
    superlative_presence = []
    comparative_presence = []
    negation_presence = []
    join_counts = []
    where_conditions = []
    subquery_counts = []
    aggregation_counts = []
    clause_types = []
    table_counts = []
    column_counts = []
    fk_counts = []
    pk_counts = []
    table_overlaps = []
    column_overlaps = []
    has_evidence = []
    evidence_lengths = []
    db_ids = []
    
    # Extract metrics directly from the processed JSON
    for item in instances:
        # Skip items with errors
        if 'error' in item:
            continue
            
        # Database ID
        db_ids.append(item.get('db_id', ''))
        
        # Difficulty
        difficulties.append(item.get('difficulty', 'unknown'))
        
        # Question features
        q_analysis = item.get('question_analysis', {})
        question_lengths.append(q_analysis.get('char_length', 0))
        
        # Entity information
        entity_presence.append(q_analysis.get('has_entities', False))
        entity_counts.append(len(q_analysis.get('entities', [])))
        
        for ent_type in q_analysis.get('entity_types', []):
            entity_types.append(ent_type)
        
        superlative_presence.append(q_analysis.get('has_superlatives', False))
        comparative_presence.append(q_analysis.get('has_comparatives', False))
        negation_presence.append(q_analysis.get('has_negation', False))
        
        # Schema overlaps
        table_overlaps.append(q_analysis.get('table_overlap_count', 0))
        column_overlaps.append(q_analysis.get('column_overlap_count', 0))
        
        # SQL features
        sql_analysis = item.get('sql_analysis', {})
        sql_lengths.append(sql_analysis.get('char_length', 0))
        join_counts.append(sql_analysis.get('join_count', 0))
        where_conditions.append(sql_analysis.get('where_conditions', 0))
        subquery_counts.append(sql_analysis.get('subquery_count', 0))
        aggregation_counts.append(sql_analysis.get('aggregation_function_count', 0))
        
        if 'clause_types' in sql_analysis:
            for clause in sql_analysis['clause_types']:
                clause_types.append(clause)
        
        # Evidence features
        evidence = item.get('evidence', '')
        has_evidence.append(bool(evidence))
        evidence_lengths.append(len(evidence))
    
    # Schema statistics
    for db_id, schema in schemas.items():
        table_counts.append(schema.get('table_count', 0))
        column_counts.append(schema.get('column_count', 0))
        pk_counts.append(schema.get('primary_key_count', 0))
        fk_counts.append(schema.get('foreign_key_count', 0))
    
    # Calculate statistics
    analysis['difficulty'] = dict(Counter(difficulties))
    
    # DB statistics
    analysis['database_stats'] = {
        'unique_dbs': len(set(db_ids)),
        'db_use_frequency': dict(Counter(db_ids).most_common(10))  # Top 10 most used DBs
    }
    
    # Question features
    analysis['question_features'] = {
        'length_stats': {
            'min': min(question_lengths) if question_lengths else 0,
            'max': max(question_lengths) if question_lengths else 0,
            'mean': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'distribution': pd.cut(question_lengths, bins=[0, 50, 100, 150, 200, float('inf')]).value_counts().to_dict() if question_lengths else {}
        },
        'entity_stats': {
            'has_entities_percent': sum(entity_presence) / len(entity_presence) * 100 if entity_presence else 0,
            'entity_count_avg': sum(entity_counts) / len(entity_counts) if entity_counts else 0,
            'entity_types': dict(Counter(entity_types))
        },
        'linguistic_features': {
            'has_superlatives_percent': sum(superlative_presence) / len(superlative_presence) * 100 if superlative_presence else 0,
            'has_comparatives_percent': sum(comparative_presence) / len(comparative_presence) * 100 if comparative_presence else 0,
            'has_negation_percent': sum(negation_presence) / len(negation_presence) * 100 if negation_presence else 0
        }
    }
    
    # SQL features
    analysis['sql_features'] = {
        'length_stats': {
            'min': min(sql_lengths) if sql_lengths else 0,
            'max': max(sql_lengths) if sql_lengths else 0,
            'mean': sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0,
        },
        'join_counts': dict(Counter(join_counts)),
        'where_conditions': dict(Counter(where_conditions)),
        'subquery_counts': dict(Counter(subquery_counts)),
        'aggregation_usage_percent': sum(1 for count in aggregation_counts if count > 0) / len(aggregation_counts) * 100 if aggregation_counts else 0,
        'clause_types': dict(Counter(clause_types)),
        'join_stats': {
            'no_join_percent': sum(1 for count in join_counts if count == 0) / len(join_counts) * 100 if join_counts else 0,
            'has_join_percent': sum(1 for count in join_counts if count > 0) / len(join_counts) * 100 if join_counts else 0,
            'multi_join_percent': sum(1 for count in join_counts if count > 1) / len(join_counts) * 100 if join_counts else 0
        }
    }
    
    # Schema features
    analysis['schema_features'] = {
        'table_count_stats': {
            'min': min(table_counts) if table_counts else 0,
            'max': max(table_counts) if table_counts else 0,
            'mean': sum(table_counts) / len(table_counts) if table_counts else 0,
        },
        'column_count_stats': {
            'min': min(column_counts) if column_counts else 0,
            'max': max(column_counts) if column_counts else 0,
            'mean': sum(column_counts) / len(column_counts) if column_counts else 0,
        },
        'keys_stats': {
            'avg_primary_keys': sum(pk_counts) / len(pk_counts) if pk_counts else 0,
            'avg_foreign_keys': sum(fk_counts) / len(fk_counts) if fk_counts else 0
        }
    }
    
    # Schema overlap
    analysis['schema_overlap'] = {
        'table_overlap_stats': {
            'min': min(table_overlaps) if table_overlaps else 0,
            'max': max(table_overlaps) if table_overlaps else 0,
            'mean': sum(table_overlaps) / len(table_overlaps) if table_overlaps else 0,
            'distribution': dict(Counter(table_overlaps))
        },
        'column_overlap_stats': {
            'min': min(column_overlaps) if column_overlaps else 0,
            'max': max(column_overlaps) if column_overlaps else 0,
            'mean': sum(column_overlaps) / len(column_overlaps) if column_overlaps else 0,
            'distribution': dict(Counter(column_overlaps))
        }
    }
    
    # Evidence features (BIRD-specific)
    analysis['evidence_features'] = {
        'has_evidence_percent': sum(has_evidence) / len(has_evidence) * 100 if has_evidence else 0,
        'length_stats': {
            'min': min(evidence_lengths) if evidence_lengths else 0,
            'max': max(evidence_lengths) if evidence_lengths else 0,
            'mean': sum(evidence_lengths) / len(evidence_lengths) if evidence_lengths else 0,
        }
    }
    
    return analysis


def visualize_bird_dataset(analysis, output_dir='outputs'):
    """
    Create visualizations for BIRD dataset analysis.
    
    Args:
        analysis: Dictionary containing BIRD analysis results
        output_dir: Directory to save visualization files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    
    # 1. Difficulty Distribution
    plt.figure(figsize=(10, 6))
    difficulties = analysis['difficulty']
    labels = list(difficulties.keys())
    values = list(difficulties.values())
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
    plt.bar(labels, values, color=colors)
    
    plt.title('BIRD Dataset: Difficulty Distribution', fontsize=15)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Number of Examples')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'bird_difficulty_distribution.png', dpi=300)
    
    # 2. SQL Complexity Features
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BIRD Dataset: SQL Complexity Analysis', fontsize=16)
    
    # 2.1 JOIN Count Distribution
    join_counts = analysis['sql_features']['join_counts']
    # Ensure all keys are strings for consistent handling
    join_counts_str = {str(k): v for k, v in join_counts.items()}
    join_labels = sorted([int(k) for k in join_counts_str.keys()])
    join_values = [join_counts_str.get(str(k), 0) for k in join_labels]
    
    ax[0, 0].bar(join_labels, join_values, color='skyblue')
    ax[0, 0].set_title('JOIN Count Distribution')
    ax[0, 0].set_xlabel('Number of JOINs')
    ax[0, 0].set_ylabel('Count')
    ax[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2.2 WHERE Conditions Distribution
    where_conditions = analysis['sql_features']['where_conditions']
    # Ensure all keys are strings for consistent handling
    where_conditions_str = {str(k): v for k, v in where_conditions.items()}
    where_labels = sorted([int(k) for k in where_conditions_str.keys()])
    where_values = [where_conditions_str.get(str(k), 0) for k in where_labels]
    
    ax[0, 1].bar(where_labels, where_values, color='lightgreen')
    ax[0, 1].set_title('WHERE Conditions Distribution')
    ax[0, 1].set_xlabel('Number of WHERE Conditions')
    ax[0, 1].set_ylabel('Count')
    ax[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2.3 SQL Clause Types
    clause_types = analysis['sql_features']['clause_types']
    clause_labels = list(clause_types.keys())
    clause_values = list(clause_types.values())
    
    ax[1, 0].bar(range(len(clause_labels)), clause_values, color='salmon')
    ax[1, 0].set_title('SQL Clause Types')
    ax[1, 0].set_xlabel('Clause Type')
    ax[1, 0].set_ylabel('Count')
    ax[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    ax[1, 0].set_xticks(range(len(clause_labels)))
    ax[1, 0].set_xticklabels(clause_labels, rotation=45, ha='right')
    
    # 2.4 Subquery Distribution
    subquery_counts = analysis['sql_features']['subquery_counts']
    # Ensure all keys are strings for consistent handling
    subquery_counts_str = {str(k): v for k, v in subquery_counts.items()}
    subquery_labels = sorted([int(k) for k in subquery_counts_str.keys()])
    subquery_values = [subquery_counts_str.get(str(k), 0) for k in subquery_labels]
    
    ax[1, 1].bar(subquery_labels, subquery_values, color='mediumpurple')
    ax[1, 1].set_title('Subquery Distribution')
    ax[1, 1].set_xlabel('Number of Subqueries')
    ax[1, 1].set_ylabel('Count')
    ax[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / 'bird_sql_complexity.png', dpi=300)
    
    # 3. Question Characteristics
    plt.figure(figsize=(14, 7))
    
    # Linguistic Features
    ling_features = {
        'Entities': analysis['question_features']['entity_stats']['has_entities_percent'],
        'Superlatives': analysis['question_features']['linguistic_features']['has_superlatives_percent'],
        'Comparatives': analysis['question_features']['linguistic_features']['has_comparatives_percent'],
        'Negation': analysis['question_features']['linguistic_features']['has_negation_percent']
    }
    
    feature_labels = list(ling_features.keys())
    feature_values = list(ling_features.values())
    
    plt.bar(feature_labels, feature_values, color=plt.cm.tab10(np.linspace(0, 1, len(feature_labels))))
    plt.title('BIRD Dataset: Linguistic Features in Questions', fontsize=15)
    plt.xlabel('Feature')
    plt.ylabel('Percentage of Questions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(feature_values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'bird_linguistic_features.png', dpi=300)
    
    # 4. Schema Complexity and Overlap
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('BIRD Dataset: Schema Complexity and Question-Schema Overlap', fontsize=16)
    
    # 4.1 Schema Statistics
    schema_stats = [
        analysis['schema_features']['table_count_stats']['mean'],
        analysis['schema_features']['column_count_stats']['mean'] / 5,  # Scaled down for visualization
        analysis['schema_features']['keys_stats']['avg_primary_keys'],
        analysis['schema_features']['keys_stats']['avg_foreign_keys']
    ]
    
    schema_labels = ['Avg Tables', 'Avg Columns (÷5)', 'Avg Primary Keys', 'Avg Foreign Keys']
    
    ax[0].bar(schema_labels, schema_stats, color=plt.cm.Paired(np.linspace(0, 1, len(schema_labels))))
    ax[0].set_title('Schema Statistics')
    ax[0].set_ylabel('Average Count')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(schema_stats):
        ax[0].text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    # 4.2 Question-Schema Overlap
    overlap_stats = [
        analysis['schema_overlap']['table_overlap_stats']['mean'],
        analysis['schema_overlap']['column_overlap_stats']['mean']
    ]
    
    overlap_labels = ['Avg Table Overlap', 'Avg Column Overlap']
    
    ax[1].bar(overlap_labels, overlap_stats, color=['#1f77b4', '#ff7f0e'])
    ax[1].set_title('Question-Schema Overlap')
    ax[1].set_ylabel('Average Count')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(overlap_stats):
        ax[1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / 'bird_schema_analysis.png', dpi=300)
    
    # 5. Evidence Feature (BIRD-specific)
    plt.figure(figsize=(10, 6))
    
    # Evidence presence pie chart
    has_evidence_pct = analysis['evidence_features']['has_evidence_percent']
    no_evidence_pct = 100 - has_evidence_pct
    
    plt.pie([has_evidence_pct, no_evidence_pct], 
            labels=['With Evidence', 'No Evidence'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66b3ff', '#ffcc99'],
            explode=(0.1, 0))
    
    plt.title('BIRD Dataset: Evidence Presence', fontsize=15)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.tight_layout()
    plt.savefig(output_path / 'bird_evidence_presence.png', dpi=300)
    
    # 6. Additional: Length Distributions
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('BIRD Dataset: Length Distributions', fontsize=16)
    
    # 6.1 Question Length Distribution
    question_lengths = [
        q for q in pd.cut(
            [analysis['question_features']['length_stats']['min']] + 
            list(range(0, int(analysis['question_features']['length_stats']['max']), 25)) + 
            [analysis['question_features']['length_stats']['max']],
            bins=10
        ).categories
    ]
    
    # Create histogram-like data
    question_bins = analysis['question_features']['length_stats']['distribution']
    question_bin_labels = [str(q) for q in question_bins.keys()]
    question_bin_values = list(question_bins.values())
    
    ax[0].bar(question_bin_labels, question_bin_values, color='#5975a4')
    ax[0].set_title('Question Length Distribution')
    ax[0].set_xlabel('Character Length Range')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels(question_bin_labels, rotation=45, ha='right')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 6.2 SQL Length Distribution
    if analysis['sql_features']['length_stats']['max'] > 0:
        sql_edges = np.linspace(0, analysis['sql_features']['length_stats']['max'], 10)
        sql_hist, sql_bins = np.histogram([analysis['sql_features']['length_stats']['mean']], bins=sql_edges)
        sql_bin_labels = [f"{int(sql_bins[i])}-{int(sql_bins[i+1])}" for i in range(len(sql_bins)-1)]
        
        ax[1].bar(sql_bin_labels, sql_hist, color='#a4596d')
        ax[1].set_title(f"SQL Length Distribution (Mean: {analysis['sql_features']['length_stats']['mean']:.1f})")
        ax[1].set_xlabel('Character Length Range')
        ax[1].set_ylabel('Count')
        ax[1].set_xticklabels(sql_bin_labels, rotation=45, ha='right')
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax[1].text(0.5, 0.5, "No SQL length data available", ha='center', va='center')
        ax[1].set_title("SQL Length Distribution")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / 'bird_length_distributions.png', dpi=300)
    
    # Close all figures
    plt.close('all')
    
    print(f"BIRD dataset visualizations saved to {output_path}")


def analyze_spider_dataset(json_file_path):
    """
    Analyze Spider dataset features based solely on metrics in the processed JSON file.
    
    Args:
        json_file_path: Path to the processed Spider JSON file
        
    Returns:
        Dict containing analysis results
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    instances = data.get('instances', [])
    schemas = data.get('schemas', {})
    
    analysis = {
        'general': {
            'total_examples': len(instances),
            'total_databases': len(schemas),
        },
        'derived_difficulty': {},
        'question_features': {},
        'sql_features': {},
        'schema_features': {},
        'linguistic_features': {},
    }
    
    # Collect all metrics
    question_lengths = []
    sql_lengths = []
    entity_presence = []
    entity_counts = []
    entity_types = []
    superlative_presence = []
    comparative_presence = []
    negation_presence = []
    join_counts = []
    where_conditions = []
    subquery_counts = []
    aggregation_counts = []
    clause_types = []
    table_counts = []
    column_counts = []
    fk_counts = []
    pk_counts = []
    table_overlaps = []
    column_overlaps = []
    db_ids = []
    
    # Derive difficulty categories
    derived_difficulties = []
    
    # Extract metrics directly from the processed JSON
    for item in instances:
        # Skip items with errors
        if 'error' in item:
            continue
            
        # Database ID
        db_ids.append(item.get('db_id', ''))
        
        # Question features
        q_analysis = item.get('question_analysis', {})
        question_lengths.append(q_analysis.get('char_length', 0))
        
        # Entity information
        entity_presence.append(q_analysis.get('has_entities', False))
        entity_counts.append(len(q_analysis.get('entities', [])))
        
        for ent_type in q_analysis.get('entity_types', []):
            entity_types.append(ent_type)
        
        superlative_presence.append(q_analysis.get('has_superlatives', False))
        comparative_presence.append(q_analysis.get('has_comparatives', False))
        negation_presence.append(q_analysis.get('has_negation', False))
        
        # Schema overlaps
        table_overlaps.append(q_analysis.get('table_overlap_count', 0))
        column_overlaps.append(q_analysis.get('column_overlap_count', 0))
        
        # SQL features
        sql_analysis = item.get('sql_analysis', {})
        sql_lengths.append(sql_analysis.get('char_length', 0))
        join_counts.append(sql_analysis.get('join_count', 0))
        where_conditions.append(sql_analysis.get('where_conditions', 0))
        subquery_counts.append(sql_analysis.get('subquery_count', 0))
        aggregation_counts.append(sql_analysis.get('aggregation_function_count', 0))
        
        if 'clause_types' in sql_analysis:
            for clause in sql_analysis['clause_types']:
                clause_types.append(clause)
        
        # Derive difficulty level based on SQL complexity
        join_ct = sql_analysis.get('join_count', 0)
        where_ct = sql_analysis.get('where_conditions', 0)
        subquery_ct = sql_analysis.get('subquery_count', 0)
        clause_ct = sql_analysis.get('clauses_count', 0)
        agg_ct = sql_analysis.get('aggregation_function_count', 0)
        
        sql_complexity = (
            join_ct * 2 +
            where_ct +
            subquery_ct * 3 +
            clause_ct +
            agg_ct
        )
        
        if sql_complexity <= 1:
            derived_difficulties.append('simple')
        elif sql_complexity <= 4:
            derived_difficulties.append('medium')
        else:
            derived_difficulties.append('complex')
    
    # Schema statistics
    for db_id, schema in schemas.items():
        table_counts.append(schema.get('table_count', 0))
        column_counts.append(schema.get('column_count', 0))
        pk_counts.append(schema.get('primary_key_count', 0))
        fk_counts.append(schema.get('foreign_key_count', 0))
    
    # Calculate statistics
    analysis['derived_difficulty'] = dict(Counter(derived_difficulties))
    
    # DB statistics
    analysis['database_stats'] = {
        'unique_dbs': len(set(db_ids)),
        'db_use_frequency': dict(Counter(db_ids).most_common(10))  # Top 10 most used DBs
    }
    
    # Question features
    analysis['question_features'] = {
        'length_stats': {
            'min': min(question_lengths) if question_lengths else 0,
            'max': max(question_lengths) if question_lengths else 0,
            'mean': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'distribution': pd.cut(question_lengths, bins=[0, 25, 50, 75, 100, float('inf')]).value_counts().to_dict() if question_lengths else {}
        },
        'entity_stats': {
            'has_entities_percent': sum(entity_presence) / len(entity_presence) * 100 if entity_presence else 0,
            'entity_count_avg': sum(entity_counts) / len(entity_counts) if entity_counts else 0,
            'entity_types': dict(Counter(entity_types))
        },
        'linguistic_features': {
            'has_superlatives_percent': sum(superlative_presence) / len(superlative_presence) * 100 if superlative_presence else 0,
            'has_comparatives_percent': sum(comparative_presence) / len(comparative_presence) * 100 if comparative_presence else 0,
            'has_negation_percent': sum(negation_presence) / len(negation_presence) * 100 if negation_presence else 0
        }
    }
    
    # SQL features
    analysis['sql_features'] = {
        'length_stats': {
            'min': min(sql_lengths) if sql_lengths else 0,
            'max': max(sql_lengths) if sql_lengths else 0,
            'mean': sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0,
        },
        'join_counts': dict(Counter(join_counts)),
        'where_conditions': dict(Counter(where_conditions)),
        'subquery_counts': dict(Counter(subquery_counts)),
        'aggregation_usage_percent': sum(1 for count in aggregation_counts if count > 0) / len(aggregation_counts) * 100 if aggregation_counts else 0,
        'clause_types': dict(Counter(clause_types)),
        'join_stats': {
            'no_join_percent': sum(1 for count in join_counts if count == 0) / len(join_counts) * 100 if join_counts else 0,
            'has_join_percent': sum(1 for count in join_counts if count > 0) / len(join_counts) * 100 if join_counts else 0,
            'multi_join_percent': sum(1 for count in join_counts if count > 1) / len(join_counts) * 100 if join_counts else 0
        }
    }
    
    # Schema features
    analysis['schema_features'] = {
        'table_count_stats': {
            'min': min(table_counts) if table_counts else 0,
            'max': max(table_counts) if table_counts else 0,
            'mean': sum(table_counts) / len(table_counts) if table_counts else 0,
        },
        'column_count_stats': {
            'min': min(column_counts) if column_counts else 0,
            'max': max(column_counts) if column_counts else 0,
            'mean': sum(column_counts) / len(column_counts) if column_counts else 0,
        },
        'keys_stats': {
            'avg_primary_keys': sum(pk_counts) / len(pk_counts) if pk_counts else 0,
            'avg_foreign_keys': sum(fk_counts) / len(fk_counts) if fk_counts else 0
        }
    }
    
    # Schema overlap
    analysis['schema_overlap'] = {
        'table_overlap_stats': {
            'min': min(table_overlaps) if table_overlaps else 0,
            'max': max(table_overlaps) if table_overlaps else 0,
            'mean': sum(table_overlaps) / len(table_overlaps) if table_overlaps else 0,
            'distribution': dict(Counter(table_overlaps))
        },
        'column_overlap_stats': {
            'min': min(column_overlaps) if column_overlaps else 0,
            'max': max(column_overlaps) if column_overlaps else 0,
            'mean': sum(column_overlaps) / len(column_overlaps) if column_overlaps else 0,
            'distribution': dict(Counter(column_overlaps))
        }
    }
    
    return analysis


def visualize_spider_dataset(analysis, output_dir='outputs'):
    """
    Create visualizations for Spider dataset analysis.
    
    Args:
        analysis: Dictionary containing Spider analysis results
        output_dir: Directory to save visualization files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    
    # 1. Derived Difficulty Distribution
    plt.figure(figsize=(10, 6))
    difficulties = analysis['derived_difficulty']
    labels = list(difficulties.keys())
    values = list(difficulties.values())
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
    plt.bar(labels, values, color=colors)
    
    plt.title('Spider Dataset: Derived Difficulty Distribution', fontsize=15)
    plt.xlabel('Difficulty Level')
    plt.ylabel('Number of Examples')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'spider_difficulty_distribution.png', dpi=300)
    
    # 2. SQL Complexity Features
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spider Dataset: SQL Complexity Analysis', fontsize=16)
    
    # 2.1 JOIN Count Distribution
    join_counts = analysis['sql_features']['join_counts']
    # Ensure all keys are strings for consistent handling
    join_counts_str = {str(k): v for k, v in join_counts.items()}
    join_labels = sorted([int(k) for k in join_counts_str.keys()])
    join_values = [join_counts_str.get(str(k), 0) for k in join_labels]
    
    ax[0, 0].bar(join_labels, join_values, color='skyblue')
    ax[0, 0].set_title('JOIN Count Distribution')
    ax[0, 0].set_xlabel('Number of JOINs')
    ax[0, 0].set_ylabel('Count')
    ax[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2.2 WHERE Conditions Distribution
    where_conditions = analysis['sql_features']['where_conditions']
    # Ensure all keys are strings for consistent handling
    where_conditions_str = {str(k): v for k, v in where_conditions.items()}
    where_labels = sorted([int(k) for k in where_conditions_str.keys()])
    where_values = [where_conditions_str.get(str(k), 0) for k in where_labels]
    
    ax[0, 1].bar(where_labels, where_values, color='lightgreen')
    ax[0, 1].set_title('WHERE Conditions Distribution')
    ax[0, 1].set_xlabel('Number of WHERE Conditions')
    ax[0, 1].set_ylabel('Count')
    ax[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2.3 SQL Clause Types
    clause_types = analysis['sql_features']['clause_types']
    clause_labels = list(clause_types.keys())
    clause_values = list(clause_types.values())
    
    ax[1, 0].bar(range(len(clause_labels)), clause_values, color='salmon')
    ax[1, 0].set_title('SQL Clause Types')
    ax[1, 0].set_xlabel('Clause Type')
    ax[1, 0].set_ylabel('Count')
    ax[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    ax[1, 0].set_xticks(range(len(clause_labels)))
    ax[1, 0].set_xticklabels(clause_labels, rotation=45, ha='right')
    
    # 2.4 Subquery Distribution
    subquery_counts = analysis['sql_features']['subquery_counts']
    # Ensure all keys are strings for consistent handling
    subquery_counts_str = {str(k): v for k, v in subquery_counts.items()}
    subquery_labels = sorted([int(k) for k in subquery_counts_str.keys()])
    subquery_values = [subquery_counts_str.get(str(k), 0) for k in subquery_labels]
    
    ax[1, 1].bar(subquery_labels, subquery_values, color='mediumpurple')
    ax[1, 1].set_title('Subquery Distribution')
    ax[1, 1].set_xlabel('Number of Subqueries')
    ax[1, 1].set_ylabel('Count')
    ax[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / 'spider_sql_complexity.png', dpi=300)
    
    # 3. Question Characteristics
    plt.figure(figsize=(14, 7))
    
    # Linguistic Features
    ling_features = {
        'Entities': analysis['question_features']['entity_stats']['has_entities_percent'],
        'Superlatives': analysis['question_features']['linguistic_features']['has_superlatives_percent'],
        'Comparatives': analysis['question_features']['linguistic_features']['has_comparatives_percent'],
        'Negation': analysis['question_features']['linguistic_features']['has_negation_percent']
    }
    
    feature_labels = list(ling_features.keys())
    feature_values = list(ling_features.values())
    
    plt.bar(feature_labels, feature_values, color=plt.cm.tab10(np.linspace(0, 1, len(feature_labels))))
    plt.title('Spider Dataset: Linguistic Features in Questions', fontsize=15)
    plt.xlabel('Feature')
    plt.ylabel('Percentage of Questions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(feature_values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'spider_linguistic_features.png', dpi=300)
    
    # 4. Schema Complexity and Overlap
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Spider Dataset: Schema Complexity and Question-Schema Overlap', fontsize=16)
    
    # 4.1 Schema Statistics
    schema_stats = [
        analysis['schema_features']['table_count_stats']['mean'],
        analysis['schema_features']['column_count_stats']['mean'] / 5,  # Scaled down for visualization
        analysis['schema_features']['keys_stats']['avg_primary_keys'],
        analysis['schema_features']['keys_stats']['avg_foreign_keys']
    ]
    
    schema_labels = ['Avg Tables', 'Avg Columns (÷5)', 'Avg Primary Keys', 'Avg Foreign Keys']
    
    ax[0].bar(schema_labels, schema_stats, color=plt.cm.Paired(np.linspace(0, 1, len(schema_labels))))
    ax[0].set_title('Schema Statistics')
    ax[0].set_ylabel('Average Count')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(schema_stats):
        ax[0].text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    # 4.2 Question-Schema Overlap
    overlap_stats = [
        analysis['schema_overlap']['table_overlap_stats']['mean'],
        analysis['schema_overlap']['column_overlap_stats']['mean']
    ]
    
    overlap_labels = ['Avg Table Overlap', 'Avg Column Overlap']
    
    ax[1].bar(overlap_labels, overlap_stats, color=['#1f77b4', '#ff7f0e'])
    ax[1].set_title('Question-Schema Overlap')
    ax[1].set_ylabel('Average Count')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(overlap_stats):
        ax[1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / 'spider_schema_analysis.png', dpi=300)
    
    # 5. SQL Complexity Distribution
    plt.figure(figsize=(12, 7))
    
    # Extract data needed for complexity calculation
    join_counts_list = []
    where_conditions_list = []
    subquery_counts_list = []
    aggregation_counts_list = []
    
    # Recreate these from the analysis stats
    for j_key, j_count in analysis['sql_features']['join_counts'].items():
        join_counts_list.extend([int(j_key)] * j_count)
    
    for w_key, w_count in analysis['sql_features']['where_conditions'].items():
        where_conditions_list.extend([int(w_key)] * w_count)
    
    for s_key, s_count in analysis['sql_features']['subquery_counts'].items():
        subquery_counts_list.extend([int(s_key)] * s_count)
    
    # Create a simplified approximation for aggregation counts based on percentage
    total_queries = analysis['general']['total_examples']
    agg_percentage = analysis['sql_features']['aggregation_usage_percent']
    agg_count = int((agg_percentage / 100) * total_queries)
    
    # Simplify by assuming queries either have 0 or 1 aggregation functions
    aggregation_counts_list = [1] * agg_count + [0] * (total_queries - agg_count)
    
    # Ensure all lists have the same length by truncating to the shortest one
    min_length = min(len(join_counts_list), len(where_conditions_list), 
                     len(subquery_counts_list), len(aggregation_counts_list))
    
    join_counts_list = join_counts_list[:min_length]
    where_conditions_list = where_conditions_list[:min_length]
    subquery_counts_list = subquery_counts_list[:min_length]
    aggregation_counts_list = aggregation_counts_list[:min_length]
    
    # Combine JOIN, WHERE, and Subquery counts for complexity histogram
    complexity_scores = []
    
    for j, w, s, a in zip(join_counts_list, where_conditions_list, 
                         subquery_counts_list, aggregation_counts_list):
        score = j * 2 + w + s * 3 + (1 if a > 0 else 0)
        complexity_scores.append(score)
    
    plt.hist(complexity_scores, bins=range(0, max(complexity_scores) + 2), alpha=0.7, color='teal', edgecolor='black')
    plt.title('Spider Dataset: SQL Complexity Score Distribution', fontsize=15)
    plt.xlabel('Complexity Score')
    plt.ylabel('Number of Queries')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(0, max(complexity_scores) + 2, 2))
    
    # Add complexity thresholds
    plt.axvline(x=1.5, color='green', linestyle='--', label='Simple-Medium Threshold')
    plt.axvline(x=4.5, color='red', linestyle='--', label='Medium-Complex Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'spider_complexity_distribution.png', dpi=300)
    
    # 6. Additional: Length Distributions
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Spider Dataset: Length Distributions', fontsize=16)
    
    # 6.1 Question Length Distribution
    question_bins = analysis['question_features']['length_stats']['distribution']
    question_bin_labels = [str(q) for q in question_bins.keys()]
    question_bin_values = list(question_bins.values())
    
    ax[0].bar(question_bin_labels, question_bin_values, color='#5975a4')
    ax[0].set_title('Question Length Distribution')
    ax[0].set_xlabel('Character Length Range')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels(question_bin_labels, rotation=45, ha='right')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 6.2 SQL Length Distribution
    if analysis['sql_features']['length_stats']['max'] > 0:
        sql_edges = np.linspace(0, analysis['sql_features']['length_stats']['max'], 10)
        sql_hist, sql_bins = np.histogram([analysis['sql_features']['length_stats']['mean']], bins=sql_edges)
        sql_bin_labels = [f"{int(sql_bins[i])}-{int(sql_bins[i+1])}" for i in range(len(sql_bins)-1)]
        
        ax[1].bar(sql_bin_labels, sql_hist, color='#a4596d')
        ax[1].set_title(f"SQL Length Distribution (Mean: {analysis['sql_features']['length_stats']['mean']:.1f})")
        ax[1].set_xlabel('Character Length Range')
        ax[1].set_ylabel('Count')
        ax[1].set_xticklabels(sql_bin_labels, rotation=45, ha='right')
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax[1].text(0.5, 0.5, "No SQL length data available", ha='center', va='center')
        ax[1].set_title("SQL Length Distribution")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / 'spider_length_distributions.png', dpi=300)
    
    # Close all figures
    plt.close('all')
    
    print(f"Spider dataset visualizations saved to {output_path}")


def print_dataset_statistics(analysis, dataset_name):
    """
    Print key statistics from dataset analysis in a readable format.
    
    Args:
        analysis: Dictionary containing dataset analysis
        dataset_name: Name of the dataset (BIRD or Spider)
    """
    print(f"\n{'='*20} {dataset_name.upper()} DATASET STATISTICS {'='*20}")
    
    # General information
    print("\n--- GENERAL INFORMATION ---")
    print(f"Total examples: {analysis['general']['total_examples']}")
    print(f"Total databases: {analysis['general']['total_databases']}")
    print(f"Unique databases: {analysis.get('database_stats', {}).get('unique_dbs', 'N/A')}")
    
    # Difficulty distribution
    if dataset_name.lower() == 'bird':
        diff_key = 'difficulty'
        diff_label = 'Difficulty'
    else:
        diff_key = 'derived_difficulty'
        diff_label = 'Derived Difficulty'
    
    print(f"\n--- {diff_label.upper()} DISTRIBUTION ---")
    for diff, count in analysis[diff_key].items():
        print(f"{diff}: {count} ({count/analysis['general']['total_examples']*100:.1f}%)")
    
    # Question features
    print("\n--- QUESTION FEATURES ---")
    q_stats = analysis['question_features']['length_stats']
    print(f"Question length (chars): min={q_stats['min']}, max={q_stats['max']}, avg={q_stats['mean']:.1f}")
    
    ling = analysis['question_features'].get('linguistic_features', {})
    print(f"Questions with entities: {analysis['question_features']['entity_stats']['has_entities_percent']:.1f}%")
    print(f"Questions with superlatives: {ling.get('has_superlatives_percent', 0):.1f}%")
    print(f"Questions with comparatives: {ling.get('has_comparatives_percent', 0):.1f}%")
    print(f"Questions with negation: {ling.get('has_negation_percent', 0):.1f}%")
    
    # SQL features
    print("\n--- SQL FEATURES ---")
    sql_stats = analysis['sql_features']['length_stats']
    print(f"SQL length (chars): min={sql_stats['min']}, max={sql_stats['max']}, avg={sql_stats['mean']:.1f}")
    
    join_stats = analysis['sql_features']['join_stats']
    print(f"Queries with no joins: {join_stats['no_join_percent']:.1f}%")
    print(f"Queries with joins: {join_stats['has_join_percent']:.1f}%")
    print(f"Queries with multiple joins: {join_stats['multi_join_percent']:.1f}%")
    
    print(f"Queries with aggregation: {analysis['sql_features']['aggregation_usage_percent']:.1f}%")
    
    # Schema features
    print("\n--- SCHEMA FEATURES ---")
    print(f"Tables per schema: avg={analysis['schema_features']['table_count_stats']['mean']:.1f}")
    print(f"Columns per schema: avg={analysis['schema_features']['column_count_stats']['mean']:.1f}")
    print(f"Primary keys per schema: avg={analysis['schema_features']['keys_stats']['avg_primary_keys']:.1f}")
    print(f"Foreign keys per schema: avg={analysis['schema_features']['keys_stats']['avg_foreign_keys']:.1f}")
    
    # Question-Schema overlap
    print("\n--- QUESTION-SCHEMA OVERLAP ---")
    print(f"Average table name overlap: {analysis['schema_overlap']['table_overlap_stats']['mean']:.2f}")
    print(f"Average column name overlap: {analysis['schema_overlap']['column_overlap_stats']['mean']:.2f}")
    
    # Evidence features (BIRD specific)
    if dataset_name.lower() == 'bird' and 'evidence_features' in analysis:
        print("\n--- EVIDENCE FEATURES (BIRD) ---")
        print(f"Questions with evidence: {analysis['evidence_features']['has_evidence_percent']:.1f}%")
        ev_stats = analysis['evidence_features']['length_stats']
        print(f"Evidence length (chars): min={ev_stats['min']}, max={ev_stats['max']}, avg={ev_stats['mean']:.1f}")
    
    print(f"\n{'='*60}")



# Example usage
if __name__ == "__main__":
    # Analyze and visualize BIRD dataset
    bird_analysis = analyze_bird_dataset('/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/output/bird_process.json')
    visualize_bird_dataset(bird_analysis, 'outputs/bird_analysis')
    print_dataset_statistics(bird_analysis, "BIRD")
    
    # Analyze and visualize Spider dataset
    spider_analysis = analyze_spider_dataset('/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/output/spider_process.json')
    visualize_spider_dataset(spider_analysis, 'outputs/spider_analysis')
    print_dataset_statistics(spider_analysis, "Spider")

    print("\nAnalysis complete! Visualizations and statistics have been generated.")
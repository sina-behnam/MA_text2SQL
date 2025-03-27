"""
BIRD Dataset Statistical Analysis

This script analyzes the BIRD (Benchmarking Instruction-Resolution for Databases) 
dataset to extract various statistics about tables, queries, and their properties.
"""

import os
import json
import re
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

# Paths - adjust these based on your directory structure
DATA_DIR = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/Data/data/bird"
DEV_JSON_PATH = os.path.join(DATA_DIR, "dev.json")
DATABASES_DIR = os.path.join(DATA_DIR, "dev_databases")


def load_dev_data(path: str = "dev_2_examples.json"):
    """Load the development data from JSON file."""
    try:
        print(f"Loading development data from: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}. Using fallback example data.")
        # If the main dev.json isn't found, try the example file
        with open("dev_2_examples.json", 'r') as f:
            return json.load(f)


def get_database_paths():
    """Get paths for all SQLite databases."""
    db_paths = []
    try:
        for db_dir in os.listdir(DATABASES_DIR):
            db_path = os.path.join(DATABASES_DIR, db_dir, f"{db_dir}.sqlite")
            if os.path.exists(db_path):
                db_paths.append(db_path)
    except FileNotFoundError:
        print(f"Database directory not found: {DATABASES_DIR}")
    return db_paths


def analyze_database_schema(db_path: str) -> Dict:
    """Analyze the schema of a database to extract table and attribute statistics."""
    results = {
        'tables': [],
        'attributes_per_table': {},
        'values_per_attribute': {},
        'numerical_attrs': {},
        'categorical_attrs': {}
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        results['tables'] = tables
        
        for table in tables:
            # Get column information
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            
            numerical_attrs = []
            categorical_attrs = []
            
            for col in columns:
                col_name = col[1]
                col_type = col[2].lower()
                
                # Determine if column is numerical or categorical
                if col_type in ('integer', 'real', 'float', 'double', 'numeric', 'decimal'):
                    numerical_attrs.append(col_name)
                else:
                    categorical_attrs.append(col_name)
                
                # Count distinct values
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT `{col_name}`) FROM `{table}`;")
                    distinct_count = cursor.fetchone()[0]
                    if table not in results['values_per_attribute']:
                        results['values_per_attribute'][table] = {}
                    results['values_per_attribute'][table][col_name] = distinct_count
                except sqlite3.OperationalError:
                    # Handle case where column might have special characters
                    pass
            
            results['attributes_per_table'][table] = len(columns)
            results['numerical_attrs'][table] = numerical_attrs
            results['categorical_attrs'][table] = categorical_attrs
        
        conn.close()
    
    except sqlite3.Error as e:
        print(f"SQLite error when analyzing {db_path}: {e}")
    
    return results


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from a SQL query."""
    tables = set()
    
    # Find tables in FROM clause
    from_matches = re.findall(r'FROM\s+`?([a-zA-Z0-9_]+)`?', sql, re.IGNORECASE)
    if from_matches:
        tables.update(from_matches)
    
    # Find tables in JOIN clauses
    join_matches = re.findall(r'JOIN\s+`?([a-zA-Z0-9_]+)`?', sql, re.IGNORECASE)
    if join_matches:
        tables.update(join_matches)
    
    return tables


def analyze_query_type(sql: str) -> Dict[str, bool]:
    """Analyze the type of SQL query."""
    sql_upper = sql.upper()
    
    query_types = {
        'simple_select': 'SELECT' in sql_upper and 'JOIN' not in sql_upper and 'GROUP BY' not in sql_upper,
        'join': 'JOIN' in sql_upper,
        'group_by': 'GROUP BY' in sql_upper,
        'having': 'HAVING' in sql_upper,
        'nested_query': sql_upper.count('SELECT') > 1 or ('IN (' in sql_upper and 'SELECT' in sql_upper.split('IN (')[1]),
        'order_by': 'ORDER BY' in sql_upper,
        'limit': 'LIMIT' in sql_upper,
        'distinct': 'DISTINCT' in sql_upper,
        'case_when': 'CASE' in sql_upper and 'WHEN' in sql_upper
    }
    
    # Additional specific categorizations
    query_types['simple_join'] = query_types['join'] and not query_types['group_by'] and not query_types['nested_query']
    query_types['group_by_having'] = query_types['group_by'] and query_types['having']
    
    return query_types


def check_aggregation_functions(sql: str) -> Dict[str, bool]:
    """Check for the presence of aggregation functions in SQL query."""
    sql_upper = sql.upper()
    
    return {
        'sum': bool(re.search(r'SUM\s*\(', sql_upper)),
        'min': bool(re.search(r'MIN\s*\(', sql_upper)),
        'max': bool(re.search(r'MAX\s*\(', sql_upper)),
        'avg': bool(re.search(r'AVG\s*\(', sql_upper)),
        'count': bool(re.search(r'COUNT\s*\(', sql_upper))
    }


def check_special_functions(sql: str) -> Dict[str, bool]:
    """Check for the presence of special functions in SQL query."""
    sql_upper = sql.upper()
    
    return {
        'cast': bool(re.search(r'CAST\s*\(', sql_upper)),
        'coalesce': bool(re.search(r'COALESCE\s*\(', sql_upper)),
        'substring': bool(re.search(r'SUBSTR(ING)?\s*\(', sql_upper)),
        'date_functions': any(f in sql_upper for f in ['DATE(', 'STRFTIME(', 'DATETIME(']),
        'math_operations': any(op in sql_upper for op in [' + ', ' - ', ' * ', ' / '])
    }


def has_numerical_values(text: str) -> bool:
    """Check if the text contains numerical values."""
    return bool(re.search(r'\b\d+\.?\d*\b', text))


def analyze_bird_dataset():
    """Main function to analyze the BIRD dataset."""
    # Load data
    dev_data = load_dev_data(DEV_JSON_PATH)
    print(f"Loaded {len(dev_data)} examples from the dataset.")
    
    # Initialize statistics
    stats = {
        'num_tables': 0,
        'num_queries': len(dev_data),
        'tables': set(),
        'databases': set(),
        'attributes_per_table': [],
        'numerical_attrs_per_table': [],
        'categorical_attrs_per_table': [],
        'values_per_attribute': [],
        'query_types': defaultdict(int),
        'numerical_in_question': 0,
        'numerical_in_query': 0,
        'aggregation_functions': defaultdict(int),
        'special_functions': defaultdict(int),
        'question_lengths': [],
        'query_lengths': []
    }
    
    # Try to analyze database schema
    db_schema_info = {}
    db_paths = get_database_paths()
    for db_path in db_paths:
        db_name = os.path.basename(os.path.dirname(db_path))
        db_schema_info[db_name] = analyze_database_schema(db_path)
        
        stats['num_tables'] += len(db_schema_info[db_name]['tables'])
        
        for table, num_attrs in db_schema_info[db_name]['attributes_per_table'].items():
            stats['attributes_per_table'].append(num_attrs)
        
        for table, num_attrs in db_schema_info[db_name]['numerical_attrs'].items():
            stats['numerical_attrs_per_table'].append(len(num_attrs))
        
        for table, num_attrs in db_schema_info[db_name]['categorical_attrs'].items():
            stats['categorical_attrs_per_table'].append(len(num_attrs))
        
        for table, attrs in db_schema_info[db_name]['values_per_attribute'].items():
            for attr, count in attrs.items():
                stats['values_per_attribute'].append(count)
    
    # If we couldn't access the database files, try to infer from SQL queries
    if not db_schema_info:
        print("No database files accessed. Inferring information from SQL queries...")
        table_attributes = defaultdict(set)
        numerical_attrs = defaultdict(set)
        categorical_attrs = defaultdict(set)
        
        for example in dev_data:
            # Extract tables from SQL
            sql = example['SQL']
            tables = extract_tables_from_sql(sql)
            stats['tables'].update(tables)
            
            # Try to infer column types from SQL usage
            sql_upper = sql.upper()
            
            # Extract column names (simplified approach)
            column_matches = re.findall(r'`([^`]+)`', sql)  # Find column names in backticks
            
            for col in column_matches:
                for table in tables:
                    table_attributes[table].add(col)
                    
                    # Heuristic: if column is used in a numerical operation, it's likely numerical
                    if re.search(rf"{re.escape(col)}\\s*[+\\-*/]|[+\\-*/]\\s*{re.escape(col)}", sql, re.IGNORECASE) or \
                       any(func in sql_upper for func in [f"SUM({col.upper()})", f"AVG({col.upper()})", 
                                                         f"MIN({col.upper()})", f"MAX({col.upper()})"]):
                        numerical_attrs[table].add(col)
                    else:
                        categorical_attrs[table].add(col)
        
        # Update stats
        stats['num_tables'] = len(stats['tables'])
        stats['attributes_per_table'] = [len(attrs) for table, attrs in table_attributes.items()]
        stats['numerical_attrs_per_table'] = [len(attrs) for table, attrs in numerical_attrs.items()]
        stats['categorical_attrs_per_table'] = [len(attrs) for table, attrs in categorical_attrs.items()]
    
    # Analyze each query
    for example in dev_data:
        question = example['question']
        sql = example['SQL']
        db_id = example.get('db_id', 'unknown')
        
        stats['databases'].add(db_id)
        stats['question_lengths'].append(len(question))
        stats['query_lengths'].append(len(sql))
        
        # Check for numerical values
        if has_numerical_values(question):
            stats['numerical_in_question'] += 1
        if has_numerical_values(sql):
            stats['numerical_in_query'] += 1
        
        # Analyze query types
        query_types = analyze_query_type(sql)
        for qtype, present in query_types.items():
            if present:
                stats['query_types'][qtype] += 1
        
        # Check for aggregation functions
        agg_functions = check_aggregation_functions(sql)
        for func, present in agg_functions.items():
            if present:
                stats['aggregation_functions'][func] += 1
        
        # Check for special functions
        special_functions = check_special_functions(sql)
        for func, present in special_functions.items():
            if present:
                stats['special_functions'][func] += 1
    
    # Calculate averages
    stats['avg_question_length'] = np.mean(stats['question_lengths']) if stats['question_lengths'] else 0
    stats['avg_query_length'] = np.mean(stats['query_lengths']) if stats['query_lengths'] else 0
    
    if stats['attributes_per_table']:
        stats['avg_attributes_per_table'] = np.mean(stats['attributes_per_table'])
        stats['min_attributes_per_table'] = np.min(stats['attributes_per_table'])
        stats['max_attributes_per_table'] = np.max(stats['attributes_per_table'])
    
    if stats['numerical_attrs_per_table']:
        stats['avg_numerical_attrs_per_table'] = np.mean(stats['numerical_attrs_per_table'])
    
    if stats['categorical_attrs_per_table']:
        stats['avg_categorical_attrs_per_table'] = np.mean(stats['categorical_attrs_per_table'])
    
    if stats['values_per_attribute']:
        stats['avg_values_per_attribute'] = np.mean(stats['values_per_attribute'])
    
    return stats


def print_statistics(stats):
    """Print the computed statistics in a readable format."""
    print("\n=== BIRD Dataset Statistics ===\n")
    
    # Global stats
    print("Global Statistics:")
    print(f"Number of databases: {len(stats['databases'])}")
    print(f"Number of tables: {stats['num_tables']}")
    print(f"Number of queries: {stats['num_queries']}")
    
    # Table attributes
    print("\nTable Attributes:")
    if 'avg_attributes_per_table' in stats:
        print(f"Average attributes per table: {stats['avg_attributes_per_table']:.2f}")
        if 'min_attributes_per_table' in stats and 'max_attributes_per_table' in stats:
            print(f"Distribution: Min={stats['min_attributes_per_table']}, Max={stats['max_attributes_per_table']}")
    
    if 'avg_values_per_attribute' in stats:
        print(f"Average values per attribute: {stats['avg_values_per_attribute']:.2f}")
    
    if 'avg_numerical_attrs_per_table' in stats:
        print(f"Average numerical attributes per table: {stats['avg_numerical_attrs_per_table']:.2f}")
    
    if 'avg_categorical_attrs_per_table' in stats:
        print(f"Average categorical attributes per table: {stats['avg_categorical_attrs_per_table']:.2f}")
    
    # Query types
    print("\nQuery Types:")
    for qtype, count in sorted(stats['query_types'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {qtype}: {count} queries ({count/stats['num_queries']*100:.1f}%)")
    
    # Values in questions/queries
    print("\nNumeric Values:")
    print(f"Questions with numerical values: {stats['numerical_in_question']} ({stats['numerical_in_question']/stats['num_queries']*100:.1f}%)")
    print(f"Queries with numerical values: {stats['numerical_in_query']} ({stats['numerical_in_query']/stats['num_queries']*100:.1f}%)")
    
    # Aggregation functions
    print("\nAggregation Functions:")
    for func, count in sorted(stats['aggregation_functions'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {func}: {count} queries ({count/stats['num_queries']*100:.1f}%)")
    
    # Special functions
    print("\nSpecial Functions:")
    for func, count in sorted(stats['special_functions'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {func}: {count} queries ({count/stats['num_queries']*100:.1f}%)")
    
    # Question and query length
    print("\nText Length:")
    print(f"Average question length: {stats['avg_question_length']:.1f} characters")
    print(f"Average query length: {stats['avg_query_length']:.1f} characters")


def plot_distributions(stats):
    """Create visualizations of key distributions."""
    # Query types distribution
    plt.figure(figsize=(10, 6))
    query_types = {k: v for k, v in stats['query_types'].items() if v > 0}
    query_types = dict(sorted(query_types.items(), key=lambda x: x[1], reverse=True))
    
    plt.bar(query_types.keys(), query_types.values())
    plt.title('Distribution of Query Types')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('query_types_distribution.png')
    
    # Attributes per table
    if stats['attributes_per_table']:
        plt.figure(figsize=(8, 5))
        plt.hist(stats['attributes_per_table'], bins=10, alpha=0.7)
        plt.title('Distribution of Attributes per Table')
        plt.xlabel('Number of Attributes')
        plt.ylabel('Frequency')
        plt.savefig('attributes_distribution.png')
    
    # Numerical vs Categorical attributes
    if stats['numerical_attrs_per_table'] and stats['categorical_attrs_per_table']:
        plt.figure(figsize=(8, 5))
        labels = ['Numerical', 'Categorical']
        values = [np.mean(stats['numerical_attrs_per_table']), 
                 np.mean(stats['categorical_attrs_per_table'])]
        
        plt.bar(labels, values)
        plt.title('Average Number of Numerical vs Categorical Attributes per Table')
        plt.ylabel('Average Number of Attributes')
        plt.savefig('numerical_vs_categorical.png')
    
    # Aggregation functions
    plt.figure(figsize=(10, 6))
    agg_functions = {k: v for k, v in stats['aggregation_functions'].items() if v > 0}
    if agg_functions:
        plt.bar(agg_functions.keys(), agg_functions.values())
        plt.title('Usage of Aggregation Functions')
        plt.savefig('aggregation_functions.png')
    
    # Question and query lengths
    plt.figure(figsize=(10, 6))
    plt.hist(stats['question_lengths'], bins=15, alpha=0.7, label='Questions')
    plt.hist(stats['query_lengths'], bins=15, alpha=0.7, label='Queries')
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('text_lengths.png')


def main():
    """Main execution function."""
    print("Starting BIRD dataset analysis...")
    
    # Analyze the dataset
    stats = analyze_bird_dataset()
    
    # Print statistics
    print_statistics(stats)
    
    # Create plots
    try:
        plot_distributions(stats)
        print("\nPlots saved to current directory.")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter, defaultdict
from pathlib import Path
import sqlparse

class BirdDatasetAnalyzer:
    """
    A class to analyze the BIRD (Benchmarking Instruction-Resolution for Databases) dataset,
    collecting comprehensive statistics about tables, queries, and their characteristics.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the analyzer with the path to the BIRD dataset.
        
        Parameters:
        dataset_path (str): Path to the BIRD dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.db_paths = []
        self.tables_info = {}
        self.queries_info = []
        self.database_stats = {}
        self.query_stats = {}
        
    def load_bird_dataset(self):
        """Load BIRD dataset metadata and queries based on official structure"""
        print("Loading BIRD dataset metadata...")
        
        # Load dev.json which contains query information
        dev_path = self.dataset_path / 'dev.json'
        if not dev_path.exists():
            # Try fallback to example file
            dev_path = self.dataset_path / 'dev_2_examples.json'
            if not dev_path.exists():
                raise FileNotFoundError(f"Could not find dev.json or dev_2_examples.json at {self.dataset_path}")
                
        with open(dev_path, 'r') as f:
            dev_data = json.load(f)
            
        # Find database directories
        db_dir = self.dataset_path / 'dev_databases'
        if not db_dir.exists():
            raise FileNotFoundError(f"Could not find database directory at {db_dir}")
            
        # Get all SQLite database paths
        self.db_paths = []
        for db_folder in db_dir.iterdir():
            if db_folder.is_dir():
                sqlite_file = db_folder / f"{db_folder.name}.sqlite"
                if sqlite_file.exists():
                    self.db_paths.append(sqlite_file)
        
        print(f"Found {len(self.db_paths)} database files")
        
        # Process queries information
        self.queries_info = []
        db_ids = set()
        
        for item in dev_data:
            db_id = item.get('db_id', '')
            db_ids.add(db_id)
            
            self.queries_info.append({
                'db_id': db_id,
                'question': item['question'],
                'query': item['SQL'],
                'evidence': item.get('evidence', ''),
                'difficulty': item.get('difficulty', 'unknown'),
                'question_id': item.get('question_id', -1)
            })
        
        print(f"Loaded {len(self.queries_info)} queries")
        
        # Initialize database schema information
        for db_id in db_ids:
            self.tables_info[db_id] = {
                'table_names': [],
                'column_names': [],
                'column_types': []
            }
        
        # Attempt to extract schema information from databases
        self._extract_db_schemas()
        
        return self.tables_info, self.queries_info
    
    def _extract_db_schemas(self):
        """Extract database schema information from SQLite files"""
        for db_path in self.db_paths:
            db_id = db_path.parent.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_names = []
                column_names = []
                column_types = []
                
                for table_idx, table in enumerate(tables):
                    table_names.append(table)
                    
                    # Get column information
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = cursor.fetchall()
                    
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2].lower()
                        
                        # Format similar to Spider: [table_idx, column_name]
                        column_names.append([table_idx, col_name])
                        column_types.append(col_type)
                
                self.tables_info[db_id] = {
                    'table_names': table_names,
                    'column_names': column_names,
                    'column_types': column_types
                }
                
                conn.close()
                
            except (sqlite3.Error, FileNotFoundError) as e:
                print(f"Error extracting schema from {db_path}: {e}")
    
    def analyze_database_schemas(self):
        """Analyze database schemas to collect statistics about tables and attributes"""
        print("Analyzing database schemas...")
        
        if not self.tables_info:
            raise ValueError("No tables information loaded. Call load_bird_dataset() first.")
            
        db_stats = {}
        
        for db_name, db_info in self.tables_info.items():
            num_tables = len(db_info['table_names'])
            if num_tables == 0:
                continue  # Skip databases with no table information
            
            # Initialize counts
            column_counts = [0] * num_tables
            numerical_counts = [0] * num_tables
            categorical_counts = [0] * num_tables
            
            # Count columns per table
            for col_info in db_info['column_names']:
                if len(col_info) >= 2 and col_info[0] >= 0:  # Skip table-less columns
                    table_idx = col_info[0]
                    if table_idx < len(column_counts):
                        column_counts[table_idx] += 1
            
            # Count numerical vs categorical columns
            for i, col_type in enumerate(db_info['column_types']):
                if i < len(db_info['column_names']):
                    col_info = db_info['column_names'][i]
                    if len(col_info) >= 2 and col_info[0] >= 0:  # Skip table-less columns
                        table_idx = col_info[0]
                        if table_idx < len(numerical_counts):
                            if col_type in ['number', 'real', 'int', 'integer', 'float', 'double', 'numeric', 'decimal']:
                                numerical_counts[table_idx] += 1
                            else:
                                categorical_counts[table_idx] += 1
            
            # Compute statistics for this database
            db_stats[db_name] = {
                'num_tables': num_tables,
                'avg_attributes_per_table': np.mean(column_counts) if column_counts else 0,
                'median_attributes_per_table': np.median(column_counts) if column_counts else 0,
                'min_attributes_per_table': min(column_counts) if column_counts else 0,
                'max_attributes_per_table': max(column_counts) if column_counts else 0,
                'avg_numerical_attributes': np.mean(numerical_counts) if numerical_counts else 0,
                'avg_categorical_attributes': np.mean(categorical_counts) if categorical_counts else 0,
                'total_numerical_attributes': sum(numerical_counts),
                'total_categorical_attributes': sum(categorical_counts),
            }
            
            # Analyze actual data in the database to get distribution of values per attribute
            db_file = None
            for path in self.db_paths:
                if db_name in path.parent.name:
                    db_file = path
                    break
                    
            if db_file:
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    
                    values_per_attribute = []
                    
                    for table_idx, table_name in enumerate(db_info['table_names']):
                        try:
                            # Get row count
                            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                            row_count = cursor.fetchone()[0]
                            
                            # Get columns for this table
                            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                            columns = [row[1] for row in cursor.fetchall()]
                            
                            # For each column, count distinct values
                            for col in columns:
                                try:
                                    cursor.execute(f"SELECT COUNT(DISTINCT `{col}`) FROM `{table_name}`")
                                    distinct_count = cursor.fetchone()[0]
                                    values_per_attribute.append(distinct_count)
                                except sqlite3.Error as e:
                                    print(f"Error analyzing column {col} in table {table_name}: {e}")
                        except sqlite3.Error as e:
                            print(f"Error analyzing table {table_name}: {e}")
                    
                    if values_per_attribute:
                        db_stats[db_name]['avg_values_per_attribute'] = np.mean(values_per_attribute)
                        db_stats[db_name]['median_values_per_attribute'] = np.median(values_per_attribute)
                        db_stats[db_name]['min_values_per_attribute'] = min(values_per_attribute)
                        db_stats[db_name]['max_values_per_attribute'] = max(values_per_attribute)
                    
                    conn.close()
                except (sqlite3.Error, FileNotFoundError) as e:
                    print(f"Error analyzing database file {db_file}: {e}")
                    # If database file can't be accessed, skip this analysis
                    db_stats[db_name]['avg_values_per_attribute'] = None
                    db_stats[db_name]['median_values_per_attribute'] = None
                    db_stats[db_name]['min_values_per_attribute'] = None
                    db_stats[db_name]['max_values_per_attribute'] = None
        
        self.database_stats = db_stats
        return db_stats
    
    def analyze_queries(self):
        """Analyze queries to collect statistics about query types, complexity, etc."""
        print("Analyzing queries...")
        
        if not self.queries_info:
            raise ValueError("No queries information loaded. Call load_bird_dataset() first.")
            
        query_types = Counter()
        agg_functions = Counter()
        special_functions = Counter()
        question_lengths = []
        query_lengths = []
        evidence_lengths = []
        queries_with_numerical = 0
        queries_with_categorical = 0
        difficulty_counts = Counter()
        queries_per_db = defaultdict(int)
        
        for query_info in self.queries_info:
            db_id = query_info['db_id']
            question = query_info['question']
            query = query_info['query']
            evidence = query_info.get('evidence', '')
            difficulty = query_info.get('difficulty', 'unknown')
            
            # Count queries per database and difficulty
            queries_per_db[db_id] += 1
            difficulty_counts[difficulty] += 1
            
            # Question, evidence, and query length
            question_lengths.append(len(question))
            query_lengths.append(len(query))
            if evidence:
                evidence_lengths.append(len(evidence))
            
            # Parse the SQL query
            try:
                parsed = sqlparse.parse(query)[0]
                
                # Analyze query type
                query_type = self._determine_query_type(parsed, query)
                query_types[query_type] += 1
            except IndexError:
                # If parsing fails, categorize as "unparseable"
                query_types["unparseable"] += 1
            
            # Check for aggregation functions
            if re.search(r'\b(SUM|MIN|MAX|AVG|COUNT)\s*\(', query, re.IGNORECASE):
                for agg in ['SUM', 'MIN', 'MAX', 'AVG', 'COUNT']:
                    if re.search(r'\b' + agg + r'\s*\(', query, re.IGNORECASE):
                        agg_functions[agg] += 1
            
            # Check for special functions
            special_funcs = {
                'CAST': r'\bCAST\s*\(',
                'LIKE': r'\bLIKE\b',
                'IN': r'\bIN\s*\(',
                'EXISTS': r'\bEXISTS\b',
                'BETWEEN': r'\bBETWEEN\b',
                'CASE': r'\bCASE\b',
                'COALESCE': r'\bCOALESCE\s*\(',
                'SUBSTRING': r'\bSUBSTR(ING)?\s*\(',
                'DATE_FUNCTIONS': r'\b(DATE|STRFTIME|DATETIME)\s*\(',
                'MATH_OPERATIONS': r'[+\-*/]'
            }
            
            for func_name, pattern in special_funcs.items():
                if re.search(pattern, query, re.IGNORECASE):
                    special_functions[func_name] += 1
            
            # Check for numerical values in question and query
            has_numerical = bool(re.search(r'\b\d+(\.\d+)?\b', question))
            has_categorical = bool(re.search(r"'[^']*'", query))
            
            if has_numerical:
                queries_with_numerical += 1
            if has_categorical:
                queries_with_categorical += 1
        
        # Calculate global statistics
        num_queries = len(self.queries_info)
        self.query_stats = {
            'total_queries': num_queries,
            'query_types': dict(query_types),
            'query_types_percentage': {k: v/num_queries*100 for k, v in query_types.items()},
            'aggregation_functions': dict(agg_functions),
            'special_functions': dict(special_functions),
            'avg_question_length': np.mean(question_lengths),
            'avg_query_length': np.mean(query_lengths),
            'avg_evidence_length': np.mean(evidence_lengths) if evidence_lengths else 0,
            'queries_with_numerical': queries_with_numerical,
            'queries_with_numerical_percentage': queries_with_numerical / num_queries * 100,
            'queries_with_categorical': queries_with_categorical,
            'queries_with_categorical_percentage': queries_with_categorical / num_queries * 100,
            'difficulty_distribution': dict(difficulty_counts),
            'difficulty_percentage': {k: v/num_queries*100 for k, v in difficulty_counts.items()},
            'queries_per_db': dict(queries_per_db),
            'avg_queries_per_db': np.mean(list(queries_per_db.values())) if queries_per_db else 0,
        }
        
        return self.query_stats
    
    def _determine_query_type(self, parsed_query, query_str):
        """Determine the type of the SQL query"""
        # Convert to lowercase for case-insensitive matching
        query_lower = query_str.lower()
        
        # Define query type categories and their detection patterns
        categories = {
            'simple_select': lambda q: 'select' in q and 'from' in q and 'join' not in q and 'group by' not in q and 'having' not in q and 'order by' not in q,
            'simple_join': lambda q: 'join' in q and 'group by' not in q and 'having' not in q,
            'group_by': lambda q: 'group by' in q and 'having' not in q,
            'group_by_having': lambda q: 'group by' in q and 'having' in q,
            'order_by': lambda q: 'order by' in q,
            'limit': lambda q: 'limit' in q,
            'distinct': lambda q: 'distinct' in q,
            'case_when': lambda q: 'case' in q and 'when' in q,
            'union': lambda q: 'union' in q,
            'intersect': lambda q: 'intersect' in q,
            'except': lambda q: 'except' in q,
        }
        
        # Check for nested queries
        has_nested = '(' in query_str and 'select' in query_lower.split('(')[1].split(')')[0]
        
        # Determine the primary query type
        query_type = 'other'
        for category, condition in categories.items():
            if condition(query_lower):
                query_type = category
                break
        
        # Add nested query indicator if applicable
        if has_nested:
            query_type = f"nested_{query_type}"
        
        return query_type
    
    def generate_global_statistics(self):
        """Generate global statistics about the dataset"""
        if not self.database_stats:
            self.analyze_database_schemas()
        if not self.query_stats:
            self.analyze_queries()
        
        # Combine database and query statistics
        global_stats = {
            'dataset_name': 'BIRD',
            'num_databases': len(self.tables_info),
            'total_tables': sum(db.get('num_tables', 0) for db in self.database_stats.values()),
            'total_queries': self.query_stats['total_queries'],
            'avg_tables_per_db': np.mean([db.get('num_tables', 0) for db in self.database_stats.values()]) if self.database_stats else 0,
            'avg_queries_per_db': self.query_stats['avg_queries_per_db'],
        }
        
        # Difficulty distribution
        if 'difficulty_distribution' in self.query_stats:
            global_stats['difficulty_distribution'] = self.query_stats['difficulty_distribution']
        
        # Attributes per table statistics
        if self.database_stats:
            attrs_stats = [db.get('avg_attributes_per_table', 0) for db in self.database_stats.values() if db.get('avg_attributes_per_table', 0) > 0]
            if attrs_stats:
                global_stats['attributes_per_table_stats'] = {
                    'mean': np.mean(attrs_stats),
                    'median': np.median(attrs_stats),
                    'min': min(attrs_stats),
                    'max': max(attrs_stats),
                }
            
            # Numerical vs categorical attributes
            num_attrs = [db.get('avg_numerical_attributes', 0) for db in self.database_stats.values() if db.get('avg_numerical_attributes', 0) > 0]
            cat_attrs = [db.get('avg_categorical_attributes', 0) for db in self.database_stats.values() if db.get('avg_categorical_attributes', 0) > 0]
            
            if num_attrs:
                global_stats['numerical_attributes_stats'] = {
                    'mean': np.mean(num_attrs),
                    'total': sum([db.get('total_numerical_attributes', 0) for db in self.database_stats.values()]),
                }
            
            if cat_attrs:
                global_stats['categorical_attributes_stats'] = {
                    'mean': np.mean(cat_attrs),
                    'total': sum([db.get('total_categorical_attributes', 0) for db in self.database_stats.values()]),
                }
        
        # Query type distribution
        global_stats['query_type_distribution'] = self.query_stats['query_types_percentage']
        
        # Aggregation function usage
        global_stats['aggregation_function_usage'] = self.query_stats['aggregation_functions']
        
        # Special function usage
        global_stats['special_function_usage'] = self.query_stats['special_functions']
        
        # Text lengths
        global_stats['avg_question_length'] = self.query_stats['avg_question_length']
        global_stats['avg_query_length'] = self.query_stats['avg_query_length']
        global_stats['avg_evidence_length'] = self.query_stats['avg_evidence_length']
        
        # Numerical/categorical presence
        global_stats['queries_with_numerical_percentage'] = self.query_stats['queries_with_numerical_percentage']
        global_stats['queries_with_categorical_percentage'] = self.query_stats['queries_with_categorical_percentage']
        
        # Values per attribute statistics
        values_per_attr_stats = [db.get('avg_values_per_attribute') for db in self.database_stats.values() 
                                if db.get('avg_values_per_attribute') is not None]
        if values_per_attr_stats:
            global_stats['values_per_attribute_stats'] = {
                'mean': np.mean(values_per_attr_stats),
                'median': np.median(values_per_attr_stats),
                'min': min(values_per_attr_stats),
                'max': max(values_per_attr_stats),
            }
        
        return global_stats
    
    def plot_statistics(self, output_dir=None):
        """Generate plots for the collected statistics"""
        if not self.database_stats:
            self.analyze_database_schemas()
        if not self.query_stats:
            self.analyze_queries()
        
        if output_dir is None:
            output_dir = "bird_analysis"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Plot distribution of tables per database
        if self.database_stats:
            plt.figure(figsize=(10, 6))
            tables_per_db = [db.get('num_tables', 0) for db in self.database_stats.values()]
            if any(tables_per_db):
                sns.histplot(tables_per_db, kde=True)
                plt.title('Distribution of Tables per Database')
                plt.xlabel('Number of Tables')
                plt.ylabel('Frequency')
                plt.savefig(f"{output_dir}/tables_per_db_distribution.png")
                plt.close()
        
        # 2. Plot distribution of attributes per table
        if self.database_stats:
            plt.figure(figsize=(10, 6))
            attrs_per_table = [db.get('avg_attributes_per_table', 0) for db in self.database_stats.values() if db.get('avg_attributes_per_table', 0) > 0]
            if attrs_per_table:
                sns.histplot(attrs_per_table, kde=True)
                plt.title('Distribution of Average Attributes per Table')
                plt.xlabel('Average Number of Attributes')
                plt.ylabel('Frequency')
                plt.savefig(f"{output_dir}/attrs_per_table_distribution.png")
                plt.close()
        
        # 3. Plot distribution of numerical vs categorical attributes
        if self.database_stats:
            valid_dbs = [db_name for db_name, db in self.database_stats.items() 
                       if db.get('avg_numerical_attributes', 0) > 0 or db.get('avg_categorical_attributes', 0) > 0]
            
            if valid_dbs:
                plt.figure(figsize=(10, 6))
                db_names = valid_dbs[:10]  # Show only top 10 for readability
                num_attrs = [self.database_stats[db].get('avg_numerical_attributes', 0) for db in db_names]
                cat_attrs = [self.database_stats[db].get('avg_categorical_attributes', 0) for db in db_names]
                
                df = pd.DataFrame({
                    'Database': db_names,
                    'Numerical': num_attrs,
                    'Categorical': cat_attrs
                })
                
                df_melted = df.melt(id_vars=['Database'], var_name='Attribute Type', value_name='Count')
                sns.barplot(x='Database', y='Count', hue='Attribute Type', data=df_melted)
                plt.title('Numerical vs Categorical Attributes (Top 10 Databases)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/numerical_vs_categorical.png")
                plt.close()
        
        # 4. Plot difficulty distribution
        if 'difficulty_distribution' in self.query_stats:
            plt.figure(figsize=(8, 6))
            diff_dict = self.query_stats['difficulty_distribution']
            if diff_dict:
                plt.bar(diff_dict.keys(), diff_dict.values())
                plt.title('Query Difficulty Distribution')
                plt.xlabel('Difficulty')
                plt.ylabel('Count')
                plt.savefig(f"{output_dir}/difficulty_distribution.png")
                plt.close()
        
        # 5. Plot query type distribution
        plt.figure(figsize=(12, 6))
        query_types = self.query_stats['query_types']
        if query_types:
            sorted_types = dict(sorted(query_types.items(), key=lambda x: x[1], reverse=True))
            
            plt.bar(sorted_types.keys(), sorted_types.values())
            plt.title('Query Type Distribution')
            plt.xlabel('Query Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/query_type_distribution.png")
            plt.close()
        
        # 6. Plot aggregation function usage
        plt.figure(figsize=(10, 6))
        agg_funcs = self.query_stats['aggregation_functions']
        if agg_funcs:
            plt.bar(agg_funcs.keys(), agg_funcs.values())
            plt.title('Aggregation Function Usage')
            plt.xlabel('Function')
            plt.ylabel('Count')
            plt.savefig(f"{output_dir}/agg_function_usage.png")
            plt.close()
        
        # 7. Plot special function usage
        plt.figure(figsize=(10, 6))
        special_funcs = self.query_stats['special_functions']
        if special_funcs:
            plt.bar(special_funcs.keys(), special_funcs.values())
            plt.title('Special Function Usage')
            plt.xlabel('Function')
            plt.ylabel('Count')
            plt.savefig(f"{output_dir}/special_function_usage.png")
            plt.close()
        
        # 8. Plot question and query length distributions
        plt.figure(figsize=(10, 6))
        question_lengths = [len(q['question']) for q in self.queries_info]
        if question_lengths:
            sns.histplot(question_lengths, kde=True)
            plt.title('Distribution of Question Lengths')
            plt.xlabel('Number of Characters')
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/question_length_distribution.png")
            plt.close()
        
        plt.figure(figsize=(10, 6))
        query_lengths = [len(q['query']) for q in self.queries_info]
        if query_lengths:
            sns.histplot(query_lengths, kde=True)
            plt.title('Distribution of Query Lengths')
            plt.xlabel('Number of Characters')
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/query_length_distribution.png")
            plt.close()
        
        # 9. Plot queries per database distribution
        plt.figure(figsize=(10, 6))
        queries_per_db = list(self.query_stats['queries_per_db'].values())
        if queries_per_db:
            sns.histplot(queries_per_db, kde=True)
            plt.title('Distribution of Queries per Database')
            plt.xlabel('Number of Queries')
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/queries_per_db_distribution.png")
            plt.close()
        
        # 10. Plot evidence length distribution (unique to BIRD)
        evidence_lengths = [len(q.get('evidence', '')) for q in self.queries_info if q.get('evidence', '')]
        if evidence_lengths:
            plt.figure(figsize=(10, 6))
            sns.histplot(evidence_lengths, kde=True)
            plt.title('Distribution of Evidence Lengths')
            plt.xlabel('Number of Characters')
            plt.ylabel('Frequency')
            plt.savefig(f"{output_dir}/evidence_length_distribution.png")
            plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def save_statistics(self, output_file):
        """Save all collected statistics to a JSON file"""
        global_stats = self.generate_global_statistics()
        
        # Combine all statistics
        all_stats = {
            'global_stats': global_stats,
            'database_stats': self.database_stats,
            'query_stats': self.query_stats
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"Statistics saved to {output_file}")

# Example usage
def main():
    # Analyze BIRD dataset
    bird_path = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/Data/data/bird"  # Update with your BIRD dataset path
    analyzer = BirdDatasetAnalyzer(bird_path)
    
    # Load BIRD dataset
    analyzer.load_bird_dataset()
    
    # Analyze database schemas
    analyzer.analyze_database_schemas()
    
    # Analyze queries
    analyzer.analyze_queries()
    
    # Generate global statistics
    global_stats = analyzer.generate_global_statistics()
    
    # Print some key statistics
    print("\n=== BIRD Dataset Statistics ===")
    print(f"Number of Databases: {global_stats['num_databases']}")
    print(f"Total Tables: {global_stats['total_tables']}")
    print(f"Total Queries: {global_stats['total_queries']}")
    
    if 'attributes_per_table_stats' in global_stats:
        print(f"Average Attributes per Table: {global_stats['attributes_per_table_stats']['mean']:.2f}")
    
    print(f"Average Question Length: {global_stats['avg_question_length']:.2f} characters")
    print(f"Average Query Length: {global_stats['avg_query_length']:.2f} characters")
    
    if 'avg_evidence_length' in global_stats:
        print(f"Average Evidence Length: {global_stats['avg_evidence_length']:.2f} characters")
    
    # Plot statistics
    analyzer.plot_statistics(output_dir="bird_analysis")
    
    # Save statistics to file
    analyzer.save_statistics("bird_stats.json")

if __name__ == "__main__":
    main()
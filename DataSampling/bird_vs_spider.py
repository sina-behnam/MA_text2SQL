"""
BIRD vs Spider Dataset Comparison

This script analyzes and compares statistics from the BIRD and Spider datasets
for text-to-SQL benchmarks.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, Any

def load_stats_files(bird_json_path: str, spider_json_path: str):
    """Load the statistics JSON files for both datasets."""
    with open(bird_json_path, 'r') as f:
        bird_stats = json.load(f)
    
    with open(spider_json_path, 'r') as f:
        spider_stats = json.load(f)
    
    return bird_stats, spider_stats

def compare_global_statistics(bird_stats: Dict[str, Any], spider_stats: Dict[str, Any]):
    """Compare global statistics between BIRD and Spider datasets."""
    bird_global = bird_stats['global_stats']
    spider_global = spider_stats['global_stats']
    
    # Define common metrics to compare
    comparison_metrics = {
        'Dataset': ['BIRD', 'Spider'],
        'Number of Databases': [bird_global['num_databases'], spider_global['num_databases']],
        'Total Tables': [bird_global['total_tables'], spider_global['total_tables']],
        'Total Queries': [bird_global['total_queries'], spider_global['total_queries']],
        'Avg Tables per DB': [bird_global['avg_tables_per_db'], spider_global['avg_tables_per_db']],
        'Avg Queries per DB': [bird_global['avg_queries_per_db'], spider_global['avg_queries_per_db']],
        'Avg Question Length (chars)': [bird_global['avg_question_length'], spider_global['avg_question_length']],
        'Avg Query Length (chars)': [bird_global['avg_query_length'], spider_global['avg_query_length']],
        'Queries with Numerical Values (%)': [bird_global['queries_with_numerical_percentage'], 
                                           spider_global['queries_with_numerical_percentage']],
        'Queries with Categorical Values (%)': [bird_global['queries_with_categorical_percentage'], 
                                             spider_global['queries_with_categorical_percentage']]
    }
    
    # Calculate averages for attribute statistics
    bird_attr_stats = bird_global.get('attributes_per_table_stats', {})
    spider_attr_stats = spider_global.get('attributes_per_table_stats', {})
    
    if bird_attr_stats and spider_attr_stats:
        comparison_metrics['Avg Attributes per Table'] = [
            bird_attr_stats.get('mean', 0),
            spider_attr_stats.get('mean', 0)
        ]
    
    # Calculate averages for numerical vs categorical attributes
    if 'numerical_attributes_stats' in bird_global and 'categorical_attributes_stats' in bird_global:
        if 'numerical_attributes_stats' in spider_global and 'categorical_attributes_stats' in spider_global:
            comparison_metrics['Avg Numerical Attributes per Table'] = [
                bird_global['numerical_attributes_stats'].get('mean', 0),
                spider_global['numerical_attributes_stats'].get('mean', 0)
            ]
            comparison_metrics['Avg Categorical Attributes per Table'] = [
                bird_global['categorical_attributes_stats'].get('mean', 0),
                spider_global['categorical_attributes_stats'].get('mean', 0)
            ]
    
    # Calculate values per attribute
    if 'values_per_attribute_stats' in bird_global and 'values_per_attribute_stats' in spider_global:
        comparison_metrics['Avg Values per Attribute'] = [
            bird_global['values_per_attribute_stats'].get('mean', 0),
            spider_global['values_per_attribute_stats'].get('mean', 0)
        ]
    
    return comparison_metrics

def create_comparison_table(comparison_metrics):
    """Create a formatted pandas DataFrame from the comparison metrics."""
    df = pd.DataFrame(comparison_metrics)
    df = df.set_index('Dataset')
    return df.T

def plot_comparison_bar_charts(comparison_df, output_dir="./comparison_plots"):
    """Create bar charts to compare key statistics between datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Basic comparison metrics
    basic_metrics = [
        'Number of Databases', 
        'Total Tables', 
        'Avg Tables per DB', 
        'Avg Queries per DB'
    ]
    
    # Plot basic metrics
    plt.figure(figsize=(12, 8))
    comparison_df.loc[basic_metrics].plot(kind='bar', rot=0)
    plt.title('Basic Dataset Metrics: BIRD vs Spider')
    plt.ylabel('Count')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/basic_metrics_comparison.png")
    plt.close()
    
    # Total queries on a separate chart (due to scale difference)
    plt.figure(figsize=(8, 6))
    comparison_df.loc[['Total Queries']].plot(kind='bar', rot=0, 
                                            color=['#1f77b4', '#ff7f0e'])
    plt.title('Total Queries: BIRD vs Spider')
    plt.ylabel('Count')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_queries_comparison.png")
    plt.close()
    
    # Attribute metrics
    attribute_metrics = [
        'Avg Attributes per Table', 
        'Avg Numerical Attributes per Table', 
        'Avg Categorical Attributes per Table'
    ]
    
    if all(metric in comparison_df.index for metric in attribute_metrics):
        plt.figure(figsize=(12, 8))
        comparison_df.loc[attribute_metrics].plot(kind='bar', rot=0)
        plt.title('Attribute Metrics: BIRD vs Spider')
        plt.ylabel('Average Count')
        plt.legend(title='Dataset')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/attribute_metrics_comparison.png")
        plt.close()
    
    # Query complexity metrics
    query_metrics = [
        'Avg Question Length (chars)', 
        'Avg Query Length (chars)'
    ]
    
    plt.figure(figsize=(10, 7))
    comparison_df.loc[query_metrics].plot(kind='bar', rot=0)
    plt.title('Query Complexity: BIRD vs Spider')
    plt.ylabel('Average Length (characters)')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/query_complexity_comparison.png")
    plt.close()
    
    # Value metrics
    value_metrics = [
        'Queries with Numerical Values (%)', 
        'Queries with Categorical Values (%)'
    ]
    
    plt.figure(figsize=(10, 7))
    comparison_df.loc[value_metrics].plot(kind='bar', rot=0)
    plt.title('Query Value Types: BIRD vs Spider')
    plt.ylabel('Percentage of Queries')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/query_values_comparison.png")
    plt.close()
    
    # Average Values per Attribute (if available)
    if 'Avg Values per Attribute' in comparison_df.index:
        plt.figure(figsize=(8, 6))
        comparison_df.loc[['Avg Values per Attribute']].plot(kind='bar', rot=0)
        plt.title('Average Values per Attribute: BIRD vs Spider')
        plt.ylabel('Average Count')
        plt.legend(title='Dataset')
        plt.yscale('log')  # Use log scale due to potential large differences
        plt.tight_layout()
        plt.savefig(f"{output_dir}/values_per_attribute_comparison.png")
        plt.close()

def compare_query_types(bird_stats, spider_stats, output_dir="./comparison_plots"):
    """Compare query types between BIRD and Spider datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    bird_query_types = bird_stats['query_stats']['query_types_percentage']
    spider_query_types = spider_stats['query_stats']['query_types_percentage']
    
    # Map similar query types between datasets
    # This is needed because the categorization may differ slightly
    bird_types = {
        'simple_select': bird_query_types.get('simple_select', 0),
        'simple_join': bird_query_types.get('simple_join', 0),
        'group_by': bird_query_types.get('group_by', 0),
        'group_by_having': bird_query_types.get('group_by_having', 0),
        'order_by': bird_query_types.get('order_by', 0),
        'nested': (bird_query_types.get('nested_simple_select', 0) + 
                  bird_query_types.get('nested_simple_join', 0) + 
                  bird_query_types.get('nested_group_by', 0) + 
                  bird_query_types.get('nested_group_by_having', 0) + 
                  bird_query_types.get('nested_order_by', 0))
    }
    
    spider_types = {
        'simple_select': spider_query_types.get('simple', 0),
        'simple_join': spider_query_types.get('simple_join', 0),
        'group_by': spider_query_types.get('group_by', 0),
        'group_by_having': spider_query_types.get('group_by_having', 0),
        'order_by': spider_query_types.get('order_by', 0),
        'nested': (spider_query_types.get('nested_simple', 0) + 
                  spider_query_types.get('nested_simple_join', 0) + 
                  spider_query_types.get('nested_group_by', 0) + 
                  spider_query_types.get('nested_group_by_having', 0) + 
                  spider_query_types.get('nested_order_by', 0))
    }
    
    # Create DataFrame for visualization
    query_types_df = pd.DataFrame({
        'BIRD': pd.Series(bird_types),
        'Spider': pd.Series(spider_types)
    })
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    query_types_df.plot(kind='bar', rot=45)
    plt.title('Query Type Distribution: BIRD vs Spider')
    plt.ylabel('Percentage of Queries')
    plt.xlabel('Query Type')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/query_types_comparison.png")
    plt.close()
    
    return query_types_df

def compare_aggregation_functions(bird_stats, spider_stats, output_dir="./comparison_plots"):
    """Compare usage of aggregation functions between datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    bird_agg_funcs = bird_stats['query_stats']['aggregation_functions']
    spider_agg_funcs = spider_stats['query_stats']['aggregation_functions']
    
    # Convert to percentage of total queries
    bird_total = bird_stats['global_stats']['total_queries']
    spider_total = spider_stats['global_stats']['total_queries']
    
    bird_agg_percent = {func: (count / bird_total * 100) 
                        for func, count in bird_agg_funcs.items()}
    spider_agg_percent = {func: (count / spider_total * 100) 
                          for func, count in spider_agg_funcs.items()}
    
    # Create DataFrame for visualization
    agg_df = pd.DataFrame({
        'BIRD': pd.Series(bird_agg_percent),
        'Spider': pd.Series(spider_agg_percent)
    })
    
    # Plot comparison
    plt.figure(figsize=(10, 7))
    agg_df.plot(kind='bar', rot=0)
    plt.title('Aggregation Function Usage: BIRD vs Spider')
    plt.ylabel('Percentage of Queries')
    plt.xlabel('Aggregation Function')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aggregation_functions_comparison.png")
    plt.close()
    
    return agg_df

def compare_special_functions(bird_stats, spider_stats, output_dir="./comparison_plots"):
    """Compare usage of special functions between datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    bird_special_funcs = bird_stats['query_stats']['special_functions']
    spider_special_funcs = spider_stats['query_stats']['special_functions']
    
    # Convert to percentage of total queries
    bird_total = bird_stats['global_stats']['total_queries']
    spider_total = spider_stats['global_stats']['total_queries']
    
    bird_special_percent = {func: (count / bird_total * 100) 
                           for func, count in bird_special_funcs.items()}
    spider_special_percent = {func: (count / spider_total * 100) 
                             for func, count in spider_special_funcs.items()}
    
    # Get common functions between both datasets
    common_funcs = set(bird_special_funcs.keys()) & set(spider_special_funcs.keys())
    
    # Create DataFrame for common functions
    common_special_df = pd.DataFrame({
        'BIRD': pd.Series({func: bird_special_percent.get(func, 0) for func in common_funcs}),
        'Spider': pd.Series({func: spider_special_percent.get(func, 0) for func in common_funcs})
    })
    
    # Plot comparison of common functions
    plt.figure(figsize=(10, 7))
    common_special_df.plot(kind='bar', rot=0)
    plt.title('Special Function Usage (Common): BIRD vs Spider')
    plt.ylabel('Percentage of Queries')
    plt.xlabel('Special Function')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/special_functions_common_comparison.png")
    plt.close()
    
    # Create full comparison with all functions
    all_funcs = set(bird_special_funcs.keys()) | set(spider_special_funcs.keys())
    all_special_df = pd.DataFrame({
        'BIRD': pd.Series({func: bird_special_percent.get(func, 0) for func in all_funcs}),
        'Spider': pd.Series({func: spider_special_percent.get(func, 0) for func in all_funcs})
    })
    
    # Plot comparison of all functions
    plt.figure(figsize=(14, 8))
    all_special_df.plot(kind='bar', rot=45)
    plt.title('All Special Function Usage: BIRD vs Spider')
    plt.ylabel('Percentage of Queries')
    plt.xlabel('Special Function')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/special_functions_all_comparison.png")
    plt.close()
    
    return all_special_df

def compare_database_stats(bird_stats, spider_stats, output_dir="./comparison_plots"):
    """Compare database statistics between datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract statistics for each database
    bird_db_stats = bird_stats['database_stats']
    spider_db_stats = spider_stats['database_stats']
    
    # Collect metrics for all databases
    bird_tables_per_db = [db['num_tables'] for db in bird_db_stats.values()]
    spider_tables_per_db = [db['num_tables'] for db in spider_db_stats.values()]
    
    bird_attrs_per_table = [db.get('avg_attributes_per_table', 0) for db in bird_db_stats.values()]
    spider_attrs_per_table = [db.get('avg_attributes_per_table', 0) for db in spider_db_stats.values()]
    
    # Create box plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Tables per database
    axes[0].boxplot([bird_tables_per_db, spider_tables_per_db], labels=['BIRD', 'Spider'])
    axes[0].set_title('Tables per Database')
    axes[0].set_ylabel('Number of Tables')
    
    # Attributes per table
    axes[1].boxplot([bird_attrs_per_table, spider_attrs_per_table], labels=['BIRD', 'Spider'])
    axes[1].set_title('Average Attributes per Table')
    axes[1].set_ylabel('Number of Attributes')
    
    plt.suptitle('Database Structure Comparison: BIRD vs Spider')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/database_structure_comparison.png")
    plt.close()
    
    # Create histograms for more detailed distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Tables per database
    axes[0, 0].hist(bird_tables_per_db, bins=10, alpha=0.7, label='BIRD')
    axes[0, 0].set_title('Tables per Database (BIRD)')
    axes[0, 0].set_xlabel('Number of Tables')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(spider_tables_per_db, bins=10, alpha=0.7, label='Spider')
    axes[0, 1].set_title('Tables per Database (Spider)')
    axes[0, 1].set_xlabel('Number of Tables')
    
    # Attributes per table
    axes[1, 0].hist(bird_attrs_per_table, bins=10, alpha=0.7, label='BIRD')
    axes[1, 0].set_title('Average Attributes per Table (BIRD)')
    axes[1, 0].set_xlabel('Number of Attributes')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(spider_attrs_per_table, bins=10, alpha=0.7, label='Spider')
    axes[1, 1].set_title('Average Attributes per Table (Spider)')
    axes[1, 1].set_xlabel('Number of Attributes')
    
    plt.suptitle('Database Structure Distribution: BIRD vs Spider')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/database_structure_distribution.png")
    plt.close()
    
    # Return comparison dictionaries
    return {
        'bird_tables_per_db': bird_tables_per_db,
        'spider_tables_per_db': spider_tables_per_db,
        'bird_attrs_per_table': bird_attrs_per_table,
        'spider_attrs_per_table': spider_attrs_per_table
    }

def bird_vs_spider_comparison(bird_json_path, spider_json_path, output_dir="./comparison_results"):
    """Perform comprehensive comparison between BIRD and Spider datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load statistics files
    bird_stats, spider_stats = load_stats_files(bird_json_path, spider_json_path)
    
    # Compare global statistics
    comparison_metrics = compare_global_statistics(bird_stats, spider_stats)
    comparison_df = create_comparison_table(comparison_metrics)
    
    # Save comparison table to CSV
    comparison_df.to_csv(f"{output_dir}/global_statistics_comparison.csv")
    
    # Create visualizations
    plot_comparison_bar_charts(comparison_df, output_dir)
    query_types_df = compare_query_types(bird_stats, spider_stats, output_dir)
    agg_funcs_df = compare_aggregation_functions(bird_stats, spider_stats, output_dir)
    special_funcs_df = compare_special_functions(bird_stats, spider_stats, output_dir)
    db_stats_comparison = compare_database_stats(bird_stats, spider_stats, output_dir)
    
    # Save comparison DataFrames to CSV
    query_types_df.to_csv(f"{output_dir}/query_types_comparison.csv")
    agg_funcs_df.to_csv(f"{output_dir}/aggregation_functions_comparison.csv")
    special_funcs_df.to_csv(f"{output_dir}/special_functions_comparison.csv")
    
    # Generate summary report
    summary_report = f"""
# BIRD vs Spider Dataset Comparison Report

## Overview
This report provides a comparison between the BIRD and Spider text-to-SQL datasets.

## Global Statistics
{comparison_df.to_markdown()}

## Key Observations

### Database Structure
- BIRD has {bird_stats['global_stats']['num_databases']} databases with {bird_stats['global_stats']['total_tables']} tables compared to Spider's {spider_stats['global_stats']['num_databases']} databases with {spider_stats['global_stats']['total_tables']} tables.
- BIRD databases have an average of {bird_stats['global_stats']['avg_tables_per_db']:.1f} tables per database, while Spider has {spider_stats['global_stats']['avg_tables_per_db']:.1f}.
- BIRD has {bird_stats['global_stats'].get('attributes_per_table_stats', {}).get('mean', 0):.1f} attributes per table on average, compared to Spider's {spider_stats['global_stats'].get('attributes_per_table_stats', {}).get('mean', 0):.1f}.

### Query Complexity
- BIRD has {bird_stats['global_stats']['total_queries']} total queries compared to Spider's {spider_stats['global_stats']['total_queries']}.
- BIRD queries are on average {bird_stats['global_stats']['avg_query_length']:.1f} characters long, compared to Spider's {spider_stats['global_stats']['avg_query_length']:.1f}.
- BIRD questions are on average {bird_stats['global_stats']['avg_question_length']:.1f} characters long, compared to Spider's {spider_stats['global_stats']['avg_question_length']:.1f}.

### Query Distribution
- BIRD has a higher percentage of simple join queries ({bird_stats['query_stats']['query_types_percentage'].get('simple_join', 0):.1f}%) compared to Spider ({spider_stats['query_stats']['query_types_percentage'].get('simple_join', 0):.1f}%).
- Spider has a higher percentage of simple select queries ({spider_stats['query_stats']['query_types_percentage'].get('simple', 0):.1f}%) compared to BIRD ({bird_stats['query_stats']['query_types_percentage'].get('simple_select', 0):.1f}%).

### SQL Function Usage
- Top aggregation function in both datasets is COUNT.
- BIRD makes more extensive use of math operations and CAST functions.

## Conclusion
BIRD and Spider differ significantly in their size, query complexity, and the types of SQL operations they cover.
"""
    
    # Write summary report to markdown file
    with open(f"{output_dir}/comparison_summary.md", 'w') as f:
        f.write(summary_report)
    
    print(f"Comparison complete. Results saved to {output_dir}")
    
    return {
        'comparison_metrics': comparison_df,
        'query_types': query_types_df,
        'aggregation_functions': agg_funcs_df,
        'special_functions': special_funcs_df,
        'database_stats': db_stats_comparison
    }

if __name__ == "__main__":
    # Paths to the JSON statistics files
    bird_path = "bird_stats.json"
    spider_path = "spider_stats.json"
    
    # Perform comparison
    results = bird_vs_spider_comparison(bird_path, spider_path)
    
    # Print summary
    print("Comparison completed successfully!")
    print("Check the 'comparison_results' directory for detailed analysis and visualizations.")
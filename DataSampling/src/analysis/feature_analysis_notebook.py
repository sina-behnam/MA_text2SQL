import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
import gzip
from pathlib import Path
import re

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Jupyter display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.3f}'.format

# %% [markdown]
# ## 1. Data Loading
# 
# First, let's load the feature files generated by the feature engineering module.

# %%
# Function to find and load feature files
def load_feature_files(base_dir, feature_type=None, dataset=None):
    """
    Load feature files from the specified directory, handling chunked files properly.
    
    Args:
        base_dir: Base directory containing feature files
        feature_type: Optional type of features to load (e.g., 'question', 'sql', 'combined')
        dataset: Optional dataset filter (e.g., 'bird', 'spider')
        
    Returns:
        DataFrame with combined features
    """
    # Find the manifest file first if it exists
    manifest_path = os.path.join(base_dir, 'feature_extraction_manifest.json')
    files_to_load = []
    
    if os.path.exists(manifest_path):
        print(f"Found manifest file: {manifest_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        # Filter files based on parameters
        for file_path in manifest.get('files', []):
            file_name = os.path.basename(file_path)
            
            # Check for chunked files pattern
            is_chunk = '_chunk' in file_name
            base_type = None
            
            # Extract base feature type from file name (handles chunked files)
            if is_chunk:
                # For chunked files like "all_datasets_features_combined_chunk0.csv.gz"
                # Extract "combined" as the feature type
                match = re.search(r'_([a-zA-Z]+)_chunk\d+', file_name)
                if match:
                    base_type = match.group(1)
            else:
                # For regular files like "all_datasets_features_combined.csv.gz"
                match = re.search(r'_([a-zA-Z]+)\.csv', file_name)
                if match:
                    base_type = match.group(1)
            
            # Apply feature type filter
            if feature_type:
                # For combined, we need special handling as it might be chunked
                if feature_type == 'combined':
                    if base_type != 'combined' and not file_name.startswith('combined'):
                        continue
                # For other types, check both patterns: _type or starts with type
                elif base_type != feature_type and not file_name.startswith(feature_type):
                    # Additional check for full word match (not partial)
                    if f"_{feature_type}" not in file_name and not file_name.startswith(feature_type):
                        continue
            
            # Apply dataset filter
            if dataset and f"{dataset}" not in file_name:
                continue
                
            # Check if file exists at given path
            if os.path.exists(file_path):
                files_to_load.append(file_path)
            else:
                # Try relative path
                rel_path = os.path.join(base_dir, os.path.basename(file_path))
                if os.path.exists(rel_path):
                    files_to_load.append(rel_path)
    else:
        # No manifest, find files manually
        # This handles chunked files automatically through glob patterns
        
        # Define base pattern
        pattern = "*"
        if feature_type:
            # For combined, look for both combined.csv and combined_chunk*.csv
            if feature_type == 'combined':
                files_to_load = glob.glob(os.path.join(base_dir, f"*{feature_type}*.csv"))
                files_to_load.extend(glob.glob(os.path.join(base_dir, f"*{feature_type}*_chunk*.csv")))
                # Also check for compressed files
                files_to_load.extend(glob.glob(os.path.join(base_dir, f"*{feature_type}*.csv.gz")))
                files_to_load.extend(glob.glob(os.path.join(base_dir, f"*{feature_type}*_chunk*.csv.gz")))
            else:
                # For other types, use standard pattern matching
                pattern = f"*_{feature_type}*"
                if dataset:
                    pattern = f"*{dataset}*_{feature_type}*"
                
                # Find all CSV files matching the pattern
                files_to_load = glob.glob(os.path.join(base_dir, pattern + ".csv"))
                files_to_load.extend(glob.glob(os.path.join(base_dir, pattern + "_chunk*.csv")))
                
                # Also check for compressed files
                files_to_load.extend(glob.glob(os.path.join(base_dir, pattern + ".csv.gz")))
                files_to_load.extend(glob.glob(os.path.join(base_dir, pattern + "_chunk*.csv.gz")))
        elif dataset:
            # Just dataset filter
            pattern = f"*{dataset}*"
            files_to_load = glob.glob(os.path.join(base_dir, pattern + ".csv"))
            files_to_load.extend(glob.glob(os.path.join(base_dir, pattern + ".csv.gz")))
        else:
            # No filters, load all files (but be careful, could be many)
            files_to_load = glob.glob(os.path.join(base_dir, "*.csv"))
            files_to_load.extend(glob.glob(os.path.join(base_dir, "*.csv.gz")))
    
    # Load and combine all files
    if not files_to_load:
        print(f"No matching files found in {base_dir}")
        return None
    
    # Sort files to ensure chunks are loaded in order
    files_to_load = sorted(files_to_load)
    
    print(f"Loading {len(files_to_load)} files:")
    dfs = []
    
    for file_path in files_to_load:
        print(f"  - {os.path.basename(file_path)}")
        try:
            # Check if file is compressed
            if file_path.endswith('.gz'):
                df = pd.read_csv(file_path, compression='gzip')
            else:
                df = pd.read_csv(file_path)
                
            dfs.append(df)
        except Exception as e:
            print(f"    Error loading {file_path}: {str(e)}")
    
    if not dfs:
        return None
        
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows with {combined_df.shape[1]} columns")
    
    return combined_df

# %%
# Specify the directory where feature files are stored
feature_dir = "outputs/features"  # Update this to your directory

# Load individual feature types
print("Loading question features...")
question_features = load_feature_files(feature_dir, 'question')

print("\nLoading SQL features...")
sql_features = load_feature_files(feature_dir, 'sql')

print("\nLoading schema features...")
schema_features = load_feature_files(feature_dir, 'schema')

# Load dataset-specific features if needed
# bird_specific = load_feature_files(feature_dir, dataset='bird')
# spider_specific = load_feature_files(feature_dir, dataset='spider')

# Try to load the difficulty distribution if available
difficulty_dist = load_feature_files(feature_dir, 'difficulty_distribution')

# You can also load the combined features if needed for specific analyses
# Note: This might be memory-intensive, so we'll avoid it by default
# combined_features = load_feature_files(feature_dir, 'combined')

# %% [markdown]
# ## 2. Dataset Overview
# 
# Let's start by examining the datasets and their basic statistics.

# %%
# Dataset overview
if question_features is not None:
    # Extract dataset info from question features
    dataset_counts = question_features['dataset'].value_counts()
    print("Questions per dataset:")
    print(dataset_counts)
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(dataset_counts, labels=dataset_counts.index, autopct='%1.1f%%', 
            shadow=True, startangle=90, explode=[0.05] * len(dataset_counts))
    plt.title('Distribution of Questions by Dataset')
    plt.axis('equal')
    plt.show()
    
    # Difficulty distribution if available
    if 'difficulty' in question_features.columns:
        # Handle unknown difficulties
        question_features['difficulty'].fillna('unknown', inplace=True)
        question_features.loc[question_features['difficulty'] == '', 'difficulty'] = 'unknown'
        
        difficulty_counts = question_features.groupby(['dataset', 'difficulty']).size().unstack(fill_value=0)
        
        # Plot difficulty distribution
        ax = difficulty_counts.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title('Question Difficulty Distribution by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Number of Questions')
        plt.legend(title='Difficulty')
        
        # Add count labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
            
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 3. Question Feature Analysis
# 
# Let's analyze the question features to understand the characteristics of natural language questions.

# %%
# Basic statistics for question features
if question_features is not None:
    print("Question Features Statistics:")
    
    # Select numeric columns
    numeric_cols = question_features.select_dtypes(include=['int', 'float']).columns
    
    # Calculate statistics
    stats_df = question_features[numeric_cols].describe().T
    stats_df['missing'] = question_features[numeric_cols].isna().sum()
    stats_df['missing_pct'] = (question_features[numeric_cols].isna().sum() / len(question_features)) * 100
    
    # Display statistics for key columns
    key_cols = [col for col in numeric_cols if any(x in col for x in 
                                                 ['length', 'count', 'similarity', 'overlap', 'entity'])]
    print(stats_df.loc[key_cols].sort_values('mean', ascending=False).head(15))
    
    # Plot distribution of question lengths
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=question_features, x='q_char_length', hue='dataset', bins=30, kde=True, element='step')
    plt.title('Distribution of Question Character Lengths')
    plt.xlabel('Character Length')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=question_features, x='q_word_length', hue='dataset', bins=20, kde=True, element='step')
    plt.title('Distribution of Question Word Counts')
    plt.xlabel('Word Count')
    
    plt.tight_layout()
    plt.show()

# %%
# Entity analysis
if question_features is not None:
    # Entity type columns
    entity_cols = [col for col in question_features.columns if col.startswith('q_has_') and col != 'q_has_entities']
    
    # Calculate percentage of questions with each entity type
    entity_pcts = question_features[entity_cols].mean() * 100
    
    # Plot entity type distribution
    plt.figure(figsize=(12, 6))
    entity_pcts.sort_values(ascending=False).plot(kind='bar')
    plt.title('Percentage of Questions with Different Entity Types')
    plt.xlabel('Entity Type')
    plt.ylabel('Percentage of Questions')
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, v in enumerate(entity_pcts.sort_values(ascending=False)):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
    plt.tight_layout()
    plt.show()
    
    # Entity analysis by dataset
    entity_by_dataset = question_features.groupby('dataset')[entity_cols].mean() * 100
    
    # Plot entity types by dataset
    plt.figure(figsize=(14, 8))
    entity_by_dataset.T.plot(kind='bar')
    plt.title('Entity Types by Dataset')
    plt.xlabel('Entity Type')
    plt.ylabel('Percentage of Questions')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()

# %%
# Schema overlap analysis
if question_features is not None:
    # Check if we have schema overlap features
    overlap_cols = [col for col in question_features.columns if 'overlap' in col or 'similarity' in col]
    
    if overlap_cols:
        # Plot average similarities and overlaps
        plt.figure(figsize=(12, 6))
        
        # Calculate average by dataset
        overlap_by_dataset = question_features.groupby('dataset')[overlap_cols].mean()
        
        # Plot
        overlap_by_dataset.plot(kind='bar')
        plt.title('Schema Overlap Metrics by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Average Value')
        plt.xticks(rotation=0)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap for overlap features
        plt.figure(figsize=(10, 8))
        correlation = question_features[overlap_cols].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
        plt.title('Correlation Between Schema Overlap Metrics')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 4. SQL Query Analysis
# 
# Now let's analyze the SQL query features to understand their characteristics.

# %%
# SQL analysis
if sql_features is not None:
    # Select key SQL features
    sql_cols = ['sql_tables_count', 'sql_join_count', 'sql_where_conditions', 
                'sql_subquery_count', 'sql_clauses_count', 'sql_agg_function_count',
                'sql_select_columns', 'sql_char_length']
    
    # Make sure all columns exist
    sql_cols = [col for col in sql_cols if col in sql_features.columns]
    
    if sql_cols:
        # Basic statistics
        print("SQL Query Statistics:")
        print(sql_features[sql_cols].describe().T)
        
        # Plot distributions
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(sql_cols[:8], 1):  # Limit to first 8 columns
            plt.subplot(2, 4, i)
            sns.histplot(data=sql_features, x=col, hue='dataset', kde=True, element='step')
            plt.title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.show()
        
        # SQL clause usage
        sql_clause_cols = [col for col in sql_features.columns if col.startswith('sql_has_')]
        
        if sql_clause_cols:
            # Calculate percentage of queries with each clause
            clause_pcts = sql_features[sql_clause_cols].mean() * 100
            
            # Plot
            plt.figure(figsize=(12, 6))
            clause_pcts.sort_values(ascending=False).plot(kind='bar')
            plt.title('Percentage of Queries with Different SQL Clauses')
            plt.xlabel('SQL Clause')
            plt.ylabel('Percentage of Queries')
            plt.xticks(rotation=45)
            
            # Add percentage labels
            for i, v in enumerate(clause_pcts.sort_values(ascending=False)):
                plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
                
            plt.tight_layout()
            plt.show()
            
            # Clause usage by dataset
            clause_by_dataset = sql_features.groupby('dataset')[sql_clause_cols].mean() * 100
            
            # Plot
            plt.figure(figsize=(14, 8))
            clause_by_dataset.T.plot(kind='bar')
            plt.title('SQL Clause Usage by Dataset')
            plt.xlabel('SQL Clause')
            plt.ylabel('Percentage of Queries')
            plt.xticks(rotation=45)
            plt.legend(title='Dataset')
            plt.tight_layout()
            plt.show()

# %% [markdown]
# ## 5. Cross-Dataset Feature Comparison
# 
# Let's compare key features across datasets using the individual feature DataFrames.

# %%
# Dataset comparison using individual feature DataFrames
if question_features is not None and sql_features is not None:
    # First, we need to create a mini-join for just the comparison
    # This is much smaller than loading the entire combined DataFrame

    # Get means of question features by dataset
    q_comparison = question_features.groupby('dataset')[
        ['q_word_length', 'q_entity_count', 'q_number_count']
    ].mean().reset_index()
    
    # Get means of SQL features by dataset
    sql_comparison = sql_features.groupby('dataset')[
        ['sql_tables_count', 'sql_join_count', 'sql_where_conditions', 'sql_agg_function_count']
    ].mean().reset_index()
    
    # Merge just these aggregated statistics
    # This is a very small DataFrame compared to merging all the raw data
    comparison_df = pd.merge(q_comparison, sql_comparison, on='dataset')
    
    # Plot aggregated features side-by-side
    plt.figure(figsize=(14, 8))
    
    # Transpose to make datasets the columns and features the rows
    comparison_df.set_index('dataset').T.plot(kind='bar')
    
    plt.title('Average Feature Values by Dataset')
    plt.xlabel('Feature')
    plt.ylabel('Average Value')
    plt.legend(title='Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Bar chart for question features by dataset
    plt.figure(figsize=(10, 6))
    q_comparison.set_index('dataset').plot(kind='bar')
    plt.title('Question Features by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.legend(title='Feature')
    plt.tight_layout()
    plt.show()
    
    # Bar chart for SQL features by dataset
    plt.figure(figsize=(10, 6))
    sql_comparison.set_index('dataset').plot(kind='bar')
    plt.title('SQL Features by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.legend(title='Feature')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Difficulty Analysis (BIRD Dataset)
# 
# Analyze question difficulty in the BIRD dataset.

# %%
# Difficulty analysis for BIRD dataset
# Use the bird_specific features if available, otherwise filter question_features
bird_data = bird_specific if bird_specific is not None else question_features[question_features['dataset'] == 'bird']

if bird_data is not None and 'difficulty' in bird_data.columns:
    # Get difficulty distribution
    bird_data['difficulty'].fillna('unknown', inplace=True)
    difficulty_counts = bird_data['difficulty'].value_counts()
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = difficulty_counts.plot(kind='bar')
    plt.title('BIRD Dataset: Question Difficulty Distribution')
    plt.xlabel('Difficulty')
    plt.ylabel('Count')
    
    # Add count labels
    for i, v in enumerate(difficulty_counts):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Check if we have the difficulty distribution data
    if difficulty_dist is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=difficulty_dist, x='difficulty', y='percentage')
        plt.title('BIRD Dataset: Question Difficulty Percentage')
        plt.xlabel('Difficulty')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        
        # Add percentage labels
        for i, row in enumerate(difficulty_dist.itertuples()):
            plt.text(i, row.percentage + 0.5, f'{row.percentage:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    # Feature distribution by difficulty - using only question features
    if 'q_word_length' in bird_data.columns:
        # Select available features
        difficulty_feature_cols = [col for col in 
                                ['q_word_length', 'q_entity_count'] 
                                if col in bird_data.columns]
        
        if difficulty_feature_cols:
            # Create boxplots for each feature by difficulty
            plt.figure(figsize=(12, 8))
            for i, col in enumerate(difficulty_feature_cols, 1):
                plt.subplot(1, len(difficulty_feature_cols), i)
                sns.boxplot(data=bird_data, x='difficulty', y=col)
                plt.title(f'{col} by Difficulty')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    # Now do the same for SQL features if available
    bird_sql = sql_features[sql_features['dataset'] == 'bird'] if sql_features is not None else None
    
    if bird_sql is not None and 'difficulty' in bird_sql.columns:
        # Select available features
        sql_difficulty_cols = [col for col in 
                             ['sql_tables_count', 'sql_join_count', 'sql_where_conditions'] 
                             if col in bird_sql.columns]
        
        if sql_difficulty_cols:
            # Create boxplots for each feature by difficulty
            plt.figure(figsize=(16, 6))
            for i, col in enumerate(sql_difficulty_cols, 1):
                plt.subplot(1, len(sql_difficulty_cols), i)
                sns.boxplot(data=bird_sql, x='difficulty', y=col)
                plt.title(f'{col} by Difficulty')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()

# %% [markdown]
# ## 7. SQL Characteristics Analysis
# 
# Let's analyze the SQL query characteristics directly from the SQL features DataFrame.

# %%
# Create a smaller feature set for analysis by merging a subset of features
if question_features is not None and sql_features is not None:
    # Get just the columns we need for this analysis
    # Using question_id as the join key
    mini_q_features = question_features[['question_id', 'dataset', 'q_word_length']].copy()
    mini_sql_features = sql_features[['question_id', 'sql_tables_count', 'sql_join_count', 
                                     'sql_agg_function_count']].copy()
    
    # Merge to create a smaller analysis DataFrame
    # This is much smaller than the full combined DataFrame
    mini_analysis = pd.merge(mini_q_features, mini_sql_features, on='question_id')
    
    # Calculate correlation between question length and SQL complexity
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=mini_analysis, x='q_word_length', y='sql_tables_count', 
                    hue='dataset', alpha=0.6)
    plt.title('Question Length vs. Tables Used in SQL')
    plt.xlabel('Question Word Length')
    plt.ylabel('Number of Tables in SQL')
    plt.tight_layout()
    plt.show()
    
    # Calculate correlations for this mini dataset
    corr = mini_analysis.drop(['question_id', 'dataset'], axis=1).corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Question and SQL Features')
    plt.tight_layout()
    plt.show()

# Analyze SQL clause usage
if sql_features is not None:
    # SQL clause usage
    sql_clause_cols = [col for col in sql_features.columns if col.startswith('sql_has_')]
    
    if sql_clause_cols:
        # Calculate percentage of queries with each clause
        clause_pcts = sql_features[sql_clause_cols].mean() * 100
        
        # Plot
        plt.figure(figsize=(12, 6))
        clause_pcts.sort_values(ascending=False).plot(kind='bar')
        plt.title('Percentage of Queries with Different SQL Clauses')
        plt.xlabel('SQL Clause')
        plt.ylabel('Percentage of Queries')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        for i, v in enumerate(clause_pcts.sort_values(ascending=False)):
            plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
            
        plt.tight_layout()
        plt.show()

    # Plot distribution of tables used
    if 'sql_tables_count' in sql_features.columns:
        plt.figure(figsize=(10, 6))
        
        # Create a histogram with KDE
        sns.histplot(data=sql_features, x='sql_tables_count', hue='dataset', kde=True,
                     element='step', bins=max(5, sql_features['sql_tables_count'].max()))
        
        plt.title('Distribution of Tables Used in SQL Queries')
        plt.xlabel('Number of Tables')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 8. Correlation Analysis
# 
# Let's analyze correlations within each feature set separately.

# %%
# Question feature correlations
if question_features is not None:
    # Select important numeric features for correlation
    q_corr_cols = [col for col in question_features.columns 
                  if col.startswith('q_') and 
                  question_features[col].dtype in ['int64', 'float64']]
    
    if len(q_corr_cols) > 1:  # Need at least 2 columns for correlation
        # Calculate correlation
        q_correlation = question_features[q_corr_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(q_correlation)
        sns.heatmap(q_correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   mask=mask, vmin=-1, vmax=1)
        plt.title('Question Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

# SQL feature correlations
if sql_features is not None:
    # Select important numeric features for correlation
    sql_corr_cols = [col for col in sql_features.columns 
                    if col.startswith('sql_') and 
                    sql_features[col].dtype in ['int64', 'float64']]
    
    if len(sql_corr_cols) > 1:  # Need at least 2 columns for correlation
        # Calculate correlation
        sql_correlation = sql_features[sql_corr_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(sql_correlation)
        sns.heatmap(sql_correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   mask=mask, vmin=-1, vmax=1)
        plt.title('SQL Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

# Cross-feature correlations using mini-merged dataset
if question_features is not None and sql_features is not None:
    # Create mini dataset with key features for correlation analysis
    q_mini = question_features[['question_id', 'q_word_length', 'q_entity_count']].copy()
    sql_mini = sql_features[['question_id', 'sql_tables_count', 'sql_join_count', 
                            'sql_where_conditions']].copy()
    
    # Merge on question_id
    mini_corr_df = pd.merge(q_mini, sql_mini, on='question_id')
    
    # Drop the ID column for correlation
    mini_corr_df = mini_corr_df.drop('question_id', axis=1)
    
    # Calculate correlation
    cross_correlation = mini_corr_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_correlation, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Cross-Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Sample Analysis
# 
# Let's examine some examples from each dataset using lightweight selections.

# %%
# Sample analysis using separate feature DataFrames
if question_features is not None:
    # Function to get sample questions
    def get_samples(df, dataset, n=5):
        samples = df[df['dataset'] == dataset].sample(min(n, df[df['dataset'] == dataset].shape[0]))
        return samples
    
    # Get samples from each dataset
    datasets = question_features['dataset'].unique()
    
    for dataset in datasets:
        print(f"\n=== Sample Questions from {dataset.upper()} (Question Features) ===")
        q_samples = get_samples(question_features, dataset)
        
        # Select columns to display
        q_display_cols = ['question_id', 'difficulty', 'q_word_length', 'q_entity_count']
        
        # Make sure all columns exist
        q_display_cols = [col for col in q_display_cols if col in q_samples.columns]
        
        if q_display_cols:
            display_df = q_samples[q_display_cols].reset_index(drop=True)
            print(display_df)

if sql_features is not None:
    # Get the same question_ids from sql features
    for dataset in datasets:
        print(f"\n=== Sample Questions from {dataset.upper()} (SQL Features) ===")
        sql_samples = get_samples(sql_features, dataset)
        
        # Select columns to display
        sql_display_cols = ['question_id', 'sql_tables_count', 'sql_join_count']
        
        # Make sure all columns exist
        sql_display_cols = [col for col in sql_display_cols if col in sql_samples.columns]
        
        if sql_display_cols:
            display_df = sql_samples[sql_display_cols].reset_index(drop=True)
            print(display_df)

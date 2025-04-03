# Text2SQL Feature Analysis and Visualization
# This notebook analyzes the outputs from the Text2SQL feature engineering module
# %%
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
    Load feature files from the specified directory.
    
    Args:
        base_dir: Base directory containing feature files
        feature_type: Optional type of features to load (e.g., 'question', 'sql')
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
            
            # Apply filters
            if feature_type and f"_{feature_type}" not in file_name and not file_name.startswith(feature_type):
                continue
                
            if dataset and f"{dataset}" not in file_name:
                continue
                
            if os.path.exists(file_path):
                files_to_load.append(file_path)
            else:
                # Try relative path
                rel_path = os.path.join(base_dir, os.path.basename(file_path))
                if os.path.exists(rel_path):
                    files_to_load.append(rel_path)
    else:
        # No manifest, find files manually
        pattern = "*"
        if feature_type:
            pattern = f"*_{feature_type}*"
        if dataset:
            pattern = f"*{dataset}*_{feature_type}*" if feature_type else f"*{dataset}*"
            
        # Find all CSV files matching the pattern
        files_to_load = glob.glob(os.path.join(base_dir, pattern + ".csv"))
        
        # Also check for compressed files
        files_to_load.extend(glob.glob(os.path.join(base_dir, pattern + ".csv.gz")))
    
    # Load and combine all files
    if not files_to_load:
        print(f"No matching files found in {base_dir}")
        return None
        
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
feature_dir = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/outputs/test_multi/features"  # Update this to your directory

# Load different feature types
question_features = load_feature_files(feature_dir, 'question')
sql_features = load_feature_files(feature_dir, 'sql')
combined_features = load_feature_files(feature_dir, 'combined')

# Load dataset-specific features
bird_features = load_feature_files(feature_dir, dataset='bird')
spider_features = load_feature_files(feature_dir, dataset='spider')

# Try to load the difficulty distribution if available
difficulty_dist = load_feature_files(feature_dir, 'difficulty_distribution')

# %% [markdown]
# ## 2. Dataset Overview
# 
# Let's start by examining the datasets and their basic statistics.

# %%
# Dataset overview
if combined_features is not None:
    # Count questions by dataset
    dataset_counts = combined_features['dataset'].value_counts()
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
    if 'difficulty' in combined_features.columns:
        # Handle unknown difficulties
        combined_features['difficulty'].fillna('unknown', inplace=True)
        combined_features.loc[combined_features['difficulty'] == '', 'difficulty'] = 'unknown'
        
        difficulty_counts = combined_features.groupby(['dataset', 'difficulty']).size().unstack(fill_value=0)
        
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
# ## 5. Dataset Comparison
# 
# Let's compare key features across datasets to understand their differences.

# %%
# Dataset comparison
if combined_features is not None:
    # Select key comparison features
    comparison_cols = [
        'q_word_length', 'q_entity_count', 'q_number_count',
        'sql_tables_count', 'sql_join_count', 'sql_where_conditions',
        'sql_agg_function_count', 'q_avg_table_similarity'
    ]
    
    # Make sure all columns exist
    comparison_cols = [col for col in comparison_cols if col in combined_features.columns]
    
    if comparison_cols:
        # Calculate average by dataset
        comparison_by_dataset = combined_features.groupby('dataset')[comparison_cols].mean()
        
        # Normalize for radar chart
        # Get min and max for each column
        mins = comparison_by_dataset.min()
        maxs = comparison_by_dataset.max()
        
        # Normalize to 0-1 range
        radar_df = (comparison_by_dataset - mins) / (maxs - mins)
        
        # Create radar chart
        datasets = radar_df.index
        num_features = len(comparison_cols)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
        
        # Add feature labels
        plt.xticks(angles[:-1], comparison_cols, size=12)
        
        # Plot each dataset
        for i, dataset in enumerate(datasets):
            values = radar_df.loc[dataset].values.tolist()
            values += values[:1]  # Close the circle
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=dataset)
            ax.fill(angles, values, alpha=0.1)
        
        plt.title('Dataset Feature Comparison', size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()
        
        # Bar chart comparison for key metrics
        plt.figure(figsize=(14, 8))
        comparison_by_dataset.plot(kind='bar')
        plt.title('Average Feature Values by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Average Value')
        plt.xticks(rotation=0)
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 6. Difficulty Analysis (BIRD Dataset)
# 
# Analyze question difficulty in the BIRD dataset.

# %%
# Difficulty analysis for BIRD dataset
if bird_features is not None and 'difficulty' in bird_features.columns:
    # Get difficulty distribution
    bird_features['difficulty'].fillna('unknown', inplace=True)
    difficulty_counts = bird_features['difficulty'].value_counts()
    
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
    
    # Feature distribution by difficulty
    # Select key features
    difficulty_feature_cols = [
        'q_word_length', 'q_entity_count', 
        'sql_tables_count', 'sql_join_count', 'sql_where_conditions'
    ]
    
    # Make sure all columns exist
    difficulty_feature_cols = [col for col in difficulty_feature_cols if col in bird_features.columns]
    
    if difficulty_feature_cols:
        # Create boxplots for each feature by difficulty
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(difficulty_feature_cols, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=bird_features, x='difficulty', y=col)
            plt.title(f'{col} by Difficulty')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 7. SQL Characteristics by Question Type
# 
# Let's analyze how SQL query characteristics vary with different question types.

# %%
# Function to categorize questions based on content
def categorize_question(question):
    question = question.lower() if isinstance(question, str) else ""
    
    categories = {
        'counting': ['how many', 'count', 'number of'],
        'comparison': ['more than', 'less than', 'greater', 'highest', 'lowest', 'maximum', 'minimum'],
        'aggregation': ['average', 'mean', 'total', 'sum'],
        'filtering': ['where', 'which', 'find', 'list'],
        'grouping': ['group', 'by each', 'for each'],
        'sorting': ['order', 'sort', 'rank']
    }
    
    for category, keywords in categories.items():
        if any(keyword in question for keyword in keywords):
            return category
    
    return 'other'

# Categorize questions if we have the text
if combined_features is not None and 'question' in combined_features.columns:
    combined_features['question_category'] = combined_features['question'].apply(categorize_question)
    
    # Count by category
    category_counts = combined_features['question_category'].value_counts()
    
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar')
    plt.title('Questions by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels
    for i, v in enumerate(category_counts):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # SQL characteristics by question category
    sql_by_category_cols = [
        'sql_tables_count', 'sql_join_count', 'sql_where_conditions',
        'sql_has_group_by', 'sql_has_order_by', 'sql_agg_function_count'
    ]
    
    # Make sure all columns exist
    sql_by_category_cols = [col for col in sql_by_category_cols if col in combined_features.columns]
    
    if sql_by_category_cols:
        # Calculate mean values by category
        sql_by_category = combined_features.groupby('question_category')[sql_by_category_cols].mean()
        
        # Convert boolean columns to percentages
        bool_cols = [col for col in sql_by_category_cols if combined_features[col].dtype == 'bool']
        if bool_cols:
            sql_by_category[bool_cols] = sql_by_category[bool_cols] * 100
        
        # Plot
        plt.figure(figsize=(14, 8))
        sql_by_category.plot(kind='bar')
        plt.title('SQL Characteristics by Question Category')
        plt.xlabel('Question Category')
        plt.ylabel('Average Value')
        plt.xticks(rotation=45)
        plt.legend(title='SQL Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 8. Correlation Analysis
# 
# Let's analyze correlations between different features to understand relationships.

# %%
# Correlation analysis
if combined_features is not None:
    # Select important features for correlation
    corr_cols = [
        'q_word_length', 'q_entity_count', 'q_number_count', 
        'sql_tables_count', 'sql_join_count', 'sql_where_conditions',
        'sql_subquery_count', 'sql_clauses_count', 'sql_agg_function_count',
        'sql_select_columns', 'sql_char_length'
    ]
    
    # Make sure all columns exist
    corr_cols = [col for col in corr_cols if col in combined_features.columns]
    
    if corr_cols:
        # Calculate correlation
        correlation = combined_features[corr_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 9. Sample Analysis
# 
# Let's examine some examples from each dataset.

# %%
# Sample analysis
if combined_features is not None:
    # Function to get sample questions
    def get_samples(df, dataset, n=5):
        samples = df[df['dataset'] == dataset].sample(min(n, df[df['dataset'] == dataset].shape[0]))
        return samples
    
    # Get samples from each dataset
    datasets = combined_features['dataset'].unique()
    
    for dataset in datasets:
        print(f"\n=== Sample Questions from {dataset.upper()} ===")
        samples = get_samples(combined_features, dataset)
        
        # Select columns to display
        display_cols = ['question_id', 'difficulty', 'q_word_length', 
                        'sql_tables_count', 'sql_join_count']
        
        # Make sure all columns exist
        display_cols = [col for col in display_cols if col in samples.columns]
        
        if display_cols:
            print(samples[display_cols])

# %% [markdown]
# ## 10. Key Findings
# 
# Based on our analysis, here are some key observations about the Text2SQL datasets:
# 
# 1. **Dataset Composition**:
#    - Distribution between BIRD and Spider datasets
#    - BIRD provides difficulty labels while Spider does not
# 
# 2. **Question Characteristics**:
#    - Average question length in each dataset
#    - Most common entity types across datasets
#    - Question complexity patterns
# 
# 3. **SQL Query Patterns**:
#    - Most commonly used SQL clauses
#    - Join and table usage differences between datasets
#    - Correlation between question features and SQL complexity
# 
# 4. **NL-to-SQL Relationship**:
#    - How question types correlate with SQL structure
#    - Entity presence and its relationship with WHERE clauses
#    - Table/column mentions in natural language questions
# 
# 5. **Schema Interaction**:
#    - How questions relate to database schemas
#    - Overlap between question terms and schema elements

# %%

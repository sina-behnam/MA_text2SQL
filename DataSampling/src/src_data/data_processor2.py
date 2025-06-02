import re
from typing import List, Dict
import os
import sqlite3
import pandas as pd
import csv
import logging
import spacy
from tqdm import tqdm
import json
import sqlparse

# for the brid dataset
from bird_dataloader import (
        load_bird_instances,
        process_bird_schemas,
        get_bird_dataset_files
    )
# for the spider dataset
from spider1_dataloader import (
        load_spider_instances,
        process_spider_schemas,
        get_spider_dataset_files
    )

# Adding the utiles directory to the path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils import read_ddl_schema_csv, clean_ddl_column

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def analyze_question(question, schema=None):
    
    doc = nlp(question)

    # Basic statistics
    char_length = len(question)
    word_length = len([token for token in doc if not token.is_punct and not token.is_space])
    
    # Entity presence using NER
    entities = []
    entity_types = set()
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
        entity_types.add(ent.label_)
    
    # Number presence
    numbers = []
    for token in doc:
        if token.like_num:
            numbers.append(token.text)
    
    # Negation presence
    has_negation = any(token.dep_ == 'neg' for token in doc)
    negation_words = [token.text for token in doc if token.dep_ == 'neg']
    
    # Comparatives/superlatives
    comparatives = []
    superlatives = []
    for token in doc:
        if token.tag_ == 'JJR' or token.tag_ == 'RBR':
            comparatives.append(token.text)
        elif token.tag_ == 'JJS' or token.tag_ == 'RBS':
            superlatives.append(token.text)
    
    # Prepare result
    result = {
        'char_length': char_length,
        'word_length': word_length,
        'entities': entities,
        'entity_types': list(entity_types),
        'has_entities': len(entities) > 0,
        'numbers': numbers,
        'has_numbers': len(numbers) > 0,
        'has_negation': has_negation,
        'negation_words': negation_words,
        'comparatives': comparatives,
        'has_comparatives': len(comparatives) > 0,
        'superlatives': superlatives,
        'has_superlatives': len(superlatives) > 0
    }

    # If schema is provided, analyze overlap
    if schema is not None:
        overlap_result = analyze_overlap_q_schema(doc, schema)
        result.update(overlap_result)
    
    return result

def analyze_sql(sql_query):
    """
    Analyze SQL query.
    
    Args:
        sql_query: The SQL query to analyze
        
    Returns:
        Dictionary with SQL analysis
    """
    # Basic statistics
    char_length = len(sql_query)
    
    # Parse SQL using sqlparse
    parsed = sqlparse.parse(sql_query)
    
    if not parsed:
        return {
            'char_length': char_length,
            'error': 'Failed to parse SQL query'
        }
    
    # Get the first statement
    stmt = parsed[0]
    
    # Extract tables from FROM and JOIN clauses
    from_pattern = r'\bFROM\s+([^\s,]+)'
    join_pattern = r'\bJOIN\s+([^\s]+)'
    
    tables_from = re.findall(from_pattern, sql_query, re.IGNORECASE)
    tables_join = re.findall(join_pattern, sql_query, re.IGNORECASE)
    
    # Clean up table names (remove aliases and quotes)
    def clean_table_name(name):
        # Remove AS alias
        name = re.sub(r'\s+AS\s+\w+', '', name, flags=re.IGNORECASE)
        # Remove quotes
        return name.strip('`"\'[]')
    
    tables_from = [clean_table_name(t) for t in tables_from]
    tables_join = [clean_table_name(t) for t in tables_join]
    
    # Combine tables from both sources
    tables = list(set(tables_from + tables_join))
    
    # Count joins
    join_count = sql_query.upper().count('JOIN')
    
    # Count WHERE conditions
    where_conditions = 0
    if 'WHERE' in sql_query.upper():
        # Extract WHERE clause
        where_clause = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|HAVING|$)', 
                                sql_query, re.IGNORECASE | re.DOTALL)
        if where_clause:
            where_text = where_clause.group(1).strip()
            # Count AND/OR operators
            where_conditions = where_text.upper().count(' AND ') + where_text.upper().count(' OR ') + 1
    
    # Count subqueries (approximation)
    subquery_count = sql_query.upper().count('SELECT') - 1
    if subquery_count < 0:
        subquery_count = 0
    
    # Count clauses
    clauses = 0
    clause_types = []
    
    if 'GROUP BY' in sql_query.upper():
        clauses += 1
        clause_types.append('GROUP BY')
        
    if 'HAVING' in sql_query.upper():
        clauses += 1
        clause_types.append('HAVING')
        
    if 'ORDER BY' in sql_query.upper():
        clauses += 1
        clause_types.append('ORDER BY')
        
    if 'LIMIT' in sql_query.upper():
        clauses += 1
        clause_types.append('LIMIT')
    
    # Count aggregation functions
    agg_functions = ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX']
    agg_function_count = 0
    found_agg_functions = []
    
    for func in agg_functions:
        # Look for the function in SQL
        pattern = r'\b' + func + r'\s*\('
        matches = re.findall(pattern, sql_query, re.IGNORECASE)
        if matches:
            agg_function_count += len(matches)
            found_agg_functions.append(func)
    
    # Count selected columns
    selected_columns = 0
    
    # Find the SELECT clause
    select_clause = ""
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
    
    if select_match:
        select_clause = select_match.group(1).strip()
        
        # Handle SELECT *
        if select_clause == '*':
            selected_columns = 1  # Count * as one column
        else:
            # Handle functions and complex expressions
            # This is a simplification - won't work for all cases
            in_function = 0
            parts = []
            current_part = ""
            
            for char in select_clause:
                if char == '(':
                    in_function += 1
                    current_part += char
                elif char == ')':
                    in_function -= 1
                    current_part += char
                elif char == ',' and in_function == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            if current_part:
                parts.append(current_part.strip())
            
            selected_columns = len(parts)
    
    # Prepare result
    result = {
        'char_length': char_length,
        'tables_count': len(tables),
        'tables': tables,
        'join_count': join_count,
        'where_conditions': where_conditions,
        'subquery_count': subquery_count,
        'clauses_count': clauses,
        'clause_types': clause_types,
        'aggregation_function_count': agg_function_count,
        'aggregation_functions': found_agg_functions,
        'select_columns': selected_columns
    }
    
    return result

def load_schemas_list(schemas_path : list):
    
    all_concatenated_schemas = pd.DataFrame()
    for schema in schemas_path:
        
        schema_name = schema['name']
        if isinstance(schema['path'], list):
            schema_path = schema['path'][0]  # Assuming the first path is the main one
        else:
            schema_path = schema['path']

        try:
            df = read_ddl_schema_csv(schema_path)
            df = clean_ddl_column(df, 'DDL')
        except Exception as e:
            missing_message = 'missing schema or crupted schema'
            df = pd.DataFrame({'table_name': None,'description' : None,'DDL': missing_message}, index=[0])
            logging.error(f"Error reading schema file {schema_path}: {e}")

        # Add schema name to the DataFrame
        df['schema_name'] = schema_name
        # Add to the all concatenated schemas
        all_concatenated_schemas = pd.concat([all_concatenated_schemas, df], ignore_index=True)

    return all_concatenated_schemas        

def analyze_overlap_q_schema(doc, schema : dict):
    """
    Analyze the overlap between the question and schema.
    """
    # tables = schema['table_name'].values    
    # columns = [column for ddl in sample_shcema['DDL'].tolist() if isinstance(ddl, str) for column in extract_columns_from_ddl(ddl)]
    tables = schema['table_names']
    columns = [column for table_columns in schema['column_names'].values() for column in table_columns]
    
    table_names = {table.lower() for table in tables}
    column_names = {column.lower() for column in columns}
    
    # Descriiption

    question_terms = set()
    question_lemmas = set()

    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            question_terms.add(token.text.lower())
            question_lemmas.add(token.lemma_.lower())

    # Compute direct term overlap
    table_overlap = {table: table.lower() in question_terms 
                     for table in table_names if table}
    table_overlap_lemma = {table: table.lower() in question_lemmas 
                           for table in table_names if table}
    
    column_overlap = {col: col.lower() in question_terms 
                      for col in column_names if col}
    column_overlap_lemma = {col: col.lower() in question_lemmas 
                            for col in column_names if col}
    
    # Count overlaps
    table_overlap_count = sum(1 for match in table_overlap.values() if match)
    table_overlap_lemma_count = sum(1 for match in table_overlap_lemma.values() if match)
    
    column_overlap_count = sum(1 for match in column_overlap.values() if match)
    column_overlap_lemma_count = sum(1 for match in column_overlap_lemma.values() if match)
        
    result = {
        'table_overlap_count': table_overlap_count,
        'table_overlap_lemma_count': table_overlap_lemma_count,
        'column_overlap_count': column_overlap_count,
        'column_overlap_lemma_count': column_overlap_lemma_count,
    }
    
    return result

def parse_sqlite_ddl_statement(ddl: str) -> dict:
    """
    Parse a DDL statement and extract table structure information.
    
    Args:
        ddl (str): A CREATE TABLE DDL statement
        
    Returns:
        Dict containing table name, column names, column types, primary keys, and foreign keys
    """
    # Initialize return structure
    table_name = ""
    columns_names = []
    column_types = {}
    primary_keys = []
    foreign_keys = []
    
    # Extract table name
    table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\w+\.)?([`"\[\]\w]+|\[[^]]+\]|`[^`]+`|"[^"]+")', 
                            ddl, re.IGNORECASE)
    if table_match:
        # Clean up table name
        table_name = re.sub(r'^\[|\]$|^"|"$|^`|`$', '', table_match.group(1))
    
    # Extract column definitions
    columns_match = re.search(r'\((.*)\)', ddl, re.DOTALL)
    if columns_match:
        column_text = columns_match.group(1)
        
        # Parse column definitions
        in_parentheses = 0
        start_pos = 0
        current_pos = 0
        column_defs = []
        
        for char in column_text:
            if char == '(':
                in_parentheses += 1
            elif char == ')':
                in_parentheses -= 1
            elif char == ',' and in_parentheses == 0:
                column_defs.append(column_text[start_pos:current_pos].strip())
                start_pos = current_pos + 1
            current_pos += 1
            
        # Add the last column definition
        if start_pos < len(column_text):
            column_defs.append(column_text[start_pos:].strip())
        
        # Process each column definition
        for col_def in column_defs:
            # Skip empty definitions
            if not col_def:
                continue
                
            # Check for standalone PRIMARY KEY constraint
            pk_match = re.search(r'^\s*PRIMARY\s+KEY\s*\(\s*(.*?)\s*\)', col_def, re.IGNORECASE)
            if pk_match:
                pk_cols = [col.strip() for col in pk_match.group(1).split(',')]
                # Clean column names (remove quotes, brackets)
                pk_cols = [re.sub(r'^\[|\]$|^"|"$|^`|`$', '', col) for col in pk_cols]
                primary_keys.extend(pk_cols)
                continue
                
            # Check for standalone FOREIGN KEY constraint
            fk_match = re.search(
                r'^\s*FOREIGN\s+KEY\s*\(\s*(.*?)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(.*?)\s*\)',
                col_def, re.IGNORECASE
            )
            if fk_match:
                fk_cols = [col.strip() for col in fk_match.group(1).split(',')]
                fk_cols = [re.sub(r'^\[|\]$|^"|"$|^`|`$', '', col) for col in fk_cols]
                ref_table = fk_match.group(2)
                ref_cols = [col.strip() for col in fk_match.group(3).split(',')]
                ref_cols = [re.sub(r'^\[|\]$|^"|"$|^`|`$', '', col) for col in ref_cols]
                
                for i, col in enumerate(fk_cols):
                    ref_col = ref_cols[i] if i < len(ref_cols) else ref_cols[0]
                    foreign_keys.append({
                        'column': col,
                        'references_table': ref_table,
                        'references_column': ref_col
                    })
                continue
                
            # Process regular column definition
            col_parts = re.match(r'([`"\[\]\w]+|\[[^]]+\]|`[^`]+`|"[^"]+")\s+([^\s,]+)', col_def)
            if not col_parts:
                continue
                
            # Extract column name and clean it
            col_name = col_parts.group(1)
            col_name = re.sub(r'^\[|\]$|^"|"$|^`|`$', '', col_name)
            
            # Extract column type
            col_type = col_parts.group(2).upper()
            
            # Add to our data structures
            columns_names.append(col_name)
            column_types[col_name] = col_type
            
            # Check for inline PRIMARY KEY
            if re.search(r'\bPRIMARY\s+KEY\b', col_def, re.IGNORECASE):
                primary_keys.append(col_name)
                
            # Check for inline REFERENCES
            ref_match = re.search(
                r'\bREFERENCES\s+(\w+)\s*\(\s*([`"\[\]\w]+)\s*\)',
                col_def, re.IGNORECASE
            )
            if ref_match:
                ref_table = ref_match.group(1)
                ref_col = re.sub(r'^\[|\]$|^"|"$|^`|`$', '', ref_match.group(2))
                
                foreign_keys.append({
                    'column': col_name,
                    'references_table': ref_table,
                    'references_column': ref_col
                })
    
    # Return the parsed data in the requested format
    return {
        'table_names': [table_name],
        'column_names': {table_name: columns_names},
        'column_types': {table_name: column_types},
        'primary_keys': {table_name: primary_keys},
        'foreign_keys': {table_name: foreign_keys},
    }

def parse_snowflake_ddl(ddl: str) -> dict:
    """
    Parse Snowflake DDL statement and extract table structure information.
    
    Args:
        ddl (str): A CREATE TABLE DDL statement from Snowflake
        
    Returns:
        Dict containing table name, column names, column types, primary keys, and foreign keys
    """
    # Initialize return structure
    table_name = ""
    columns_names = []
    column_types = {}
    primary_keys = []
    foreign_keys = []
    
    # Extract the full table identifier, which can have multiple parts
    # Snowflake table format can be: DATABASE.SCHEMA.TABLE or even more complex parts
    table_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TRANSIENT|TEMPORARY\s+)?\s*TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)', 
                           ddl, re.IGNORECASE)
    
    if table_match:
        # Extract the fully qualified name
        full_table_name = table_match.group(1)
        
        # Split on dots and take the last part as the table name
        parts = full_table_name.split('.')
        table_name = parts[-1]
        
        # For debugging
        # print(f"Full table name: {full_table_name}")
        # print(f"Extracted table name: {table_name}")
    
    # Extract column definitions
    columns_match = re.search(r'\(\s*(.*?)\s*\)\s*;?', ddl, re.DOTALL)
    if columns_match:
        column_text = columns_match.group(1)
        
        # Parse column definitions
        in_parentheses = 0
        start_pos = 0
        current_pos = 0
        column_defs = []
        
        for char in column_text:
            if char == '(':
                in_parentheses += 1
            elif char == ')':
                in_parentheses -= 1
            elif char == ',' and in_parentheses == 0:
                column_defs.append(column_text[start_pos:current_pos].strip())
                start_pos = current_pos + 1
            current_pos += 1
            
        # Add the last column definition
        if start_pos < len(column_text):
            column_defs.append(column_text[start_pos:].strip())
        
        # Process each column definition
        for col_def in column_defs:
            # Skip empty definitions
            if not col_def:
                continue
                
            # Check for standalone PRIMARY KEY constraint
            pk_match = re.search(r'^\s*(?:CONSTRAINT\s+\w+\s+)?PRIMARY\s+KEY\s*\(\s*(.*?)\s*\)', col_def, re.IGNORECASE)
            if pk_match:
                pk_cols = [col.strip() for col in pk_match.group(1).split(',')]
                # Clean column names (remove quotes)
                pk_cols = [re.sub(r'^"|"$', '', col) for col in pk_cols]
                primary_keys.extend(pk_cols)
                continue
                
            # Check for standalone FOREIGN KEY constraint
            fk_match = re.search(
                r'^\s*(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(\s*(.*?)\s*\)\s*REFERENCES\s+([^\s(]+)\s*\(\s*(.*?)\s*\)',
                col_def, re.IGNORECASE
            )
            if fk_match:
                fk_cols = [col.strip() for col in fk_match.group(1).split(',')]
                fk_cols = [re.sub(r'^"|"$', '', col) for col in fk_cols]
                
                # Handle the referenced table which might have multiple parts
                ref_full_table = fk_match.group(2)
                ref_parts = ref_full_table.split('.')
                ref_table = ref_parts[-1]
                ref_schema = ".".join(ref_parts[:-1]) if len(ref_parts) > 1 else None
                
                ref_cols = [col.strip() for col in fk_match.group(3).split(',')]
                ref_cols = [re.sub(r'^"|"$', '', col) for col in ref_cols]
                
                for i, col in enumerate(fk_cols):
                    ref_col = ref_cols[i] if i < len(ref_cols) else ref_cols[0]
                    foreign_keys.append({
                        'column': col,
                        'references_table': ref_table,
                        'references_column': ref_col,
                        'references_schema': ref_schema
                    })
                continue
            
            # Process regular column definition
            # Format: column_name TYPE [constraints]
            col_match = re.match(r'^\s*([^\s]+)\s+([A-Za-z0-9_]+(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?)', col_def)
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2).upper()
                
                # Add to our data structures
                columns_names.append(col_name)
                column_types[col_name] = col_type
                
                # Check for inline PRIMARY KEY
                if re.search(r'\bPRIMARY\s+KEY\b', col_def, re.IGNORECASE):
                    primary_keys.append(col_name)
                    
                # Check for inline REFERENCES
                ref_match = re.search(
                    r'\bREFERENCES\s+([^\s(]+)\s*\(\s*([^\s)]+)\s*\)',
                    col_def, re.IGNORECASE
                )
                if ref_match:
                    # Handle the referenced table which might have multiple parts
                    ref_full_table = ref_match.group(1)
                    ref_parts = ref_full_table.split('.')
                    ref_table = ref_parts[-1]
                    ref_schema = ".".join(ref_parts[:-1]) if len(ref_parts) > 1 else None
                    
                    ref_col = ref_match.group(2)
                    
                    foreign_keys.append({
                        'column': col_name,
                        'references_table': ref_table,
                        'references_column': ref_col,
                        'references_schema': ref_schema
                    })
    
    # Return the parsed data in the requested format
    return {
        'table_names': [table_name],
        'column_names': {table_name: columns_names},
        'column_types': {table_name: column_types},
        'primary_keys': {table_name: primary_keys},
        'foreign_keys': {table_name: foreign_keys},
    }

def analyze_schema(schema, parser):    
    """
    Analyze the schema by parsing the DDL statements.

    Args:
        schema (pd.DataFrame): DataFrame containing the schema information.
        parser (function): Function to parse the DDL statements.
    Returns:
        dict: Parsed schema information including table names, column names, types, primary keys, and foreign keys.
    """

    # check if the schema is not empty and has 3 columns namely: table_name, DDL
    if schema.empty or not all(col in schema.columns for col in ['table_name', 'DDL']):
        logging.error("Schema DataFrame is empty or missing required columns.")
        return None;

    ddl_statements = schema['DDL'].tolist()

    # There is an important note here:
    # ! We are ignoring the database which has multiple schemas !!!! 
    # # if there is column name schema_name
    # if 'schema_name' in schema.columns:
    #     # if there is a schema_name column, we need to group by schema_name
    #     ddl_statements = schema.groupby('schema_name')['DDL'].apply(list).tolist()

    # Initialize result structure
    result = {
        'table_names': [],
        'column_names': {},
        'column_types': {},
        'primary_keys': {},
        'foreign_keys': {},
    }

    # Parse each DDL statement
    for ddl in ddl_statements:
        if not isinstance(ddl,str) or ddl == 'missing schema or crupted schema' or ddl == 'nan':
            continue;
        try:
            parsed = parser(ddl)
        except Exception as e:
            logging.error(f"Error parsing DDL statement: {ddl}\nError: {e}")
            raise e

        # Merge the results
        result['table_names'].extend(parsed['table_names'])
        result['column_names'].update(parsed['column_names'])
        result['column_types'].update(parsed['column_types'])
        result['primary_keys'].update(parsed['primary_keys'])
        result['foreign_keys'].update(parsed['foreign_keys'])

    return result

def save_enriched_instance(enriched_instance, output_path, count):
    """
    Save the enriched instance to a JSON file.
    
    Args:
        enriched_instance (dict): The enriched instance to save.
        output_path (str): The path to save the JSON file.
    """
    file_name = ''
    try:
        database_name = enriched_instance['database']['name']
        # if question_id
        question_id = None
        if 'id' in enriched_instance:
            question_id = enriched_instance['id']

        dataset_name = enriched_instance['dataset']

        file_name = f"{dataset_name}_{database_name}_{question_id}_{count}.json" if question_id else f"{dataset_name}_{database_name}_{count}.json"
        output_file_path = os.path.join(output_path, file_name)

        with open(output_file_path, 'w') as f:
            json.dump(enriched_instance, f, indent=4)

        return count + 1
    
    except Exception as e:
        logging.error(f"Error saving enriched instance {file_name}: {e}")
        return count


def load_bird(bird_dataset_path,split='dev',schema_processe_required = True):
    
    instances_path,db_dir,schema_dir = get_bird_dataset_files(bird_dataset_path,split)

    # Process the schemas
    if schema_processe_required and not os.path.exists(os.path.join(bird_dataset_path, f'{split}_schemas')):
        # Process the schemas
        logging.info(f"Processing schemas for {split} split...")
        process_bird_schemas(
            db_dir=os.path.join(bird_dataset_path, f'{split}_databases'),
            output_schema_dir=os.path.join(bird_dataset_path, f'{split}_schemas'),
            with_description=False
        )
    else:
        logging.info(f"Schema processing not required for {split} split or already done.")

    instances = load_bird_instances(instances_path, db_dir, schema_dir)
    print(f"Loaded {len(instances)} instances from {instances_path}")

    return instances, db_dir, schema_dir

def load_spider(spider_dataset_path,split='dev',schema_processe_required = True):
    
    instances_path,db_dir,schema_dir = get_spider_dataset_files(spider_dataset_path,split)

    # Process the schemas
    if schema_processe_required:
        # Process the schemas
        logging.info(f"Processing schemas for {split} split...")
        process_spider_schemas(
            db_dir=db_dir,
            output_schema_dir=os.path.join(spider_dataset_path, f'{split}_schemas'),
            with_description=False
        )
    else:
        logging.info(f"Schema processing not required for {split} split or already done.")

    # Load the instances
    instances = load_spider_instances(
        instances_file=instances_path,
        db_dir=db_dir,
        processed_schema_dir=os.path.join(spider_dataset_path, f'{split}_schemas'),
        limit=None
    )
    logging.info(f"Loaded {len(instances)} instances from {instances_path}.")
    return instances

def load_spider2(SPIDER2_DATASET_PATH,dataset_type):

    data_path = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'spider2-lite.jsonl')
    # Data/Spider2/spider2-lite/evaluation_suite/gold/sql
    queries_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'evaluation_suite', 'gold', 'sql')
    # Data/Spider2/spider2-lite/resource/documents
    external_knowledge_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'resource', 'documents')
    # Data/Spider2/spider2-lite/resource/databases/spider2-localdb
    sqlite_file_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'resource', 'databases', 'spider2-localdb')     

    from spider2_dataloader import load_data, get_spider2_files

    schemas_path_df = get_spider2_files(
        dataset_dir=SPIDER2_DATASET_PATH,
        category=dataset_type,
        available_dbs=['sqlite', 'snowflake'],
        method='csv'
    )

    instances = load_data(data_file_path= data_path,
                            limit=None,
                            queries_dir=queries_dir,
                            external_knowledge_dir=external_knowledge_dir,
                            schemas_path_df=schemas_path_df,
                            available_dbs=['sqlite','snowflake'],
                            sqlites_file_dir=sqlite_file_dir,
                            dataset_type=dataset_type
                          )
    
    logging.info(f"Loaded {len(instances)} instances from {data_path}.")
    return instances

def process(dataset :  str = 'bird', dataset_path : str = '/path/to/bird_dataset', save_enriched : bool = False, **kwargs):
    """
    Process the dataset instances and analyze them.

    Args:
        dataset (str): The name of the dataset to process (e.g., 'bird', 'spider', 'spider2').
        dataset_path (str): The path to the dataset directory.
        save_enriched (bool): Whether to save the enriched instances to a file.
        **kwargs: Additional keyword arguments for specific dataset configurations.

    Returns:
        List[dict]: A list of enriched instances with analysis results.
    """

    # Load the bird dataset
    if dataset == 'bird':
        # getting the split from **kwargs
        split = kwargs.get('split', 'dev')
        instances, db_dir, schema_dir = load_bird(dataset_path, split, schema_processe_required=False)
    elif dataset == 'spider':
        # getting the split from **kwargs
        split = kwargs.get('split', 'dev')
        instances = load_spider(dataset_path, split, schema_processe_required=False)
    elif dataset == 'spider2':
        instances = load_spider2(dataset_path, dataset_type=kwargs.get('dataset_type', 'dev'))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    if save_enriched:
        # Process the schemas
        output_dir = os.path.join(dataset + '_enriched_instances')
        # create the output directory by removing the existing directory even if it is not empty
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        # create the output directory
        os.makedirs(output_dir, exist_ok=True)
        count = 0

    enriched_instances = []
    for instance in tqdm(instances, desc=f"Processing {dataset} instances"):    
        # load the schema         
        schema = load_schemas_list(instance['schemas'])
        # Analyze the schema by parsing the DDL 
        database_type = instance['database']['type']
        if database_type == 'sqlite':
            parsed_schema = analyze_schema(schema, parse_sqlite_ddl_statement)
        elif database_type == 'snowflake':
            parsed_schema = analyze_schema(schema, parse_snowflake_ddl)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
        
        # check if the parsed schema is None
        if parsed_schema is None:
            logging.error(f"Parsed schema is None for instance: {instance['id'] if 'id' in instance else 'unknown'} with database type {database_type}. Skipping instance.")
            continue;
        
        # replacing the instance['schemas'] with the schema dataframe by converting it to a dictionary
        instance['schemas'] = schema.to_dict(orient='records')
        # Analyze the question
        question = instance['question']
        question_analysis = analyze_question(question, parsed_schema)

        # Analyze the SQL query
        sql_query = instance['sql']
        sql_analysis = analyze_sql(sql_query)

        # enrich the instance by adding the question analysis to the current instance
        instance['question_analysis'] = question_analysis
        # enrich the instance by adding the sql analysis to the current instance
        instance['sql_analysis'] = sql_analysis
        # Add the enriched instance to the list
        enriched_instances.append(instance)
        if save_enriched:
            # Save the enriched instance
            count = save_enriched_instance(instance, output_dir, count)
    
    return enriched_instances
            
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and analyze dataset instances.")
    parser.add_argument('--dataset', type=str, default='bird', help='Dataset name (default: bird)')
    parser.add_argument('--dataset_path', type=str, default='/path/to/bird_dataset', help='Path to the dataset directory')

    parser.add_argument('--save_enriched', action='store_true', help='Whether to save the enriched instances to a file')
    parser.add_argument('--split', type=str, default='dev', help='Dataset split (default: dev)')
    parser.add_argument('--dataset_type', type=str, default='dev', help='Dataset type for spider2 (default: dev)')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Processing dataset: {args.dataset} from path: {args.dataset_path}")
    enriched_instances = process(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        save_enriched=args.save_enriched,
        split=args.split,
        dataset_type=args.dataset_type
    )
    logging.info(f"Processed {len(enriched_instances)} instances.")



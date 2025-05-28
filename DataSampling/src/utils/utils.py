import os
import pandas as pd
from typing import Dict
import logging
import json
import glob
from typing import List
import re


def _get_instance_model_name(instance: Dict) -> str:
    try:
        model_name = instance['inference_results']['model']['model_name']

        if model_name.find('/') != -1:
            model_name = model_name.split('/')[1]
        
        return model_name

    except KeyError:
        logging.error(f"Key 'model_name' not found in instance: {instance}")
        return None

def load_data(data_dirs: List[str]) -> pd.DataFrame:

    data = {}
    for data_dir in data_dirs:
        data_list = glob.glob(data_dir + '**/instance_*.json', recursive=True)

        for data_file in data_list:
            
            with open(data_file, 'r') as f:
                data_json = json.load(f)
            
                model_name = _get_instance_model_name(data_json)
                if not model_name:
                    logging.error(f"Model name not found for file: {data_file}")
                    continue

                if data_json['id'] not in data:
                    data[data_json['id']] = []
                data[data_json['id']].append((model_name,data_json))
            
    # Convert the dictionary to a DataFrame where the columns are id, all models names, and the rows are the instances
    data_list = []
    for instance_id, instances in data.items():
        instance_dict = {'id': instance_id}
        for model_name, instance in instances:
            instance_dict[model_name] = instance
        data_list.append(instance_dict)
    
    df = pd.DataFrame(data_list)
    # df = df.set_index('id')
    return df

def _match_data_path_dataset(root_data_dir,dataset_name: str, requested_path) -> str:
    """
    Match the requested path to the dataset type directory.
    Currently supports 'bird' and 'spider' datasets.
    This function is mainly use for loading schema information and database files.
    Args:
        root_data_dir: Root directory for the dataset
        dataset_name: Name of the dataset (e.g., 'bird', 'spider')
        requested_path: Path to the data file
    Returns:
        Full path to the requested data file
    """
    
    if dataset_name == 'bird':
        return os.path.join(root_data_dir, 'bird_subset', requested_path)
    elif dataset_name == 'spider':
        return os.path.join(root_data_dir, 'spider_subset', requested_path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets are 'bird' and 'spider'.")    

def process_schema(root_data_dir : str,schema_info: Dict, dataset_name : str) -> str:
        """
        Process schema information into a text representation for embedding.
        
        Args:
            root_data_dir: Root directory for the dataset
            schema_info: Schema information from the database
            dataset_name: Name of the dataset (e.g., 'bird', 'spider')
            
        Returns:
            String representation of the schema
        """
        if not schema_info:
            return ""
            
        schema_text = ""
        
        # Process schema information if it's available as a path
        schemas_path = schema_info.get('path', [])
        for schema_path in schemas_path:
            # Try to load schema from the path
            schema_path = _match_data_path_dataset(root_data_dir,dataset_name, schema_path)

            try:
                if os.path.exists(schema_path): 
                    schema_df = pd.read_csv(schema_path)
                    
                    # Extract relevant columns: table_name, description, DDL
                    for _, row in schema_df.iterrows():
                        table_name = row.get('table_name', '')
                        description = row.get('description', '')
                        ddl = row.get('DDL', '')
                        
                        if table_name:
                            schema_text += f"Table: {table_name} "
                        if description:
                            schema_text += f"Description: {description} "
                        if ddl:
                            schema_text += f"DDL: {ddl} "
            except Exception as e:
                print(f"Error loading schema from {schema_path}: {e}")
                continue
            
        return schema_text

def extract_sql_query_from_text( text: str) -> str:
    """
    Extract SQL query from text, handling various formats (JSON, code blocks, etc.)
    
    Args:
        text: Raw text output that may contain SQL query
        
    Returns:
        Extracted SQL query as a string, or empty string if extraction fails
    """
    # Removing the <think> and </think> tags if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>', '', text)
    # Try to find SQL in JSON format first
    json_match = re.search(r'(\{.*"sql".*\})', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            # Try to parse the matched string as JSON
            json_obj = json.loads(json_str)
            if "sql" in json_obj:
                return json_obj["sql"]
        except json.JSONDecodeError:
            pass
    
    # Try to find SQL in code blocks with ```sql or ```SQL format
    sql_code_block = re.search(r'```(?:sql|SQL)\s*([\s\S]*?)```', text, re.DOTALL)
    if sql_code_block:
        return sql_code_block.group(1).strip()
    
    # Try to find SQL in any code blocks
    any_code_block = re.search(r'```\s*([\s\S]*?)```', text, re.DOTALL)
    if any_code_block:
        code_content = any_code_block.group(1).strip()
        # Check if it looks like SQL (contains SELECT, FROM, etc.)
        if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', code_content, re.IGNORECASE):
            return code_content
    
    # Try to find patterns that look like SQL queries directly in the text
    sql_patterns = [
        # Look for SELECT statement
        r'(?:query:?\s*)?(SELECT\s+[\s\S]*?(?:FROM\s+[\s\S]*?)(?:;|$))',
        # Look for other common SQL statements
        r'(?:query:?\s*)?(INSERT\s+INTO\s+[\s\S]*?(?:;|$))',
        r'(?:query:?\s*)?(UPDATE\s+[\s\S]*?(?:;|$))',
        r'(?:query:?\s*)?(DELETE\s+FROM\s+[\s\S]*?(?:;|$))',
        r'(?:query:?\s*)?(CREATE\s+TABLE\s+[\s\S]*?(?:;|$))'
    ]
    
    for pattern in sql_patterns:
        sql_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
    
    # If we still haven't found a SQL query, look for "The SQL query is:" patterns
    sql_intro_match = re.search(r'(?:The SQL query is:?|Here\'s the SQL:?|Generated SQL:?)\s*([\s\S]*?)(?:\n\n|$)', text, re.DOTALL)
    if sql_intro_match:
        # Get the content after the introduction
        potential_sql = sql_intro_match.group(1).strip()
        # Check if it looks like SQL
        if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', potential_sql, re.IGNORECASE):
            return potential_sql
    
    # No SQL query found
    return ""
import os
import pandas as pd
from typing import Dict
import logging
import json
import glob
from typing import List, Tuple, Union, Optional
import certifi
import re
import sqlparse
import io
import sqlite3
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Ensure the logging captures all messages, including debug
logging.getLogger().setLevel(logging.DEBUG)

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

def load_password_and_api_key(key_file_path):
    """
    """
    with open(key_file_path, 'r') as f:
        api_key = f.read().strip()
    return api_key


def connect_to_mongodb_atlas_1(username: str, password: str):
    """
    Connect to MongoDB Atlas using the provided username and password.
    Args:
        username: MongoDB Atlas username
        password: MongoDB Atlas password
    """
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi

    uri = f"mongodb+srv://{username}:{password}@cluster0.ixr2wwl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

def create_sql_prompt(question: str, schema_text: str, evidence: str = None):
            
    # Format system message
    system_message = (
        "You are a database expert. "
        "You are supposed to provide a SQL query based on the user's question and the provided database schema. "
        "Your response must be in JSON format with a field named 'sql' containing the generated SQL query. "
        "Example response format: {\"sql\": \"SELECT * FROM table WHERE condition\"}"
    )
    
    # Format user message
    user_message = f"{question}\n\n"
    
    # Add evidence if available
    if evidence:
        user_message += f"Evidence: {evidence}\n\n"
    
    user_message += f"Schema: {schema_text}\n\n"
    
    user_message += "Please provide the SQL query in the specified JSON format."
    
    return system_message, user_message

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

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query by removing extra spaces and formatting.
    
    Args:
        sql: SQL query string
        
    Returns:
        Normalized SQL query string
    """
    # Use sqlparse to format the SQL query
    parsed = sqlparse.parse(sql)
    
    # Convert parsed SQL back to string
    normalized_sql = sqlparse.format(str(parsed[0]), reindent=True, keyword_case='upper')
    
    # Remove extra spaces
    normalized_sql = re.sub(r'\s+', ' ', normalized_sql).strip()
    
    return normalized_sql

def check_exact_match(predicted_sql: str, ground_truth_sql: str) -> bool:
    """
    Check if predicted SQL exactly matches ground truth after normalization.
    
    Args:
        predicted_sql: Predicted SQL query
        ground_truth_sql: Ground truth SQL query
        
    Returns:
        True if exact match, False otherwise
    """
    # Normalize both queries
    normalized_pred = normalize_sql(predicted_sql)
    normalized_gt = normalize_sql(ground_truth_sql)
    
    # Compare normalized queries
    return normalized_pred == normalized_gt

def check_execution_accuracy(predicted_sql: str, ground_truth_sql: str, 
                             db_connection: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Check if predicted SQL executes correctly and produces the same output as ground truth.
    
    Args:
        predicted_sql: Predicted SQL query
        ground_truth_sql: Ground truth SQL query
        db_connection: SQLite database connection
        
    Returns:
        Tuple of (is_correct, error_message)
    """
    try:
        # Execute ground truth SQL
        cursor = db_connection.cursor()
        cursor.execute(ground_truth_sql)
        ground_truth_result = cursor.fetchall()
        
        # Convert to pandas DataFrame for easier comparison
        ground_truth_df = pd.DataFrame(ground_truth_result)
        
        try:
            # Execute predicted SQL
            cursor.execute(predicted_sql)
            predicted_result = cursor.fetchall()
            # Simple check 
            if set(predicted_result) == set(ground_truth_result):
                return True, ""
            
            # Convert to pandas DataFrame
            predicted_df = pd.DataFrame(predicted_result)
            
            # Check if the results match
            if ground_truth_df.shape == predicted_df.shape:
                # Sort both dataframes if they have values (not empty)
                if not ground_truth_df.empty and not predicted_df.empty:
                    # First handle column ordering - reindex both DataFrames with sorted column names
                    # This ensures column order doesn't affect comparison
                    if len(ground_truth_df.columns) > 0:
                        ground_truth_columns = sorted(ground_truth_df.columns)
                        predicted_columns = sorted(predicted_df.columns)
                        
                        # If column sets are different, DataFrames are not equal
                        if set(ground_truth_columns) != set(predicted_columns):
                            return False, "Results have different column sets"
                        
                        # Reindex with sorted columns
                        ground_truth_df = ground_truth_df[ground_truth_columns]
                        predicted_df = predicted_df[predicted_columns]
                    
                    # Now sort by values in each row
                    ground_truth_sorted = ground_truth_df.sort_values(by=list(ground_truth_df.columns)).reset_index(drop=True)
                    predicted_sorted = predicted_df.sort_values(by=list(predicted_df.columns)).reset_index(drop=True)
                    
                    # Check equality
                    return ground_truth_sorted.equals(predicted_sorted), ""
                else:
                    # If both empty, that's a match
                    return ground_truth_df.empty == predicted_df.empty, ""
            else:
                return False, f"Results have different shapes: ground truth {ground_truth_df.shape} vs predicted {predicted_df.shape}"
            
        except Exception as e:
            return False, f"Execution error: {str(e)}"
            
    except Exception as e:
        return False, f"Ground truth execution error: {str(e)}"
    
def get_sqlite_db_connection(database_path: str) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.
    
    Args:
        database_path: Path to the SQLite database file
        
    Returns:
        SQLite connection object
    """
    database_path = database_path.split(database_path.split('/')[0])[1]
    
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    
    return conn

def read_ddl_schema_csv(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Robust function to read CSV files containing DDL statements with complex formatting.
    
    This function handles:
    - Embedded quotes within DDL statements
    - Multi-line DDL content
    - Commas within quoted DDL fields
    - Empty description fields
    
    Args:
        file_path (str): Path to the CSV file
        encoding (str): File encoding, default is 'utf-8'
    
    Returns:
        pd.DataFrame: Parsed DataFrame with columns: table_name, description, DDL
    """
    
    try:
        # Method 1: Try with standard pandas parameters for complex CSV
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            quotechar='"',           # Standard quote character
            quoting=1,               # QUOTE_ALL - treat all fields as potentially quoted
            skipinitialspace=True,   # Skip whitespace after delimiter
            escapechar='\\',         # Handle escaped characters
            doublequote=True,        # Handle double quotes as escapes
            engine='python'          # Use Python engine for better handling of complex cases
        )
        return df
        
    except Exception as e:
        logging.warning(f"Method 1 failed with error: {e}")
        
        try:
            # Method 2: Manual parsing approach for highly problematic files
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # Split into lines and process manually
            lines = content.strip().split('\n')
            headers = [col.strip() for col in lines[0].split(',')]
            
            data_rows = []
            current_row = []
            in_quotes = False
            current_field = ""
            
            for line_num, line in enumerate(lines[1:], 2):
                i = 0
                while i < len(line):
                    char = line[i]
                    
                    if char == '"' and not in_quotes:
                        in_quotes = True
                    elif char == '"' and in_quotes:
                        # Check for double quote (escaped quote)
                        if i + 1 < len(line) and line[i + 1] == '"':
                            current_field += '"'
                            i += 1  # Skip the next quote
                        else:
                            in_quotes = False
                    elif char == ',' and not in_quotes:
                        current_row.append(current_field.strip())
                        current_field = ""
                    else:
                        current_field += char
                    
                    i += 1
                
                # If we're still in quotes, this is a multi-line field
                if in_quotes:
                    current_field += "\\n"  # Add line break representation
                else:
                    # End of row
                    current_row.append(current_field.strip())
                    if len(current_row) == len(headers):
                        data_rows.append(current_row)
                        current_row = []
                        current_field = ""
                    else:
                        # Incomplete row, continue to next line
                        current_field += " "
            
            # Handle any remaining incomplete row
            if current_row or current_field:
                current_row.append(current_field.strip())
                if len(current_row) == len(headers):
                    data_rows.append(current_row)
            
            df = pd.DataFrame(data_rows, columns=headers)
            return df
            
        except Exception as e2:
            logging.warning(f"Method 2 failed with error: {e2}")
            
            # Method 3: Fallback with very permissive settings
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=',',
                    quotechar='"',
                    quoting=3,  # QUOTE_NONE - don't use quoting
                    engine='python',
                    error_bad_lines=False,
                    warn_bad_lines=True,
                    on_bad_lines='skip'
                )
                return df
                
            except Exception as e3:
                logging.warning(f"All methods failed. Last error: {e3}")
                raise Exception("Unable to parse the CSV file with any method")

def clean_ddl_column(df: pd.DataFrame, ddl_column: str = 'DDL') -> pd.DataFrame:
    """
    Clean and format the DDL column for better readability and processing.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        ddl_column (str): Name of the DDL column to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned DDL column
    """
    df_cleaned = df.copy()
    
    if ddl_column in df_cleaned.columns:
        # Remove extra quotes and clean up formatting
        df_cleaned[ddl_column] = df_cleaned[ddl_column].str.replace('""', '"')
        df_cleaned[ddl_column] = df_cleaned[ddl_column].str.strip('"')
        df_cleaned[ddl_column] = df_cleaned[ddl_column].str.replace('\\n', '\n')
        
        # Handle empty descriptions
        if 'description' in df_cleaned.columns:
            df_cleaned['description'] = df_cleaned['description'].fillna('')
    
    return df_cleaned
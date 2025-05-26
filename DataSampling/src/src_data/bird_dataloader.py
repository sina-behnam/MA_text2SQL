import os
import glob
import json
import sqlite3
import os
import pandas as pd
import logging
import csv
import sqlite3
from typing import Dict, List
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------ phase 1 ------------------------------------------------------

def get_bird_dataset_files(bird_dataset_path,split):
    """
    Get the paths to the bird dataset files.
    Args:
        bird_dataset_path (str): Path to the bird dataset directory.
        split (str): The split of the dataset ('train', 'val', 'test').
    Returns:
        tuple: Paths to the bird dataset files.
    """
    
    instances_file = os.path.join(bird_dataset_path, f'{split}.json')
    db_dir = os.path.join(bird_dataset_path, f'{split}_databases')
    schema_dir = os.path.join(bird_dataset_path, f'{split}_schemas')

    return instances_file, db_dir, schema_dir

def generate_ddl_from_sqlite_2(sqlite_file: str) -> Dict[str, str]:
    """
    Extract CREATE TABLE DDL statements from a SQLite file.
    
    Args:
        sqlite_file (str): Path to the SQLite database file
        
    Returns:
        Dict[str, str]: A dictionary mapping table names to their DDL statements
    """
    if not os.path.exists(sqlite_file):
        raise FileNotFoundError(f"SQLite file not found: {sqlite_file}")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    # Query sqlite_master table to get all table definitions
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = cursor.fetchall()
    
    # Create a dictionary to hold table_name -> DDL mapping
    ddl_map = {}
    
    for table_name, ddl in tables:
        if ddl:  # Ensure DDL is not None
            ddl_map[table_name] = ddl
    
    # Close the connection
    cursor.close()
    conn.close()
    
    return ddl_map

def generate_ddl_from_sqlite(db_path : str) -> dict:
    """Generate DDL statements from an existing SQLite database file."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get a list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    
    # A dict of table_name: DDL statement
    ddl_statements = {}
    
    for table in tables:
        table_name = table[0]

        ddl_statements[str(table_name)] = []
        
        # Get CREATE statement for each table
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_statement = cursor.fetchone()[0]
        ddl_statements[table_name] = create_statement + ";"
        
        # Get CREATE statements for indexes on this table
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}' AND sql IS NOT NULL;")
        indexes = cursor.fetchall()
        
        for idx in indexes:
            ddl_statements[table_name].append(idx[0] + ";")
    
    conn.close()
    return ddl_statements

def get_generated_description(table_name):
    raise NotImplementedError("This function is not implemented yet. Please implement the function to get the generated description for the table.")

def save_ddl_to_file(ddl_statements : dict, output_path : str, with_description : bool = False) -> pd.DataFrame:
    """
    Save DDL statements to a csv file. including the columns table_name,description, DDL.

    Args:
        ddl_statements (dict): Dictionary of DDL statements.
        output_path (str): Path to save the CSV file.
        with_description (bool): Whether to include generated descriptions in the CSV.

    Returns:
        pd.DataFrame: DataFrame containing the DDL statements and descriptions.
    """

    data = {
        'table_name': [],
        'description': [],
        'DDL': []
    }

    for statement in ddl_statements.items():
        table_name = statement[0]
        ddl = statement[1]
        
        description = None
        if with_description:
            description = get_generated_description(table_name)

        data['table_name'].append(table_name)
        data['description'].append(description)
        data['DDL'].append(ddl)

    df = pd.DataFrame(data)

    df.to_csv(output_path, index=False,quoting=csv.QUOTE_NONNUMERIC)
    return df

def process_bird_schemas(db_dir, output_schema_dir, with_description = False):
    """
    Process the bird schemas and save them to CSV files.

    Args:
        db_dir (str): Directory containing the SQLite database files.
        output_schema_dir (str): Directory to save the output CSV files.
        with_description (bool): Whether to include generated descriptions in the CSV.
    """
    
    if not os.path.exists(output_schema_dir):
        os.makedirs(output_schema_dir)

    db_files = glob.glob(os.path.join(db_dir, '**/*.sqlite'), recursive=True)
    if len(db_files) == 0:
        logger.warning(f"No SQLite files found in {db_dir}.")
        return

    for db_file in db_files:
        db_name = os.path.basename(db_file).split('.')[0]
        os.makedirs(os.path.join(output_schema_dir, db_name), exist_ok=True)
        output_path = os.path.join(output_schema_dir, db_name, f"{db_name}.csv")
        
        ddl_statements = generate_ddl_from_sqlite(db_file)
        df = save_ddl_to_file(ddl_statements, output_path, with_description)
        logger.info(f"Processed {db_name} and saved to {output_path}")

# ------------------------------------------------------ phase 2 ------------------------------------------------------


def get_bird_database_files_paths(db_dir, database_name):
    """
    Get the paths to the bird database files.
    Args:
        db_dir (str): Path to the bird database directory.
        database_name (str): Name of the database.
    Returns:
        tuple: Paths to the SQLite and CSV files in the database.
    """
    
    sqlite_files = glob.glob(os.path.join(db_dir, database_name, '*.sqlite'))
    if len(sqlite_files) == 0:
        raise FileNotFoundError(f"No SQLite files found in {os.path.join(db_dir, database_name)}")
    
    csv_files = glob.glob(os.path.join(db_dir, database_name,'database_description' , '*.csv'))

    if len(sqlite_files) == 0:
        logger.error(f"No SQLite files found in {os.path.join(db_dir, database_name)}")
    
    return sqlite_files, csv_files

def get_bird_processed_schemas(schemas_dir, database_name):
    """
    Get the paths to the processed bird schemas.
    Args:
        schemas_dir (str): Path to the bird schemas directory.
        database_name (str): Name of the database.
    Returns:
        tuple: Paths to the SQLite and CSV files in the schemas directory.
    """
    
    csv_files = glob.glob(os.path.join(schemas_dir, database_name, '*.csv'))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(schemas_dir, database_name)}")
    
    return csv_files

def make_bird_database_dict(instance, db_dir):
    """
    Create a dictionary for the bird database instance.
    Args:
        instance (dict): Bird instance dictionary.
        db_dir (str): Path to the bird database directory.
    Returns:
        dict: Dictionary containing the database information.
    """
    
    db_name = instance['db_id']
    db_path, csv_files = get_bird_database_files_paths(db_dir, db_name)
    
    return {
        'name': db_name,
        'path': db_path,
        'csv_files': csv_files,
        'type': 'sqlite',
    }

def make_bird_schemas(instance, schemas_dir):
    """
    Create a dictionary for the bird schema instance.
    Args:
        instance (dict): Bird instance dictionary.
        schemas_dir (str): Path to the bird schemas directory.
    Returns:
        list: List of dictionaries containing the schema information.
    """
    
    db_name = instance['db_id']
    schema_path = get_bird_processed_schemas(schemas_dir, db_name)
    
    return [{
        'name': db_name,
        'path': schema_path,
        'type': 'csv',
    }]

def standardize_bird_instance(instance):

    """
    Standardize the bird instance dictionary.
    Args:
        instance (dict): Bird instance dictionary.
    Returns:
        dict: Standardized bird instance dictionary.
    """
   
    standardized_instance = {
        'id': instance['question_id'],
        'dataset': 'bird',
        'database': instance['database'],
        'schemas': instance['schemas'],
        'question': instance['question'],
        'sql': instance['SQL'],
        'evidence': instance['evidence'],
        'difficulty': instance['difficulty'],
    }
    
    return standardized_instance
    
def load_bird_instances(instances_file, db_dir, processed_schema_dir, limit=None):
    """
    Load bird instances from a JSON file.
    Args:
        instances_file (str): Path to the JSON file containing bird instances.
        db_dir (str): Path to the bird database directory.
        processed_schema_dir (str): Path to the processed schemas directory.
        limit (int, optional): Limit the number of instances to load. If None, load all instances.
    Returns:
        list: List of bird instances.
    """

    if not os.path.exists(instances_file):
        raise FileNotFoundError(f"File {instances_file} does not exist.")

    with open(instances_file, 'r') as f:
        instances = json.load(f)

    if limit is not None:
        instances = instances[:limit]

    standard_instances = []
    for instance in instances:
        # ! Attention in the original instance,
        # ! there are some keys that we are replacing them but it required to process the two below function and we replace them later in standardize_bird_instance
        # Process the original instance to make the database section
        instance['database'] = make_bird_database_dict(instance, db_dir)
        # Process the original instance to make the schema section
        instance['schemas'] = make_bird_schemas(instance, processed_schema_dir) 
        # * Standardize the instance
        standard_instances.append(standardize_bird_instance(instance))

    return standard_instances



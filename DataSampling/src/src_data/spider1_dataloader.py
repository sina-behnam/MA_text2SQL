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
import sqlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------ phase 1 ------------------------------------------------------

def get_spider_dataset_files(spider_dataset_path,split):
    """
    Get the paths to the spider dataset files.
    Args:
        spider_dataset_path (str): Path to the spider dataset directory.
        split (str): The split of the dataset ('train', 'val', 'test').
    Returns:
        tuple: Paths to the spider dataset files.
    """
    
    instances_file = os.path.join(spider_dataset_path, f'{split}.json')
    db_dir = os.path.join(spider_dataset_path, f'database')
    schema_dir = db_dir

    return instances_file, db_dir, schema_dir

def generate_ddl_from_sqlite(sqlite_file: str) -> Dict[str, str]:
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

def process_spider_schemas(db_dir, output_schema_dir, with_description = False):
    """
    Process the spider2 schemas and save them to CSV files.

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

def get_spider_database_files_paths(db_dir, database_name):
    """
    Get the paths to the spider database files.
    Args:
        db_dir (str): Path to the spider database directory.
        database_name (str): Name of the database.
    Returns:
        list: List of paths to the SQLite files in the database directory.
    """
    
    sqlite_files = glob.glob(os.path.join(db_dir, database_name, '*.sqlite'))
    if len(sqlite_files) == 0:
        raise FileNotFoundError(f"No SQLite files found in {os.path.join(db_dir, database_name)}")
    
    
    if len(sqlite_files) == 0:
        logger.error(f"No SQLite files found in {os.path.join(db_dir, database_name)}")
    
    return sqlite_files

def get_spider_processed_schemas(schemas_dir, database_name):
    """
    Get the paths to the processed spider schemas.
    Args:
        schemas_dir (str): Path to the spider schemas directory.
        database_name (str): Name of the database.
    Returns:
        tuple: Paths to the SQLite and CSV files in the schemas directory.
    """
    
    csv_files = glob.glob(os.path.join(schemas_dir, database_name, '*.csv'))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(schemas_dir, database_name)}")
    
    return csv_files

def make_spider_database_dict(instance, db_dir):
    """
    Create a dictionary for the spider database instance.
    Args:
        instance (dict): spider instance dictionary.
        db_dir (str): Path to the spider database directory.
    Returns:
        dict: Dictionary containing the database information.
    """
    
    db_name = instance['db_id']
    db_path = get_spider_database_files_paths(db_dir, db_name)
    
    return {
        'name': db_name,
        'path': db_path,
        'type': 'sqlite',
    }

def make_spider_schema_dict(instance, schemas_dir):
    """
    Create a dictionary for the spider schema instance.
    Args:
        instance (dict): spider instance dictionary.
        schemas_dir (str): Path to the spider schemas directory.
    Returns:
        dict: Dictionary containing the schema information.
    """
    
    db_name = instance['db_id']
    schema_path = get_spider_processed_schemas(schemas_dir, db_name)
    
    return {
        'name': db_name,
        'path': schema_path,
        'type': 'csv',
    }

def generate_difficulty(sql_query):
    # Parse the SQL and get all non-whitespace tokens
    sql_tokens = []
    for statement in sqlparse.parse(sql_query):
        sql_tokens.extend([token for token in statement.flatten() if not token.is_whitespace])
    if len(sql_tokens) > 160:
        return 'challenging'
    elif len(sql_tokens) > 80:
        return 'moderate'
    else:
        return 'simple'

def standardize_spider_instance(instance,count):

    """
    Standardize the spider instance dictionary.
    Args:
        instance (dict): spider instance dictionary.
    Returns:
        dict: Standardized spider instance dictionary.
    """
   
    standardized_instance = {
        'id': count,
        'dataset': 'spider',
        'database': instance['database'],
        'schemas': instance['schemas'],
        'question': instance['question'],
        'sql': instance['query'],
        'evidence': '',
        'difficulty': generate_difficulty(instance['query']),
    }
    
    return standardized_instance
    
def load_spider_instances(instances_file, db_dir, processed_schema_dir, limit=None):
    """
    Load spider instances from a JSON file.
    Args:
        instances_file (str): Path to the JSON file containing spider instances.
        db_dir (str): Path to the spider database directory.
        processed_schema_dir (str): Path to the processed schemas directory.
        limit (int, optional): Limit the number of instances to load. If None, load all instances.
    Returns:
        list: List of spider instances.
    """

    if not os.path.exists(instances_file):
        raise FileNotFoundError(f"File {instances_file} does not exist.")

    with open(instances_file, 'r') as f:
        instances = json.load(f)

    if limit is not None:
        instances = instances[:limit]

    standard_instances = []
    count = 0
    for instance in instances:
        # ! Attention in the original instance,
        # ! there are some keys that we are replacing them but it required to process the two below function and we replace them later in standardize_spider_instance
        # Process the original instance to make the database section
        instance['database'] = make_spider_database_dict(instance, db_dir)
        # Process the original instance to make the schema section
        instance['schemas'] = make_spider_schema_dict(instance, processed_schema_dir) 
        # * Standardize the instance
        standard_instances.append(standardize_spider_instance(instance,count))

        count += 1

    return standard_instances

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
    

if __name__ == "__main__":

    load_spider(
        spider_dataset_path='/home/mahmoud/Downloads/spider',
        split='train',
        schema_processe_required=True
    )
    



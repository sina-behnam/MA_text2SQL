import glob
import os
import pandas as pd
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,)
logger = logging.getLogger(__name__)
# disable the logger
# logger.setLevel(logging.CRITICAL)
# enable the logger
logger.setLevel(logging.INFO)

SPIDER2_DATASET_PATH = 'Your/Path/To/Data/Spider2'  # Change this to your Spider2 dataset path

def get_schemas_path(dataset_dir, method='json'):
    """
    Get the path to the database schemas in the Spider2 dataset.

    Args:
        dataset_dir (str): The path to the Spider2 dataset directory.
    Returns:
        dict: A dictionary where the keys are database names and the values are lists of paths to the database schema files.
    """
    if method == 'json':
        database_files_path = glob.glob(dataset_dir + os.sep + '**' + os.sep + '*.json', recursive=True)
    elif method == 'csv':
        database_files_path = glob.glob(dataset_dir + os.sep + '**' + os.sep + '*.csv', recursive=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    db_schemas_path = {}

    for file in database_files_path:
        databases_path = file.split(dataset_dir)[-1]
        # Getting the database name from the path to which database this files is belonging to
        database_name = databases_path.split(os.sep)[1]
        # create a list based on the database name in db_schemas_path
        if database_name not in db_schemas_path:
            db_schemas_path[database_name] = []
        db_schemas_path[database_name].append(file)

    return db_schemas_path

def prepare_spider2_lite_files(dataset_dir, available_dbs=None,method=None):
    """
    Prepare the Spider2 Lite dataset.
    """
    db_dir = os.path.join(dataset_dir, 'spider2-lite','resource','databases')

    db_paths = {}

    for available_db in available_dbs:
        
        if available_db == 'snowflake':
            snowflake_dir = os.path.join(db_dir, 'snowflake')

            db_paths['snowflake'] = get_schemas_path(snowflake_dir,method=method)

        elif available_db == 'sqlite':
            sqlite_dir = os.path.join(db_dir, 'sqlite')

            db_paths['sqlite'] = get_schemas_path(sqlite_dir,method=method)

        elif available_db == 'bigquery':
            bigquery_dir = os.path.join(db_dir, 'bigquery')

            db_paths['bigquery'] = get_schemas_path(bigquery_dir,method=method)
    
        else:
            raise ValueError(f"Unknown database type: {available_db}")
        
    return pd.DataFrame.from_dict(db_paths, orient='index').T


def get_spider2_files(dataset_dir,category,available_dbs=None,method=None) -> pd.DataFrame:
    """
    Get the path to the Spider2 dataset directory.
    """
    if category == 'snow':
        raise NotImplementedError("Snowflake database preparation is not implemented yet.")
    elif category == 'lite':
        return prepare_spider2_lite_files(dataset_dir, available_dbs,method)
    else:
        raise ValueError(f"Unknown category: {category}")
    
def read_json_file(file_path):
    """
    Read a JSON file and return its content.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))  
    return data

def sync_question_key_name(instance):
        """
        Ensure the question key name is consistent across instances.
        This method checks if the instance has a 'question' key and renames it if necessary.
        If the instance has a different key name for the question (e.g., 'instruction' or 'query'), it renames it to 'question'.
        If no valid key is found, it raises a KeyError.
        Args:
            instance: Dictionary containing the original instance data
        Returns:
            Updated instance dictionary with the correct question key name
        """
        if 'question' in instance:
            return instance
        elif 'instruction' in instance:
            instance['question'] = instance.pop('instruction')
        elif 'query' in instance:
            instance['question'] = instance.pop('query')
        else:
            # If no known key exists, raise an error
            raise KeyError("No valid question key found in the instance.")
        return instance   

def get_sql_query_per_instance(instance_id,quires_dir):
        """
        Get the SQL query for a specific instance ID.

        Args:
            instance_id: ID of the instance
        Returns:
            SQL query string or None if not found  
        """
        if quires_dir is not None:
            # Get the path to the SQL file
            sql_file_path = os.path.join(quires_dir, f"{instance_id}.sql")
            if not os.path.exists(sql_file_path):
                logger.debug(f"SQL file not found for instance {instance_id}: {sql_file_path}")
                return None

            # Read the SQL query from the file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_query = f.read()
            
            return sql_query.strip()
        else:
            logger.error(f"No queries directory specified for instance {instance_id}")
            raise ValueError(f"No queries directory specified for instance {instance_id}")

def get_external_knowledge_instance(external_knowledge_file: str,external_know_dir: str):
        """
        Get external knowledge for a specific instance.

        Args:
            external_knowledge_file: Name of the external knowledge file.
            external_know_dir: Directory containing the external knowledge files.
        
        Returns:
            External knowledge string or None if not found
        """

        if external_knowledge_file is not None and (external_knowledge_file != '' or external_knowledge_file != []):
            # Load the external knowledge from the directory
            external_knowledge_path = os.path.join(external_know_dir, external_knowledge_file)
            if not os.path.exists(external_knowledge_path):
                raise FileNotFoundError(f"External knowledge file not found: {external_knowledge_path}")
            # IT is a .md file 
            with open(external_knowledge_path, 'r', encoding='utf-8') as f:
                external_knowledge_data = f.read()
            
            return external_knowledge_data.strip()
        else:
            return None
        
def get_database_schema(database_name : str, schema_paths_df : pd.DataFrame, available_dbs: list = None):
    """
    Check if the database schema exists for a given database name.
    This function checks if the database name exists in the provided database schemas path dictionary.

    Args:
        database_name: Name of the database
        schema_paths_df: Dataframe containing the database schemas paths
        available_dbs: List of available databases to check against
    Returns:
        tuple: (database_category, schema_path) if the database schema exists,
                None if the database schema does not exist.
    """
    if available_dbs is None:
        available_dbs = schema_paths_df.columns.tolist()

    if database_name not in schema_paths_df.index:
        logger.warning(f"Database name {database_name} not found in the schema paths dataframe.")
        return None
    
    schema_paths = schema_paths_df.loc[database_name, available_dbs].dropna().to_dict()

    # ! Here the database name may not belong to any of the available databases, so the schema_paths will be empty
    if not schema_paths:
        return None
    
    # ! Getting ONLY the first key of the schema_paths dictionary, meaning that we are assuming that the a database name can only belong to one category of 
    # ! database, e.g. sqlite, bigquery, snowflake
    if len(schema_paths) > 1:
        logger.warning(f"Multiple database categories found for database name: {database_name}. Using the first one found.")

    database_cat = list(schema_paths.keys())[0] 

    return database_cat, schema_paths[database_cat]
    

def get_sqlite_db_file_path(sqlite_file_dir, db_name):
    """
    Check if the SQLite database file by the name of the database name exists in the directory.
    Then it returns the path to the SQLite database file.

    Args:
        sqlite_file_dir: Directory containing the SQLite database files
        db_name: Name of the database 
    
    Returns:
        str: Path to the SQLite database file
    """
    # Check if the SQLite database file exists in the directory
    sqlite_db_file_path = os.path.join(sqlite_file_dir, db_name + '.sqlite')
    if os.path.exists(sqlite_db_file_path):
        return sqlite_db_file_path
    return None

def standarize_spider2_instance(instance, count):

    new_instance = {
        'id': count,
        'original_instance_id': instance.get('instance_id', None),
        'dataset' : instance.get('dataset', None),
        'question': instance.get('question', None),
        'sql': instance.get('sql', None),
        'database': instance.get('database', None),
        'schemas': instance.get('schemas', None),
        'evidence': instance.get('evidence', None)
    }
    return new_instance

def load_data(data_file_path,
                limit,
                queries_dir=None,
                external_knowledge_dir=None,
                schemas_path_df=None,
                available_dbs=None,
                sqlites_file_dir=None,
                dataset_type='lite'
              ):
        """
        Load the Spider2 dataset.

        Returns:
            List of examples
        """
        # check the exisitace of the file
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"File not found: {data_file_path}")
        
        # Read the JSON file
        data = read_json_file(data_file_path)
            
        # Apply limit if specified
        if limit is not None:
            data = data[:limit]

        results_data =[]
        
        for count,value in enumerate(data):

            instance = value.copy()  # Create a copy of the instance to avoid modifying the original data

            instance_id = instance.get('instance_id')

            if instance_id is None:
                logger.warning("No instance ID found for example")
                continue

            # Check if the database exists for the given instance
            database_name = instance.get('db')
            if database_name is None:
                logger.warning(f"No database name found for instance {instance_id}")
                continue

            # Ensure the question key name is consistent
            instance = sync_question_key_name(instance)

            query = get_sql_query_per_instance(instance_id, queries_dir)

            if query is None:
                continue
            
            # Add the SQL query to the instance
            instance['sql'] = query

            external_knowledge = instance.get('external_knowledge', None)

            instance['evidence'] = get_external_knowledge_instance(external_knowledge, external_knowledge_dir)

            # Check if the database schema exists for the given instance
            db_schemas = get_database_schema(database_name, schemas_path_df, available_dbs)
            if db_schemas is None:
                continue

            database_cat, db_schemas_path = db_schemas

            if database_cat == 'sqlite':
                
                sqlite_path = get_sqlite_db_file_path(sqlites_file_dir, database_name)
                
                if not sqlite_path:
                    logger.warning(f"SQLite database file not found for instance {instance_id} with database name {database_name}")
                    continue

                instance['database'] = {
                    'name': database_name,
                    'path': sqlite_path,
                    'type': database_cat
                }
            elif database_cat == 'snowflake':
                instance['database'] = {
                    'name': database_name,
                    'path': 'Call the snowflake API to get the database',
                    'type': database_cat
                }
            elif database_cat == 'bigquery':
                instance['database'] = {
                    'name': database_name,
                    'path': 'Call the snowflake API to get the database',
                    'type': database_cat
                }
            else:
                logger.warning(f"Unknown database category for instance {instance_id} with database name {database_name} with category {database_cat}")
                continue

            instance['schemas'] = []
            for db_schema in db_schemas_path:
                instance['schemas'].append({
                    'name': db_schema.split(os.sep)[-2],  # Extract the database name from the path
                    'path': db_schema
                })

            instance['dataset'] = f'spider2-{dataset_type}'

            # Standardize the instance and append it to the results
            results_data.append(standarize_spider2_instance(instance, count))

        logger.info(f"Loaded {len(results_data)} examples from Spider2 dataset")
        
        logger.info("IF the number of loaded Data are less than what you expected, is because of the missing SQL queries in GOLD Directory")
        return results_data

if __name__ == "__main__":
    data_path = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'spider2-lite.jsonl')
    # Data/Spider2/spider2-lite/evaluation_suite/gold/sql
    queries_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'evaluation_suite', 'gold', 'sql')
    # Data/Spider2/spider2-lite/resource/documents
    external_knowledge_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'resource', 'documents')
    # Data/Spider2/spider2-lite/resource/databases/spider2-localdb
    sqlite_file_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'resource', 'databases', 'spider2-localdb')

    resulted_data = load_data(data_path,
                limit=None,
                queries_dir=queries_dir,
                external_knowledge_dir=external_knowledge_dir,
                schemas_path_df= get_spider2_files(
                    dataset_dir=os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite'),
                    category='lite',
                    available_dbs=['sqlite', 'snowflake', 'bigquery'],
                    method='json'
                ),
                available_dbs=['sqlite', 'snowflake'],
                sqlites_file_dir=sqlite_file_dir,
                dataset_type='lite'
                )
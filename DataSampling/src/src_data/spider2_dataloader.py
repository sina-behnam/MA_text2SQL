import glob
import os

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
        
    return db_paths

def get_spider2_files(dataset_dir,category,available_dbs=None,method=None):
    """
    Get the path to the Spider2 dataset directory.
    """
    if category == 'snow':
        raise NotImplementedError("Snowflake database preparation is not implemented yet.")
    elif category == 'lite':
        return prepare_spider2_lite_files(dataset_dir, available_dbs,method)
    else:
        raise ValueError(f"Unknown category: {category}")
    

import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,)
logger = logging.getLogger(__name__)
# disable the logger
# logger.setLevel(logging.CRITICAL)
# enable the logger
logger.setLevel(logging.INFO)

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
        
def get_database_schema(database_name, db_schemas_path_dict, available_dbs=None):
    """
    Check if the database schema exists for a given database name.
    This function checks if the database name exists in the provided database schemas path dictionary.

    Args:
        database_name: Name of the database
        db_schemas_path: Dictionary containing the paths to the database schemas
        available_dbs: List of available databases to check against

    Returns:
        bool: True if the database schema exists, False otherwise
        list: List of paths to the database schema files
    """
    for available_db in available_dbs:
        if available_db in db_schemas_path_dict:
            if database_name in db_schemas_path_dict[available_db]:
                return True,available_db,db_schemas_path_dict[available_db][database_name]
        else:
            return None,available_db,None
    return False,None,None

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
        return True,sqlite_db_file_path
    return False,None

def load_data(data_file_path,
                limit,
                queries_dir=None,
                external_knowledge_dir=None,
                db_schemas_path_dict=None,
                available_dbs=None,
                sqlites_file_dir=None
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
        
        for instance in data:

            instance_id = instance.get('instance_id')

            if instance_id is None:
                logger.warning("No instance ID found for example")
                continue

            # Ensure the question key name is consistent
            instance = sync_question_key_name(instance)

            query = get_sql_query_per_instance(instance_id, queries_dir)

            if query is None:
                logger.warning(f"No SQL query found for instance {instance_id}")
                continue
            # Add the SQL query to the instance
            instance['sql'] = query

            external_knowledge = instance.get('external_knowledge', None)

            instance['external_knowledge'] = get_external_knowledge_instance(external_knowledge, external_knowledge_dir)

            # Check if the database schema exists for the given instance
            database_name = instance.get('db')
            if database_name is None:
                logger.warning(f"No database name found for instance {instance_id}")
                continue

            # Check if the database schema exists for the given instance
            db_schema_exists, dataset_cat ,db_schemas_path = get_database_schema(database_name, db_schemas_path_dict, available_dbs)
            if db_schema_exists == None:
                logger.warning(f"The dataset is exluded therefore this dataset not included in the schema path: {dataset_cat} for instance {instance_id} with database name {database_name}")
                continue
            elif db_schema_exists == False:
                logger.warning(f"Database schema not found for instance {instance_id} with database name {database_name}")
                continue

            if dataset_cat == 'sqlite':
                is_sqlite_exist, sqlite_path = get_sqlite_db_file_path(sqlites_file_dir, database_name)
                if not is_sqlite_exist:
                    logger.warning(f"SQLite database file not found for instance {instance_id} with database name {database_name}")
                    continue
                instance['sqlite_db_path'] = sqlite_path
            
            instance['dataset_category'] = dataset_cat
            instance['database_schema'] = db_schemas_path
            results_data.append(instance)

        logger.info(f"Loaded {len(results_data)} examples from Spider2 dataset")
        
        logger.info("IF the number of loaded Data are less than what you expected, is because of the missing SQL queries in GOLD Directory")
        return results_data

if __name__ == '__main__':

    SPIDER2_DATASET_PATH = '/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/Data/Spider2'

    spider2_schemas_path = get_spider2_files(SPIDER2_DATASET_PATH, 'lite', available_dbs=['snowflake', 'sqlite','bigquery'], method='csv')

    data_path = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'spider2-lite.jsonl')
    # Data/Spider2/spider2-lite/evaluation_suite/gold/sql
    queries_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'evaluation_suite', 'gold', 'sql')
    # Data/Spider2/spider2-lite/resource/documents
    external_knowledge_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'resource', 'documents')
    # Data/Spider2/spider2-lite/resource/databases/spider2-localdb
    sqlite_file_dir = os.path.join(SPIDER2_DATASET_PATH, 'spider2-lite', 'resource', 'databases', 'spider2-localdb')
    # Load the data
    data = load_data(data_path,
                    limit=None,
                    queries_dir=queries_dir,
                    external_knowledge_dir=external_knowledge_dir,
                    db_schemas_path_dict=spider2_schemas_path,
                    available_dbs=['snowflake', 'sqlite','bigquery'],
                    sqlites_file_dir=sqlite_file_dir
                    )

    print(data)
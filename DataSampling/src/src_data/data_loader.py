import os
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import argparse
import glob
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('text2sql_dataloader')

# Custom filter to only allow specific log levels
class CustomLevelFilter(logging.Filter):
    def __init__(self, allowed_levels):
        super().__init__()
        self.allowed_levels = allowed_levels

    def filter(self, record):
        # Only allow messages whose level is exactly in the allowed_levels list.
        return record.levelno in self.allowed_levels

class BaseDataset:
    """
    Base dataset class for Text2SQL tasks.
    This class provides the common functionality for all dataset implementations.
    """
    
    @staticmethod
    def configure_logger(levels=None):
        """
        Configure the logger to show only messages whose level is exactly in the levels list.
        
        Args:
            levels: List of logging levels to display (e.g., [logging.INFO]) 
                    If None, the logger is reset to default behavior.
        
        Example:
            # Only INFO messages will be shown (WARNING and ERROR are ignored)
            Spider2Dataset.configure_logger([logging.INFO])
            
            # Only INFO and ERROR messages will be shown (WARNING messages are filtered out)
            Spider2Dataset.configure_logger([logging.INFO, logging.ERROR])
        """
        logger = logging.getLogger('text2sql_dataloader')
        
        # Remove all existing handlers and filters
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create a new StreamHandler for console output
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Configure logger level and filters
        if levels:
            # Set logger to the minimum level to capture all desired logs
            logger.setLevel(min(levels))
            
            # Create and add level filter to the handler (not the logger)
            custom_filter = CustomLevelFilter(levels)
            console_handler.addFilter(custom_filter)
            
            # Ensure logs aren't duplicated through parent loggers
            logger.propagate = False
            
            # Test log to confirm configuration
            logger.info(f"Logger configured to show these levels: {[logging.getLevelName(lvl) for lvl in levels]}")
        else:
            # Default behavior - show INFO and above
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.info("Logger reset to default behavior (showing INFO and above)")
    
    def __init__(self, 
                 base_dir: str,
                 dataset_name: str,
                 split: str = 'train',
                 limit: Optional[int] = None):
        """
        Initialize the base dataset.
        
        Args:
            base_dir: Base directory containing the data
            dataset_name: Name of the dataset
            split: Data split ('train', 'dev', 'test')
            limit: Optional limit on the number of examples to load
        """
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.data = []
        self.db_schemas = {}
        
    def load_data(self) -> List[Dict]:
        """
        Load dataset from files.
        Should be implemented by subclasses.
        
        Returns:
            List of examples
        """
        raise NotImplementedError
    
    def load_schemas(self) -> Dict:
        """
        Load database schemas.
        Should be implemented by subclasses.
        
        Returns:
            Dictionary mapping database names to schema information
        """
        raise NotImplementedError
    
    def get_db_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Get a SQLite database connection.
        
        Args:
            db_name: Name of the database
        
        Returns:
            SQLite connection object
        """
        raise NotImplementedError
    
    def get_example_by_idx(self, idx: int) -> Dict:
        """
        Get a specific example by index.
        
        Args:
            idx: Index of the example
        
        Returns:
            Example as a dictionary
        """
        if not self.data:
            self.load_data()
        
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range (0 to {len(self.data)-1})")
        
        return self.data[idx]
    
    def get_schema_by_db_name(self, db_name: str) -> Dict:
        """
        Get schema for a specific database.
        
        Args:
            db_name: Name of the database
        
        Returns:
            Schema information as a dictionary
        """
        db_name = db_name.lower()

        if not self.db_schemas:
            self.load_schemas()
        
        if db_name not in self.db_schemas:
            raise ValueError(f"Database '{db_name}' not found in schema")
        
        return self.db_schemas[db_name]
    
    def __len__(self) -> int:
        """
        Get number of examples in the dataset.
        
        Returns:
            Number of examples
        """
        if not self.data:
            self.load_data()
        
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get example by index.
        
        Args:
            idx: Index of the example
        
        Returns:
            Example as a dictionary
        """
        return self.get_example_by_idx(idx)


class SpiderDataset(BaseDataset):
    """
    Spider dataset implementation for Text2SQL tasks.
    """
    
    def __init__(self, 
                 base_dir: str,
                 split: str = 'train',
                 include_others: bool = True,
                 limit: Optional[int] = None):
        """
        Initialize the Spider dataset.
        
        Args:
            base_dir: Base directory containing the Spider data
            split: Data split ('train', 'dev', 'test')
            include_others: Whether to include train_others data for training
            limit: Optional limit on the number of examples to load
        """
        super().__init__(base_dir, 'spider', split, limit)
        self.include_others = include_others and split == 'train'
        self.tables_file = os.path.join(base_dir, 'tables.json')
        self.db_dir = os.path.join(base_dir, 'database')
        
    def load_data(self) -> List[Dict]:
        """
        Load Spider dataset from files.
        
        Returns:
            List of examples
        """
        if self.data:
            return self.data
        
        # Determine which files to load based on split
        files_to_load = []
        if self.split == 'train':
            files_to_load.append(os.path.join(self.base_dir, 'train_spider.json'))
            if self.include_others:
                files_to_load.append(os.path.join(self.base_dir, 'train_others.json'))
        elif self.split == 'dev':
            files_to_load.append(os.path.join(self.base_dir, 'dev.json'))
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Load data from files
        all_data = []
        for file_path in files_to_load:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        # Apply limit if specified
        if self.limit is not None:
            all_data = all_data[:self.limit]
        
        self.data = all_data
        return self.data
    
    def load_schemas(self) -> Dict:
        """
        Load Spider database schemas from tables.json.
        
        Returns:
            Dictionary mapping database names to schema information
        """ 
        if self.db_schemas:
            return self.db_schemas
        
        if not os.path.exists(self.tables_file):
            raise FileNotFoundError(f"Tables file not found: {self.tables_file}")
        
        with open(self.tables_file, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        # Process schemas
        db_schemas = {}
        for entry in tables_data:
            try:
                db_id = entry['db_id']
                tables = entry['table_names_original']
                table_names = entry['table_names']
                
                # Process columns and table relationships
                columns = []
                table_to_columns = {i: [] for i in range(len(tables))}
                
                # Check how column information is structured in the Spider dataset
                # In Spider, column_names often has format [(table_idx, column_name), ...]
                column_names = entry['column_names']
                column_names_original = entry['column_names_original']
                column_types = entry.get('column_types', [])
                
                # Extract table indices and column names from the structure
                for i, (col_entry, col_orig_entry) in enumerate(zip(column_names, column_names_original)):
                    # Handle different possible structures
                    if isinstance(col_entry, (list, tuple)) and len(col_entry) >= 2:
                        # Format: [table_idx, column_name]
                        table_idx = col_entry[0]
                        col_name = col_entry[1]
                    else:
                        # Fallback if structure is unexpected
                        table_idx = -1
                        col_name = col_entry
                    
                    if isinstance(col_orig_entry, (list, tuple)) and len(col_orig_entry) >= 2:
                        col_orig_name = col_orig_entry[1]
                    else:
                        col_orig_name = col_orig_entry
                    
                    # Get column type if available
                    col_type = column_types[i] if i < len(column_types) else "text"
                    
                    columns.append({
                        'id': i,
                        'name': col_name,
                        'original_name': col_orig_name,
                        'table_idx': table_idx,
                        'table': tables[table_idx] if table_idx >= 0 and table_idx < len(tables) else None,
                        'type': col_type
                    })
                    
                    if table_idx >= 0 and table_idx < len(tables):
                        table_to_columns[table_idx].append(i)
                
                # Build schema
                schema = {
                    'db_id': db_id,
                    'tables': [{'id': i, 'name': name, 'original_name': orig_name} 
                            for i, (name, orig_name) in enumerate(zip(table_names, tables))],
                    'columns': columns,
                    'table_to_columns': table_to_columns,
                    'foreign_keys': entry.get('foreign_keys', []),
                    'primary_keys': entry.get('primary_keys', [])
                }
                
                db_schemas[db_id] = schema
            except Exception as e:
                logger.error(f"Error processing schema for database {entry.get('db_id', 'unknown')}: {str(e)}")
                # Continue to next schema instead of failing completely
                continue
            
        self.db_schemas = db_schemas
        logger.info(f"Loaded {len(self.db_schemas)} schemas from Spider dataset")
        return self.db_schemas
    
    def get_db_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Get a SQLite database connection for a Spider database.
        
        Args:
            db_name: Name of the database
        
        Returns:
            SQLite connection object
        """
        db_path = os.path.join(self.db_dir, db_name, f"{db_name}.sqlite")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        return conn
    
    def get_tables_for_schema(self, db_name: str) -> Dict[str, pd.DataFrame]:
        """
        Get tables for a database schema.
        
        Args:
            db_name: Name of the database
        
        Returns:
            Dictionary mapping table names to DataFrames
        """
        schema = self.get_schema_by_db_name(db_name)
        conn = self.get_db_connection(db_name)
        
        tables = {}
        for table in schema['tables']:
            table_name = table['original_name']
            try:
                query = f"SELECT * FROM {table_name} LIMIT 5;"
                tables[table_name] = pd.read_sql_query(query, conn)
            except Exception as e:
                logger.warning(f"Warning: Could not read table {table_name}: {e}")
                tables[table_name] = pd.DataFrame()
        
        conn.close()
        return tables
    
    def get_sample_with_schema(self, idx: int) -> Dict:
        """
        Get a sample with its schema.
        
        Args:
            idx: Index of the example
        
        Returns:
            Dictionary containing example and its schema
        """
        example = self.get_example_by_idx(idx)
        db_name = example['db_id']
        schema = self.get_schema_by_db_name(db_name)
        
        return {
            'example': example,
            'schema': schema
        }
    
    def get_table_schemas_text(self, db_name: str) -> str:
        """
        Get a text representation of the schema for a database.
        Useful for prompting LLMs.
        
        Args:
            db_name: Name of the database
        
        Returns:
            Text representation of the schema
        """
        schema = self.get_schema_by_db_name(db_name)
        tables = schema['tables']
        table_to_columns = schema['table_to_columns']
        columns = schema['columns']
        
        lines = [f"Database: {db_name}"]
        lines.append("Tables:")
        
        for table in tables:
            table_id = table['id']
            table_name = table['original_name']
            lines.append(f"  {table_name}")
            
            # Add columns for this table
            col_indexes = table_to_columns[table_id]
            for col_idx in col_indexes:
                col = columns[col_idx]
                col_name = col['original_name']
                col_type = col['type']
                pk_flag = "PRIMARY KEY" if col_idx in schema['primary_keys'] else ""
                lines.append(f"    {col_name} ({col_type}) {pk_flag}")
        
        # Add foreign key constraints
        if schema['foreign_keys']:
            lines.append("Foreign Keys:")
            for fk in schema['foreign_keys']:
                from_col = columns[fk[0]]['original_name']
                from_table = tables[columns[fk[0]]['table_idx']]['original_name']
                to_col = columns[fk[1]]['original_name']
                to_table = tables[columns[fk[1]]['table_idx']]['original_name']
                lines.append(f"  {from_table}.{from_col} -> {to_table}.{to_col}")
        
        return "\n".join(lines)

class BirdDataset(BaseDataset):
    """
    Bird dataset implementation for Text2SQL tasks.
    """
    
    def __init__(self, 
                 base_dir: str,
                 split: str = 'train',
                 limit: Optional[int] = None,
                 load_evidence: bool = True):
        """
        Initialize the Bird dataset.
        
        Args:
            base_dir: Base directory containing the Bird data
            split: Data split ('train', 'dev', 'test')
            limit: Optional limit on the number of examples to load
            load_evidence: Whether to load evidence from the dataset
        """
        super().__init__(base_dir, 'bird', split, limit)
        self.data_file = os.path.join(base_dir, f"{split}.json")
        self.db_dir = os.path.join(base_dir, f"{split}_databases")
        self.load_evidence = load_evidence
        
    def load_data(self) -> List[Dict]:
        """
        Load Bird dataset from files.
        
        Returns:
            List of examples
        """
        if self.data:
            return self.data
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Apply limit if specified
        if self.limit is not None:
            data = data[:self.limit]
        
        self.data = data
        return self.data
    
    def load_schemas(self) -> Dict:
        """
        Load Bird database schemas from database description files.
        
        Returns:
            Dictionary mapping database names to schema information
        """ 
        if self.db_schemas:
            return self.db_schemas
        
        if not os.path.exists(self.db_dir):
            raise FileNotFoundError(f"Database directory not found: {self.db_dir}")
        
        db_schemas = {}
        # Get all database directories
        db_dirs = [d for d in os.listdir(self.db_dir) if os.path.isdir(os.path.join(self.db_dir, d))]
        
        for db_id in db_dirs:
            db_path = os.path.join(self.db_dir, db_id)
            desc_dir = os.path.join(db_path, "database_description")
            
            # Flag to track if description directory exists
            has_description_files = os.path.exists(desc_dir)
            
            # Extract table information from CSV files
            tables = []
            columns = []
            table_to_columns = {}
            foreign_keys = []
            primary_keys = []
            
            # Extract schema using SQLite metadata
            try:
                conn = self.get_db_connection(db_id)
                cursor = conn.cursor()
                
                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_names = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
                
                for table_idx, table_name in enumerate(table_names):
                    tables.append({
                        'id': table_idx,
                        'name': table_name.lower(),
                        'original_name': table_name
                    })
                    table_to_columns[table_idx] = []
                    
                    # Get columns for this table - handle reserved keywords by quoting
                    try:
                        cursor.execute(f'PRAGMA table_info("{table_name}");')
                        cols = cursor.fetchall()
                    except sqlite3.OperationalError:
                        # Try with square brackets if double quotes don't work
                        try:
                            cursor.execute(f"PRAGMA table_info([{table_name}]);")
                            cols = cursor.fetchall()
                        except sqlite3.OperationalError as e:
                            logger.error(f"Could not get columns for table {table_name}: {e}")
                            cols = []
                    
                    for col in cols:
                        col_id = len(columns)
                        col_name = col[1]
                        col_type = col[2]
                        is_pk = col[5] == 1  # The primary key flag
                        
                        columns.append({
                            'id': col_id,
                            'name': col_name.lower(),
                            'original_name': col_name,
                            'table_idx': table_idx,
                            'table': table_name,
                            'type': col_type
                        })
                        
                        table_to_columns[table_idx].append(col_id)
                        
                        if is_pk:
                            primary_keys.append(col_id)
                
                # Get foreign keys - handle reserved keywords
                for table_name in table_names:
                    try:
                        cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
                        fks = cursor.fetchall()
                    except sqlite3.OperationalError:
                        try:
                            cursor.execute(f"PRAGMA foreign_key_list([{table_name}]);")
                            fks = cursor.fetchall()
                        except sqlite3.OperationalError as e:
                            logger.error(f"Could not get foreign keys for table {table_name}: {e}")
                            fks = []
                    
                    if fks:
                        table_idx = next((i for i, t in enumerate(tables) if t['original_name'] == table_name), None)
                        if table_idx is not None:
                            for fk in fks:
                                from_col_name = fk[3]
                                to_table = fk[2]
                                to_col_name = fk[4]
                                
                                # Find column ids
                                from_col_id = next((c['id'] for c in columns if c['table_idx'] == table_idx and c['original_name'] == from_col_name), None)
                                to_table_idx = next((i for i, t in enumerate(tables) if t['original_name'] == to_table), None)
                                to_col_id = next((c['id'] for c in columns if c['table_idx'] == to_table_idx and c['original_name'] == to_col_name), None)
                                
                                if from_col_id is not None and to_col_id is not None:
                                    foreign_keys.append([from_col_id, to_col_id])
                
                conn.close()
            except Exception as e:
                logger.error(f"Error extracting schema for {db_id}: {e}")
                # Continue to next database instead of stopping completely
                continue
            
            # Add description information from CSV files if available
            if has_description_files:
                # Get all CSV files in the description directory
                csv_files = [f for f in os.listdir(desc_dir) if f.endswith('.csv')]
                
                # Process each CSV description file
                for csv_file in csv_files:
                    table_name = os.path.basename(csv_file).replace(".csv", "")
                    try:
                        # Try different encodings - this is key to fixing the encoding issues
                        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
                        desc_df = None
                        
                        for encoding in encodings_to_try:
                            try:
                                desc_df = pd.read_csv(os.path.join(desc_dir, csv_file), encoding=encoding, 
                                                    on_bad_lines='skip', engine='python')
                                break  # If successful, break the loop
                            except UnicodeDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"Error with encoding {encoding} for {csv_file}: {e}")
                                continue
                        
                        if desc_df is None:
                            logger.error(f"Could not read CSV file {csv_file} with any encoding")
                            continue
                            
                        # Find the table index
                        table_idx = next((i for i, t in enumerate(tables) if t['original_name'].lower() == table_name.lower()), None)
                        
                        if table_idx is not None:
                            # Add description to the table
                            tables[table_idx]['description_file'] = csv_file
                            
                            # Process each row in description file to find column descriptions
                            for _, row in desc_df.iterrows():
                                col_desc = {}
                                
                                # Extract column information from description
                                for col in desc_df.columns:
                                    if pd.notna(row[col]):
                                        col_desc[col] = row[col]
                                
                                if 'column_name' in col_desc:
                                    col_name = col_desc['column_name']
                                    # Find the column
                                    col_ids = table_to_columns.get(table_idx, [])
                                    for col_id in col_ids:
                                        if columns[col_id]['original_name'].lower() == col_name.lower():
                                            # Add description to column
                                            columns[col_id]['description'] = col_desc.get('column_description', '')
                                            columns[col_id]['value_description'] = col_desc.get('value_description', '')
                                            break
                    except Exception as e:
                        logger.error(f"Error processing description CSV for {table_name}: {e}")
            
            # Build schema
            schema = {
                'db_id': db_id,
                'tables': tables,
                'columns': columns,
                'table_to_columns': table_to_columns,
                'foreign_keys': foreign_keys,
                'primary_keys': primary_keys,
                'has_description_files': has_description_files
            }
            
            db_schemas[db_id] = schema
        
        self.db_schemas = db_schemas
        return self.db_schemas
    
    def get_db_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Get a SQLite database connection for a Bird database.
        
        Args:
            db_name: Name of the database
        
        Returns:
            SQLite connection object
        """
        # Updated path to match the actual Bird dataset structure
        db_path = os.path.join(self.db_dir, db_name, f"{db_name}.sqlite")
        
        if not os.path.exists(db_path):
            # Try alternative path patterns
            alt_paths = [
                os.path.join(self.db_dir, db_name, "sqlite", f"{db_name}.sqlite"),
                os.path.join(self.db_dir, db_name, "database", f"{db_name}.sqlite"),
                os.path.join(self.db_dir, db_name, "db", f"{db_name}.sqlite")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    db_path = path
                    break
            else:
                raise FileNotFoundError(f"Database file not found for {db_name}. Tried: {db_path} and alternatives")
        
        conn = sqlite3.connect(db_path)
        return conn
    
    def get_tables_for_schema(self, db_name: str) -> Dict[str, pd.DataFrame]:
        """
        Get tables for a database schema.
        
        Args:
            db_name: Name of the database
        
        Returns:
            Dictionary mapping table names to DataFrames
        """
        schema = self.get_schema_by_db_name(db_name)
        conn = self.get_db_connection(db_name)
        
        tables = {}
        for table in schema['tables']:
            table_name = table['original_name']
            try:
                # Handle reserved keywords by quoting table names
                query = f'SELECT * FROM "{table_name}" LIMIT 5;'
                tables[table_name] = pd.read_sql_query(query, conn)
            except Exception as e:
                try:
                    # Try with square brackets if double quotes don't work
                    query = f"SELECT * FROM [{table_name}] LIMIT 5;"
                    tables[table_name] = pd.read_sql_query(query, conn)
                except Exception as e2:
                    logger.warning(f"Warning: Could not read table {table_name}: {e2}")
                    tables[table_name] = pd.DataFrame()
        
        conn.close()
        return tables
    
    def get_evidence_for_example(self, idx: int) -> str:
        """
        Get the evidence text for a specific example.
        
        Args:
            idx: Index of the example
        
        Returns:
            Evidence text
        """
        example = self.get_example_by_idx(idx)
        return example.get('evidence', '')
    
    def get_sample_with_schema_and_evidence(self, idx: int) -> Dict:
        """
        Get a sample with its schema and evidence.
        
        Args:
            idx: Index of the example
        
        Returns:
            Dictionary containing example, schema, and evidence
        """
        example = self.get_example_by_idx(idx)
        db_name = example['db_id']
        
        try:
            schema = self.get_schema_by_db_name(db_name)
        except Exception as e:
            logger.warning(f"Warning: Could not load schema for {db_name}: {e}")
            schema = {"error": str(e)}
        
        return {
            'example': example,
            'schema': schema,
            'evidence': example.get('evidence', '')
        }
    
    def get_table_schemas_text(self, db_name: str) -> str:
        """
        Get a text representation of the schema for a database.
        Useful for prompting LLMs.
        
        Args:
            db_name: Name of the database
        
        Returns:
            Text representation of the schema
        """
        try:
            schema = self.get_schema_by_db_name(db_name)
        except Exception as e:
            return f"Error loading schema for {db_name}: {str(e)}"
            
        tables = schema['tables']
        table_to_columns = schema['table_to_columns']
        columns = schema['columns']
        
        lines = [f"Database: {db_name}"]
        lines.append("Tables:")
        
        for table in tables:
            table_id = table['id']
            table_name = table['original_name']
            lines.append(f"  {table_name}")
            
            # Add description if available
            if 'description' in table and table['description']:
                lines.append(f"    Description: {table['description']}")
            
            # Add columns for this table
            col_indexes = table_to_columns.get(table_id, [])
            for col_idx in col_indexes:
                if col_idx >= len(columns):
                    continue  # Skip if column index is out of range
                    
                col = columns[col_idx]
                col_name = col['original_name']
                col_type = col['type']
                pk_flag = "PRIMARY KEY" if col_idx in schema.get('primary_keys', []) else ""
                
                # Base column info
                col_line = f"    {col_name} ({col_type}) {pk_flag}"
                lines.append(col_line)
                
                # Add description if available
                if 'description' in col and col['description']:
                    lines.append(f"      Description: {col['description']}")
                
                # Add value description if available
                if 'value_description' in col and col['value_description']:
                    lines.append(f"      Values: {col['value_description']}")
        
        # Add foreign key constraints
        if schema.get('foreign_keys', []):
            lines.append("Foreign Keys:")
            for fk in schema['foreign_keys']:
                try:
                    from_col = columns[fk[0]]['original_name']
                    from_table = tables[columns[fk[0]]['table_idx']]['original_name']
                    to_col = columns[fk[1]]['original_name']
                    to_table = tables[columns[fk[1]]['table_idx']]['original_name']
                    lines.append(f"  {from_table}.{from_col} -> {to_table}.{to_col}")
                except (IndexError, KeyError) as e:
                    # Skip invalid foreign key references
                    continue
        
        return "\n".join(lines)
    
    def analyze_db_schema(self, db_name: str) -> Dict:
        """
        Analyze a database schema and return detailed statistics.
        
        Args:
            db_name: Name of the database
            
        Returns:
            Dictionary with database schema statistics
        """
        schema = self.get_schema_by_db_name(db_name)
        
        # Calculate basic statistics
        num_tables = len(schema['tables'])
        num_columns = len(schema['columns'])
        num_fk = len(schema.get('foreign_keys', []))
        num_pk = len(schema.get('primary_keys', []))
        
        # Calculate columns per table
        cols_per_table = {}
        for table in schema['tables']:
            table_id = table['id']
            table_name = table['original_name']
            cols_per_table[table_name] = len(schema['table_to_columns'].get(table_id, []))
        
        # Count column types
        column_types = {}
        for col in schema['columns']:
            col_type = col.get('type', '').upper()
            if col_type in column_types:
                column_types[col_type] += 1
            else:
                column_types[col_type] = 1
        
        # Create analysis result
        analysis = {
            'db_id': db_name,
            'num_tables': num_tables,
            'num_columns': num_columns,
            'num_foreign_keys': num_fk,
            'num_primary_keys': num_pk,
            'avg_columns_per_table': num_columns / num_tables if num_tables else 0,
            'columns_per_table': cols_per_table,
            'column_types': column_types,
            'has_description_files': schema.get('has_description_files', False),
        }
        
        return analysis

class Spider2Dataset(BaseDataset):

    def __init__(self, base_dir, split = 'train',
                  limit = None, is_snow :bool = False,
                  is_lite : bool = False,
                  ):
        super().__init__(base_dir, 'spider2', split, limit)

        self.data_directory = None

        self.data_to_load = self._init_file_paths_(base_dir,is_snow,is_lite)

        # self.database_cat_set = ['bigquery','snowflake','sqlite']
        self.database_cat_set = ['snowflake','sqlite']

    def _init_file_paths_(self, base_dir,is_snow,is_lite) -> str:
        """
        Initialize file paths for Spider2 dataset.

        Args:
            base_dir: Base directory containing the Spider2 data
            is_snow: Whether to use the Spider2-Snow dataset
            is_lite: Whether to use the Spider2-Lite dataset
        Returns:
            Path to the data file
        """
        if is_snow:
            print("Loading Spider2-Snow dataset")
            # data
            # base_dir/spider2-snow/spider2-snow.jsonl
            self.data_directory = os.path.join(self.base_dir, 'spider2-snow')
            # base_dir/spider2-snow/resource/databases
            self.db_dir = os.path.join(base_dir, 'spider2-snow', 'resource','databases')
            # base_dir/spider2-snow/evaluation_suite/gold/sql
            self.quires_dir = os.path.join(base_dir, 'spider2-snow', 'evaluation_suite','gold','sql')
            # base_dir/spider2-snow/resource/documents
            self.external_know_dir = os.path.join(base_dir, 'spider2-snow', 'resource','documents')

            return os.path.join(self.data_directory, 'spider2-snow.jsonl')
        
        elif not is_snow and is_lite:
            print("Loading Spider2-Lite dataset")
            # data
            # base_dir/spider2-lite/spider2-lite.jsonl
            self.data_directory = os.path.join(self.base_dir, 'spider2-lite')
            # base_dir/spider2-lite/resource/databases
            self.db_dir = os.path.join(base_dir, 'spider2-lite', 'resource','databases')
            # base_dir/spider2-lite/resource/documents
            self.external_know_dir = os.path.join(base_dir, 'spider2-lite', 'resource','documents')
            # base_dir/spider2-lite/evaluation_suite/gold/sql
            self.quires_dir = os.path.join(base_dir, 'spider2-lite', 'evaluation_suite','gold','sql')

            return os.path.join(self.data_directory, 'spider2-lite.jsonl')
        
        else:
            self.db_dir = os.path.join(base_dir, 'spider2', 'resource','databases')
            # self.external_know_dir = os.path.join(base_dir, 'spider2', 'resource','documents') 
            raise NotImplementedError("Spider2 dataset is not implemented for this configuration")

    def load_data(self) -> List[Dict]:
        """
        Load the Spider2 dataset.

        Returns:
            List of examples
        """

        if self.data:
            return self.data
        
        # Determine which files to load based on split
        files_to_load = []
        
        files_to_load.append(self.data_to_load)

        # Load data from files
        all_data = []
        for file_path in files_to_load:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    all_data.append(json.loads(line))  

        # Apply limit if specified
        if self.limit is not None:
            all_data = all_data[:self.limit]
        
        for instance in all_data:

            instance_id = instance.get('instance_id')

            if instance_id is None:
                logger.warning("No instance ID found for example")
                continue

            # Ensure the question key name is consistent
            instance = self.sync_question_key_name(instance)

            query = self.get_sql_query_per_instance(instance_id)

            if query is None:
                logger.warning(f"No SQL query found for instance {instance_id}")
                continue
            # Add the SQL query to the instance
            instance['sql'] = query

            external_knowledge = instance.get('external_knowledge', None)

            instance['external_knowledge'] = self.get_external_knowledge_instance(external_knowledge)
            
            self.data.append(instance)

        logger.info(f"Loaded {len(self.data)} examples from Spider2 dataset")
        
        logger.info("IF the number of loaded Data are less than what you expected, is because of the missing SQL queries in GOLD Directory")
        return self.data

    def load_schemas(self) -> Dict:
        """
        Load database schemas for Spider2 dataset from individual JSON files.

        Returns:
            Dictionary mapping database names to schema information
        """
        if self.db_schemas:
            return self.db_schemas

        if not os.path.exists(self.db_dir):
            raise FileNotFoundError(f"Database directory not found: {self.db_dir}")

        db_schemas = {}

        # Get all schema files and group by database name (determined from directory path)
        self.schema_files_by_db = {}
        my_separatetor = '_$_'
        for file in glob.glob(self.db_dir + '/**/*.json', recursive=True):
            # Get the category of the databases
            cat_path = file.split(self.db_dir + os.sep)[-1]
            cat_name_database = cat_path.split(os.sep)[0]
            # Get the len from the json file name up to the 
            if len(cat_path.split('/')) == 4:
                # it means that that include schema file too
                database_name = cat_path.split(os.sep)[1]
                schema_file = cat_path.split(os.sep)[2]
                database_name = cat_name_database + my_separatetor + database_name + my_separatetor + schema_file
            elif len(cat_path.split('/')) == 3:
                # it means that that there is no schema file
                database_name = cat_path.split(os.sep)[1]
                database_name = cat_name_database + my_separatetor + database_name 
            else:
                logger.warning(len(cat_path.split(os.sep)))
                logger.warning('Error: the path is not correct')
                raise ValueError(f"Error: the path is not correct {cat_path}")

            # appending the database name to the schema_files_by_db dictionary
            if database_name not in self.schema_files_by_db:
                self.schema_files_by_db[database_name] = []
            # appending the file to the schema_files_by_db dictionary
            self.schema_files_by_db[database_name].append(file)

        # Process schema files for each database
        for db_name, schema_files in self.schema_files_by_db.items():

            cat_name = str(db_name.split(my_separatetor)[0])
            # Check if the database category is in the allowed set
            if cat_name not in self.database_cat_set:
                continue

            # Initialize schema structure
            tables = []
            columns = []
            table_to_columns = {}

            # Extract the DB_NAME from db_name 
            # ! ATTENTION: we are only using the `database_name` part of the db_name
            db_name_parts = db_name.split(my_separatetor)[1]

            logger.info(f"Processing database {db_name_parts} with {len(schema_files)} schema files from {cat_name.upper()} category")

            for schema_path in schema_files:
                
                if not os.path.exists(schema_path):
                    logger.warning(f"Schema file not found: {schema_path}")
                    continue
                try:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        table_schema = json.load(f)

                    # Extract table information
                    table_full_name = table_schema.get('table_fullname', '')
                    # Extract just the last part if it contains dots
                    if '.' in table_full_name:
                        table_name = table_full_name.split('.')[-1]
                    else:
                        table_name = table_full_name

                    table_idx = len(tables)
                    tables.append({
                        'id': table_idx,
                        'name': table_name.lower(),
                        'original_name': table_full_name # This one should be use for tracking later on.
                    })

                    # Initialize column mapping for this table
                    table_to_columns[table_idx] = []

                    # Extract column information
                    column_names = table_schema.get('nested_column_names', [])
                    column_types = table_schema.get('nested_column_types', [])
                    descriptions = table_schema.get('description', [])

                    # Ensure lists have the same length
                    max_len = max(len(column_names), len(column_types), len(descriptions) if descriptions else 0)
                    column_names = column_names + [''] * (max_len - len(column_names))
                    column_types = column_types + ['TEXT'] * (max_len - len(column_types))
                    if descriptions:
                        descriptions = descriptions + [None] * (max_len - len(descriptions))
                    else:
                        descriptions = [None] * max_len

                    # Process columns
                    for i, (col_name, col_type, description) in enumerate(zip(column_names, column_types, descriptions)):
                        col_id = len(columns)
                        columns.append({
                            'id': col_id,
                            'name': col_name if col_name else '',
                            'original_name': col_name if col_name else '',
                            'table_idx': table_idx,
                            'table': table_name,
                            'type': col_type if col_type else 'ERROR',
                            'description': description
                        })

                        table_to_columns[table_idx].append(col_id)

                except Exception as e:
                    logger.error(f"Error processing schema file {schema_file} for database {db_name_parts}: {e}")
                    continue
                
            # Create the schema entry for this database
            db_schemas[db_name_parts] = {
                'db_id': db_name_parts,
                'db_cat': cat_name,
                'tables': tables,
                'columns': columns,
                'table_to_columns': table_to_columns,
                'foreign_keys': [],  # ! Empty list as we don't have FK info
                'primary_keys': []   # ! Empty list as we don't have PK info
            }

        self.db_schemas = db_schemas
        logger.info(f"Loaded {len(self.db_schemas)} schemas from Spider2 dataset")
        return self.db_schemas

    def get_schema_by_db_name(self, db_name: str) -> Optional[Dict]:
        """
        Get schema for a specific database in Spider2.
        
        Args:
            db_name: Name of the database
        
        Returns:
            Schema information as a dictionary
        """
        if not self.db_schemas:
            self.load_schemas()
        # Check if the database name is in the loaded schemas or either in the schema_files_by_db 
        # IF The db_name even wouldn't be in the schema_files_by_db, then it means reading the schema has some issues
        if db_name not in self.db_schemas:
            is_missing = True    
            for db_in_files in self.schema_files_by_db.keys():
                if db_name.lower() in db_in_files.lower():
                    is_missing = False
                    return None;
            
            if is_missing:
                raise ValueError(f"Database '{db_name}' not found in schema files, it means reading the schema has some issues")
        else:
            return self.db_schemas[db_name]

    def sync_question_key_name(self,instance : Dict) -> Dict:

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

    def get_sql_query_per_instance(self, instance_id : str) -> Optional[str]:
        """
        Get the SQL query for a specific instance ID.

        Args:
            instance_id: ID of the instance
        Returns:
            SQL query string or None if not found  
        """
        if self.quires_dir is not None:
            # Get the path to the SQL file
            sql_file_path = os.path.join(self.quires_dir, f"{instance_id}.sql")
            if not os.path.exists(sql_file_path):
                # logger.debug(f"SQL file not found for instance {instance_id}: {sql_file_path}")
                return None

            # Read the SQL query from the file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_query = f.read()
            
            return sql_query.strip()
        else:
            logger.warning(f"No queries directory specified for instance {instance_id}")

        return None
    
    def get_external_knowledge_instance(self, external_knowledge_file: str) -> Optional[str]:
        """
        Get external knowledge for a specific instance.

        Args:
            external_knowledge_file: Name of the external knowledge file
        
        Returns:
            External knowledge string or None if not found
        """

        if external_knowledge_file is not None and (external_knowledge_file != '' or external_knowledge_file != []):
            # Load the external knowledge from the directory
            external_knowledge_path = os.path.join(self.external_know_dir, external_knowledge_file)
            if not os.path.exists(external_knowledge_path):
                raise FileNotFoundError(f"External knowledge file not found: {external_knowledge_path}")
            # IT is a .md file 
            with open(external_knowledge_path, 'r', encoding='utf-8') as f:
                external_knowledge_data = f.read()
            
            return external_knowledge_data.strip()
        else:
            return None


class DataLoader:
    """
    Data loader for Text2SQL tasks.
    """
    
    @staticmethod
    def get_dataset(dataset_name: str, **kwargs) -> BaseDataset:
        """
        Get a dataset instance by name.
        
        Args:
            dataset_name: Name of the dataset
            **kwargs: Additional arguments to pass to the dataset constructor
        
        Returns:
            Dataset instance
        """
        if dataset_name.lower() == 'spider':
            return SpiderDataset(**kwargs)
        elif dataset_name.lower() == 'spider2':
            return Spider2Dataset(**kwargs,is_lite=True)
        elif dataset_name.lower() == 'bird':
            return BirdDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


def test_spider2_dataset():
    """
    Test function for Spider2 dataset.
    """

    dataset = DataLoader.get_dataset('spider2',
                            base_dir='Data/Spider2',
                            split='train',
                            is_snow=False, is_lite=True)
    
    # Load the dataset
    data = dataset.load_data()
    print(f"Loaded {len(data)} examples from Spider2 dataset")


# Usage example
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Load and process Text2SQL datasets')
    parser.add_argument('--dataset', type=str, choices=['bird', 'spider','spider2'], required=True,
                        help='Dataset type to load (bird or spider)')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing the dataset')
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'],
                        help='Dataset split to load (train/dev/test)')
    parser.add_argument('--output', type=str, default='outputs/schema.json',
                        help='Output file path for schema information')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    
    args = parser.parse_args()
    
    # Set the logging level based on command-line argument
    logging_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger('text2sql_dataloader')
    logger.setLevel(logging_level)
    
    logger.info(f"Logging level set to {args.log_level}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load the dataset based on command-line arguments
    dataset = DataLoader.get_dataset(args.dataset, base_dir=args.base_dir, split=args.split)
    
    # Load data and schemas
    data = dataset.load_data()
    schema = dataset.load_schemas()
    
    # save the loaded schema to a json file
    with open(args.output, 'w') as f:
        json.dump(schema, f, indent=4)
    
    logger.info(f"Saved schema for {args.dataset} to {args.output}")


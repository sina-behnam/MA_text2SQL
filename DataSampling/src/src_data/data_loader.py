import os
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd


class BaseDataset:
    """
    Base dataset class for Text2SQL tasks.
    This class provides the common functionality for all dataset implementations.
    """
    
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
            db_id = entry['db_id']
            tables = entry['table_names_original']
            table_names = entry['table_names']
            
            # Process columns and table relationships
            columns = []
            table_to_columns = {i: [] for i in range(len(tables))}
            
            for i, (col_name, col_orig_name, table_idx) in enumerate(zip(
                entry['column_names'], 
                entry['column_names_original'], 
                entry['column_to_table']
            )):
                columns.append({
                    'id': i,
                    'name': col_name,
                    'original_name': col_orig_name,
                    'table_idx': table_idx,
                    'table': tables[table_idx] if table_idx >= 0 else None,
                    'type': entry['column_types'][i]
                })
                
                if table_idx >= 0:
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
        
        self.db_schemas = db_schemas
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
                print(f"Warning: Could not read table {table_name}: {e}")
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
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


# Usage example
if __name__ == "__main__":
    # Example usage
    base_dir = "/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/Data/data/spider"
    
    # Create a dataset
    dataset = DataLoader.get_dataset('spider', base_dir=base_dir, split='train')
    
    # Load data and schemas
    data = dataset.load_data()
    schemas = dataset.load_schemas()
    
    # Print statistics
    print(f"Loaded {len(data)} examples")
    print(f"Loaded {len(schemas)} database schemas")
    
    # Get a sample
    if data:
        sample = dataset[0]
        db_name = sample['db_id']
        
        # Print schema as text
        schema_text = dataset.get_table_schemas_text(db_name)
        print(f"\nExample schema for '{db_name}':")
        print(schema_text)
        
        # Print the question and SQL
        print(f"\nQuestion: {sample['question']}")
        print(f"SQL: {sample['query']}")
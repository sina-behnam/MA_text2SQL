import os
import json
import sqlite3
import glob
from typing import List, Dict, Any, Tuple
import together

# Initialize Together API client
# You'll need to set your API key as an environment variable
api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")

# Initialize the Together client
client = together.Together(api_key=api_key)

def get_spider_databases(spider_dir: str) -> List[str]:
    """Get a list of all SQLite database paths in the Spider dataset."""
    db_paths = glob.glob(os.path.join(spider_dir, "database", "*", "*.sqlite"))
    return db_paths

def get_table_schema_from_spider_db(db_path: str) -> str:
    """Extract the schema information from a Spider dataset database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table information
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        column_descriptions = []
        for col in columns:
            col_id, col_name, col_type, not_null, default_val, pk = col
            column_descriptions.append(f"- {col_name} ({col_type})" + 
                                     (", PRIMARY KEY" if pk else "") +
                                     (", NOT NULL" if not_null else ""))
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fks = cursor.fetchall()
        for fk in fks:
            id, seq, table_ref, from_col, to_col, on_update, on_delete, match = fk
            column_descriptions.append(f"- FOREIGN KEY: {from_col} REFERENCES {table_ref}({to_col})")
        
        schema_info.append(f"Table: {table_name}\n" + "\n".join(column_descriptions))
    
    conn.close()
    return "\n\n".join(schema_info)

def get_sample_data(db_path: str, num_samples: int = 3) -> str:
    """Get sample data for each table to provide context."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table information
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    
    sample_data = []
    for table in tables:
        table_name = table[0]
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {num_samples};")
            rows = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            if rows:
                sample_data.append(f"Table: {table_name} (Sample Data)")
                sample_data.append(f"Columns: {', '.join(columns)}")
                
                for row in rows:
                    row_str = ", ".join([f"{col}: {val}" for col, val in zip(columns, row)])
                    sample_data.append(f"Row: {row_str}")
                
                sample_data.append("")  # Add empty line between tables
        except sqlite3.Error as e:
            sample_data.append(f"Error getting sample data for {table_name}: {str(e)}")
    
    conn.close()
    return "\n".join(sample_data)

def text_to_sql(natural_language_query: str, db_schema: str, sample_data: str = "") -> str:
    """
    Convert natural language query to SQL using Together.ai's LLM API.
    
    Args:
        natural_language_query: The question in natural language
        db_schema: The database schema information
        sample_data: Optional sample data from the database
        
    Returns:
        The generated SQL query
    """
    # Include sample data if provided
    sample_data_section = ""
    if sample_data:
        sample_data_section = f"""
Sample Data (for context only):
{sample_data}
"""
    
    prompt = f"""You are an expert SQL assistant. Your task is to convert the following natural language question into a valid SQL query based on the database schema provided.

Database Schema:
{db_schema}
{sample_data_section}

User Question: {natural_language_query}

Please provide ONLY the SQL query without any explanation. The query should be valid SQLite syntax.
"""
    
    # Call the Together.ai API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1  # Low temperature for more deterministic outputs
    )
    
    sql_query = response.choices[0].message.content.strip()
    
    # Clean up the response in case it contains markdown or explanations
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
    return sql_query

def execute_sql(query: str, db_path: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Execute the SQL query and return the results and success status."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert to list of dictionaries
        result_list = [dict(row) for row in results]
        
        conn.close()
        return result_list, True
    except sqlite3.Error as e:
        return [{"error": str(e)}], False

def evaluate_on_spider_dev(spider_dir: str, num_examples: int = 10) -> None:
    """Evaluate the text-to-SQL model on Spider dev set."""
    # Load Spider dev set
    with open(os.path.join(spider_dir, "dev.json"), "r") as f:
        dev_examples = json.load(f)
    
    # Limit number of examples to evaluate
    dev_examples = dev_examples[:min(num_examples, len(dev_examples))]
    
    results = []
    for example in dev_examples:
        db_name = example["db_id"]
        question = example["question"]
        gold_query = example["query"]
        db_path = os.path.join(spider_dir, "database", db_name, f"{db_name}.sqlite")
        
        # Skip if database doesn't exist
        if not os.path.exists(db_path):
            results.append({
                "db_name": db_name,
                "question": question,
                "gold_query": gold_query,
                "generated_query": "N/A",
                "error": "Database file not found",
                "success": False
            })
            continue
        
        schema = get_table_schema_from_spider_db(db_path)
        sample_data = get_sample_data(db_path)
        
        generated_query = text_to_sql(question, schema, sample_data)
        query_results, success = execute_sql(generated_query, db_path)
        
        results.append({
            "db_name": db_name,
            "question": question,
            "gold_query": gold_query,
            "generated_query": generated_query,
            "execution_results": query_results if len(query_results) <= 5 else f"{len(query_results)} rows returned",
            "execution_success": success
        })
    
    # Save results to a file
    with open("spider_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate simple execution accuracy
    execution_success_count = sum(1 for result in results if result.get("execution_success", False))
    print(f"Execution success rate: {execution_success_count}/{len(results)} = {execution_success_count/len(results)*100:.2f}%")

def explore_spider_database(spider_dir: str) -> None:
    """Interactive mode to explore a Spider database."""
    # List all available databases
    db_paths = get_spider_databases(spider_dir)
    if not db_paths:
        print("No database files found in the specified directory.")
        return
    
    print("Available databases:")
    for i, path in enumerate(db_paths):
        db_name = os.path.basename(os.path.dirname(path))
        print(f"{i+1}. {db_name}")
    
    # Let the user choose a database
    while True:
        try:
            choice = int(input("\nSelect a database number (or 0 to exit): "))
            if choice == 0:
                return
            if 1 <= choice <= len(db_paths):
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    db_path = db_paths[choice-1]
    db_name = os.path.basename(os.path.dirname(db_path))
    print(f"\nSelected database: {db_name}")
    
    # Get and display schema
    schema = get_table_schema_from_spider_db(db_path)
    print("\nDatabase Schema:")
    print(schema)
    
    # Get sample data
    sample_data = get_sample_data(db_path, num_samples=2)
    print("\nSample Data:")
    print(sample_data)
    
    # Interactive mode for queries
    print("\nEnter natural language queries (or 'exit' to quit):")
    while True:
        query = input("\nQuery: ")
        if query.lower() == 'exit':
            break
        
        sql = text_to_sql(query, schema, sample_data)
        print(f"\nGenerated SQL: {sql}")
        
        results, success = execute_sql(sql, db_path)
        print("\nResults:")
        if success:
            if len(results) > 10:
                # Display first 10 results only if there are many
                print(json.dumps(results[:10], indent=2))
                print(f"... and {len(results) - 10} more rows")
            else:
                print(json.dumps(results, indent=2))
        else:
            print("Query execution failed:", results[0]["error"])

def main():
    # Path to Spider dataset directory
    spider_dir = input("Enter the path to Spider dataset directory: ")
    
    if not os.path.exists(spider_dir):
        print(f"Directory {spider_dir} does not exist.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Explore a Spider database")
        print("2. Evaluate on Spider dev set")
        print("3. Exit")
        
        choice = input("\nSelect an option: ")
        
        if choice == "1":
            explore_spider_database(spider_dir)
        elif choice == "2":
            num_examples = int(input("Number of examples to evaluate (default 10): ") or "10")
            evaluate_on_spider_dev(spider_dir, num_examples)
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
import os
import re
import glob
import sqlite3
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
import sqlparse
import openai
import json
import torch
from datetime import datetime

# Set your API key
API_KEY = 'your_api_key_here'
os.environ["TOGETHER_API_KEY"] = API_KEY

# Dataset paths - configure these for your environment
BIRD_DATASET_PATH = 'path/to/bird_subset/stratified_output'
SPIDER_DATASET_PATH = 'path/to/spider_subset/spider_stratified_output_200'

from models import ModelProvider,TogetherAIProvider, OpenAIProvider, LocalHuggingFaceProvider


class Text2SQLPipeline:
    """
    Pipeline for Text2SQL task: loading data, generating SQL queries, and evaluating results.
    Supports both API-based and local models.
    """
    
    def __init__(self, bird_path: str, spider_path: str, model_config: Dict = None):
        """
        Initialize the pipeline with dataset paths and model configuration.
        
        Args:
            bird_path: Path to BIRD dataset
            spider_path: Path to SPIDER dataset
            model_config: Configuration for the model to use, with keys:
                - "type": "together_ai", "openai", or "local"
                - "name": Model name (for API models) or path (for local models)
                - "api_key": API key (for API models)
                - "device": Device to use for local models ("cpu", "cuda", "auto")
                - "max_new_tokens": Maximum tokens to generate (for local models)
        """
        self.bird_path = bird_path
        self.spider_path = spider_path
        
        # Default model config
        default_model_config = {
            "type": "together_ai",
            "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "api_key": API_KEY,
            "device": "auto",
            "max_new_tokens": 512
        }
        
        # Use provided config or default
        self.model_config = model_config or default_model_config

        # Load data
        self.train_data, self.test_data = self.load_data()
        
        # Initialize the model provider based on config
        self._init_model_provider()
    
    def _init_model_provider(self):
        """Initialize the model provider based on the configuration"""
        model_type = self.model_config.get("type", "together_ai").lower()
        model_name = self.model_config.get("name")
        api_key = self.model_config.get("api_key")
        
        if model_type == "together_ai":
            self.model_provider = TogetherAIProvider(model_name, api_key)
        elif model_type == "openai":
            self.model_provider = OpenAIProvider(model_name, api_key)
        elif model_type == "local":
            device = self.model_config.get("device", "auto")
            max_new_tokens = self.model_config.get("max_new_tokens", 512)
            self.model_provider = LocalHuggingFaceProvider(model_name, device, max_new_tokens)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Store model info for later use
        self.model_info = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_sql_semantic_equivalence(self, predicted_sql: str, ground_truth_sql: str, 
                                  question: str) -> Tuple[bool, str]:
        """
        Use the configured model to determine if two SQL queries are semantically equivalent,
        even if they have syntactic differences.
        
        Args:
            predicted_sql: Predicted SQL query
            ground_truth_sql: Ground truth SQL query
            question: The original natural language question
            
        Returns:
            Tuple of (is_equivalent, explanation)
        """
        # Create system message for the judge
        system_message = (
            "You are a SQL expert tasked with determining if two SQL queries are semantically equivalent. "
            "This means they may have syntactic differences but would return the same results when executed "
            "on the same database. Common acceptable differences include: "
            "- Different column ordering in SELECT statements "
            "- Presence or absence of column aliases (AS) "
            "- Different formatting, spacing, or capitalization "
            "- Use of quotes around identifiers "
            "- Simple reordering of conditions that doesn't change the logic "
            "\n\nYour response must be in JSON format with two fields: "
            "'equivalent' (true/false) and 'explanation' (a brief explanation of your judgment)."
        )
        
        # Create user message
        user_message = (
            f"Question: {question}\n\n"
            f"Gold SQL Query: {ground_truth_sql}\n\n"
            f"Generated SQL Query: {predicted_sql}\n\n"
            "Are these two SQL queries semantically equivalent? Provide your judgment."
        )
        
        try:
            # Generate response using the configured model provider
            raw_response = self.model_provider.generate(system_message, user_message)
            
            # Try to extract JSON
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                try:
                    json_obj = json.loads(json_match.group(1))
                    if "equivalent" in json_obj:
                        return json_obj["equivalent"], json_obj.get("explanation", "")
                except json.JSONDecodeError:
                    pass
            
            # If JSON extraction fails, look for yes/no in the response
            if re.search(r'\b(yes|equivalent|same|equal)\b', raw_response, re.IGNORECASE):
                return True, "Model indicated equivalence but didn't provide structured output"
            elif re.search(r'\b(no|not equivalent|different|not the same)\b', raw_response, re.IGNORECASE):
                return False, "Model indicated non-equivalence but didn't provide structured output"
            
            # Default to relying on execution results
            return False, "Could not determine semantic equivalence from model response"
        
        except Exception as e:
            # If the model call fails, default to relying on execution results
            return False, f"Error in semantic check: {str(e)}"
    
    def load_data(self) -> Tuple[List[Tuple[Dict, str]], List[Tuple[Dict, str]]]:
        """
        Load data from BIRD and SPIDER datasets and split into train/test sets.
        Returns data with file paths.
        
        Returns:
            Tuple of (train_data, test_data) where each item is (json_data, file_path)
        """
        # Load JSON files from directories
        bird_data = self._load_json_files(self.bird_path)
        spider_data = self._load_json_files(self.spider_path)
        
        # Combine datasets
        all_data = bird_data + spider_data
        
        # Split into train/test sets (only use the data part for splitting, but keep the paths)
        data_only = [item[0] for item in all_data]
        train_indices, test_indices = train_test_split(
            range(len(data_only)), test_size=0.2, random_state=42
        )
        
        train_data = [all_data[i] for i in train_indices]
        test_data = [all_data[i] for i in test_indices]
        
        print(f"Total data points: {len(all_data)}")
        print(f"Bird data points: {len(bird_data)}")
        print(f"Spider data points: {len(spider_data)}")
        print(f"Training data points: {len(train_data)}")
        print(f"Testing data points: {len(test_data)}")
        
        return train_data, test_data
    
    def _load_json_files(self, dir_path: str) -> List[Tuple[Dict, str]]:
        """
        Load all JSON files from a directory and track their file paths.
        
        Args:
            dir_path: Directory path containing JSON files
            
        Returns:
            List of tuples (loaded_json_object, file_path)
        """
        data = []
        for filepath in glob.glob(os.path.join(dir_path, 'instance_*.json')):
            with open(filepath, 'r') as file:
                json_data = json.load(file)
                # Store the file path with the data
                data.append((json_data, filepath))
        return data
    
    def prepare_for_inference(self, instance: Dict, root_dir: str) -> Dict:
        """
        Prepare an instance for inference by loading the schema.
        
        Args:
            instance: Data instance
            root_dir: Root directory for the dataset
            
        Returns:
            Prepared instance with schema loaded
        """
        schemas_path = instance['schemas']['path']
        schemas_df = []
        
        for schema_path in schemas_path:
            s_p = root_dir + schema_path.split(schema_path.split('/')[0])[1]    
            schemas_df.append(pd.read_csv(s_p))
            
        schema_df = pd.concat(schemas_df, ignore_index=True)
        
        return {
            'id': instance['id'],
            'evidence': instance['evidence'],
            'dataset': instance['dataset'],
            'question': instance['question'],
            'sql': instance['sql'],
            'schema': schema_df.to_dict(orient='records'),
            'database': instance['database'],
        }
    
    def create_sql_prompt(self, instance: Dict, root_dir: str) -> Tuple[str, str]:
        """
        Create a prompt for SQL generation.
        
        Args:
            instance: Data instance
            root_dir: Root directory for the dataset
            
        Returns:
            Tuple of (system_message, user_message)
        """
        # Prepare instance for inference
        inference_data = self.prepare_for_inference(instance, root_dir)
        
        # Extract question and schema
        question = inference_data['question']
        schema = inference_data['schema']
        
        # Format the system message
        system_message = (
            "You are a database expert. "
            "You are supposed to provide a SQL query based on the user's question and the provided database schema. "
            "Your response must be in JSON format with a field named 'sql' containing the generated SQL query. "
            "Example response format: {\"sql\": \"SELECT * FROM table WHERE condition\"}"
        )
        
        # Format the user message
        user_message = f"{question}\n\nHere is the database schema:\n```\n{schema}\n```"
        
        return system_message, user_message
    
    def extract_sql_query_from_text(self, text: str) -> str:
        """
        Extract SQL query from text, handling various formats (JSON, code blocks, etc.)
        
        Args:
            text: Raw text output that may contain SQL query
            
        Returns:
            Extracted SQL query as a string, or empty string if extraction fails
        """
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
    
    def generate_sql_query(self, instance: Dict, dataset_type: str) -> Tuple[str, str]:
        """
        Generate a SQL query for a given instance using the configured model.
        
        Args:
            instance: Data instance
            dataset_type: Type of dataset ('bird' or 'spider')
            
        Returns:
            Tuple of (raw_response, generated_sql)
        """
        # Determine root directory based on dataset type
        root_dir = self.bird_path if dataset_type == 'bird' else self.spider_path
        
        # Get the prompt messages
        system_message, user_message = self.create_sql_prompt(instance, root_dir)
        
        try:
            # Generate response using the configured model provider
            raw_response = self.model_provider.generate(system_message, user_message)
            
            # Extract SQL query from the raw response using the enhanced function
            generated_sql = self.extract_sql_query_from_text(raw_response)
            
            if generated_sql:
                return raw_response, generated_sql
            else:
                # If extraction fails, return the raw response
                print("Failed to extract SQL from response. Returning raw response.")
                return raw_response, ""
        except Exception as e:
            # Handle errors
            error_message = f"Model error: {str(e)}"
            print(error_message)
            return error_message, ""
    
    def get_sqlite_db_connection(self, database_path: str, root_dir: str) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Args:
            database_path: Path to the SQLite database file
            root_dir: Root directory for the dataset
            
        Returns:
            SQLite connection object
        """
        database_path = root_dir + database_path.split(database_path.split('/')[0])[1]
        
        # Connect to the SQLite database
        conn = sqlite3.connect(database_path)
        
        return conn
    
    def check_execution_accuracy(self, predicted_sql: str, ground_truth_sql: str, 
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
    
    def normalize_sql(self, sql: str) -> str:
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
    
    def check_exact_match(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """
        Check if predicted SQL exactly matches ground truth after normalization.
        
        Args:
            predicted_sql: Predicted SQL query
            ground_truth_sql: Ground truth SQL query
            
        Returns:
            True if exact match, False otherwise
        """
        # Normalize both queries
        normalized_pred = self.normalize_sql(predicted_sql)
        normalized_gt = self.normalize_sql(ground_truth_sql)
        
        # Compare normalized queries
        return normalized_pred == normalized_gt
    
    def evaluate_instance(self, instance: Dict, generated_sql: str, dataset_type: str) -> Dict:
        """
        Evaluate the generated SQL query against the ground truth.
        
        Args:
            instance: Data instance
            generated_sql: Generated SQL query
            dataset_type: Type of dataset ('bird' or 'spider')
            
        Returns:
            Evaluation results as a dictionary with the full instance data and prediction information
        """
        # Determine root directory based on dataset type
        root_dir = self.bird_path if dataset_type == 'bird' else self.spider_path
        
        # Get database connection
        db_connection = self.get_sqlite_db_connection(instance['database']['path'][0], root_dir)
        
        # Check execution accuracy
        exec_correct, exec_error = self.check_execution_accuracy(
            generated_sql, instance['sql'], db_connection
        )
        
        # Check exact match
        exact_match = self.check_exact_match(generated_sql, instance['sql'])
        
        # If not exact match but execution is correct, or execution failed,
        # check semantic equivalence using the model
        semantic_equivalent = None
        semantic_explanation = None
        
        # Determine semantic equivalence based on exact match, execution correctness, and errors
        if exact_match:
            # If exact match, queries are semantically equivalent
            semantic_equivalent = True
            semantic_explanation = "Exact match found"
        elif exec_correct:
            # If execution is correct but not exact match, consider it semantically equivalent
            semantic_equivalent = True
            semantic_explanation = "Execution correct but not exact match"
        elif exec_error and exec_error.strip():
            # If there's a non-empty execution error, queries are not semantically equivalent
            semantic_equivalent = False
            semantic_explanation = f"Execution failed: {exec_error}"
        else:
            # Otherwise, use the model to check semantic equivalence
            # This catches cases with no exact match, incorrect execution, but no specific error
            semantic_equivalent, semantic_explanation = self.check_sql_semantic_equivalence(
            generated_sql, instance['sql'], instance['question']
            )
        
        # Close database connection
        db_connection.close()
        
        return {
            'instance': instance,
            'has_prediction': True,
            'predicted_output': {
                'generated_sql': generated_sql,
                'execution_correct': exec_correct,
                'execution_error': exec_error,
                'exact_match': exact_match,
                'semantic_equivalent': semantic_equivalent,
                'semantic_explanation': semantic_explanation
            }
        }
    
    def run_pipeline(self, num_samples: int = 10, save_updated_files: bool = True, 
                    output_dir: str = None) -> Dict:
        """
        Run the complete pipeline: load data, generate SQL, evaluate, and update JSON files.
        
        Args:
            num_samples: Number of samples to evaluate
            save_updated_files: Whether to save updated JSON files
            output_dir: Directory to save updated files (if None, will update files in place)
            
        Returns:
            Evaluation results with comprehensive information
        """
        # Use a subset for evaluation
        if num_samples:
            train_data = self.train_data[:num_samples]
            
        eval_data = train_data
        
        results = []
        
        # Process each instance
        for instance_data, file_path in eval_data:
            print(f"Processing instance {instance_data['id']}...")
            
            # Determine dataset type
            dataset_type = instance_data['dataset']
            
            # Generate SQL query
            raw_response, generated_sql = self.generate_sql_query(instance_data, dataset_type)
            
            if generated_sql:
                # Evaluate the generated SQL
                evaluation = self.evaluate_instance(instance_data, generated_sql, dataset_type)
                # Add model information
                evaluation['model'] = self.model_info
                results.append(evaluation)
                
                # Update the original instance data with inference results
                instance_data['inference_results'] = {
                    'has_prediction': True,
                    'model': self.model_info,
                    'predicted_output': {
                        'generated_sql': generated_sql,
                        'execution_correct': evaluation['predicted_output']['execution_correct'],
                        'execution_error': evaluation['predicted_output']['execution_error'],
                        'exact_match': evaluation['predicted_output']['exact_match'],
                        'semantic_equivalent': evaluation['predicted_output'].get('semantic_equivalent', None),
                        'semantic_explanation': evaluation['predicted_output'].get('semantic_explanation', '')
                    }
                }
                
                print(f"Execution correct: {evaluation['predicted_output']['execution_correct']}")
                print(f"Exact match: {evaluation['predicted_output']['exact_match']}")
                print(f"Semantic equivalent: {evaluation['predicted_output'].get('semantic_equivalent', False)}")
            else:
                # Failed to extract SQL
                failed_result = {
                    'instance': instance_data,
                    'has_prediction': False,
                    'predicted_output': {
                        'sql': None,
                        'raw_response': raw_response
                    },
                    'model': self.model_info
                }
                results.append(failed_result)
                
                # Update the original instance data with failure information
                instance_data['inference_results'] = {
                    'has_prediction': False,
                    'model': self.model_info,
                    'predicted_output': {
                        'sql': None,
                        'raw_response': raw_response
                    }
                }
                
                print("Failed to extract SQL from model response")
                
            # Save the updated instance data to file
            if save_updated_files:
                self._save_updated_instance(instance_data, file_path, output_dir)
                
            print("-" * 50)
        
        # Calculate overall metrics
        num_eval = len(results)
        num_with_prediction = sum(1 for r in results if r.get('has_prediction', False))
        
        # Only consider instances with valid predictions for accuracy metrics
        exec_correct = sum(1 for r in results if r.get('has_prediction', False) and 
                           r['predicted_output'].get('execution_correct', False))
        exact_match = sum(1 for r in results if r.get('has_prediction', False) and 
                          r['predicted_output'].get('exact_match', False))
        semantic_equivalent = sum(1 for r in results if r.get('has_prediction', False) and 
                                 r['predicted_output'].get('semantic_equivalent', False))
        
        metrics = {
            'num_evaluated': num_eval,
            'num_with_prediction': num_with_prediction,
            'prediction_rate': num_with_prediction / num_eval if num_eval > 0 else 0,
            'execution_accuracy': exec_correct / num_with_prediction if num_with_prediction > 0 else 0,
            'exact_match_accuracy': exact_match / num_with_prediction if num_with_prediction > 0 else 0,
            'semantic_equivalent_accuracy': semantic_equivalent / num_with_prediction if num_with_prediction > 0 else 0,
            'model': self.model_info
        }
        
        print(f"Prediction rate: {metrics['prediction_rate']:.2f}")
        print(f"Execution accuracy: {metrics['execution_accuracy']:.2f}")
        print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.2f}")
        print(f"Semantic equivalence accuracy: {metrics['semantic_equivalent_accuracy']:.2f}")
        
        return metrics
        
    def _save_updated_instance(self, instance_data: Dict, original_file_path: str, output_dir: str = None):
        """
        Save the updated instance data back to a JSON file.
        
        Args:
            instance_data: The updated instance data
            original_file_path: Path to the original JSON file
            output_dir: Directory to save the updated file (if None, will update file in place)
        """
        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine the new file path
            file_name = os.path.basename(original_file_path)
            new_file_path = os.path.join(output_dir, file_name)
        else:
            # Update the file in place
            new_file_path = original_file_path
        
        # Save the updated instance data to the file
        with open(new_file_path, 'w') as f:
            json.dump(instance_data, f, indent=2)
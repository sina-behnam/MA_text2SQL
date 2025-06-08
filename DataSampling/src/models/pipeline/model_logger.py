import os
import re
import glob
import sqlite3
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import sqlparse
import json
import torch
from datetime import datetime
import logging
from tqdm import tqdm

from models import ModelProvider, TogetherAIProvider, OpenAIProvider, LocalHuggingFaceProvider, AnthropicProvider


class SchemaUnderstandingLogger:
    """Dedicated logger for schema understanding responses"""
    
    def __init__(self, log_dir: str = "schema_understanding_logs"):
        """
        Initialize schema understanding logger
        
        Args:
            log_dir: Directory to store schema understanding logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logger
        self.logger = logging.getLogger('schema_understanding')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent propagation to root logger
        
        # Create file handler for schema understanding logs
        log_filename = f"schema_understanding_{self.session_timestamp}.log"
        self.log_file_path = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger (remove existing handlers first)
        if self.logger.handlers:
            self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        
        # Also create a JSON log file for structured data
        self.json_log_file = os.path.join(log_dir, f"schema_understanding_{self.session_timestamp}.json")
        self.schema_responses = []
        
        self.logger.info("="*80)
        self.logger.info("SCHEMA UNDERSTANDING SESSION STARTED")
        self.logger.info("="*80)
    
    def log_schema_introduction(self, schema_key: str, schema_info: Dict, 
                              introduction_prompt: str, model_response: str, 
                              model_info: Dict, dataset_type: str):
        """
        Log schema introduction interaction
        
        Args:
            schema_key: Unique identifier for the schema
            schema_info: Database schema information
            introduction_prompt: The prompt sent to the model
            model_response: Model's response to schema introduction
            model_info: Model configuration information
            dataset_type: Type of dataset (bird/spider)
        """
        # Log to text file
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SCHEMA: {schema_key}")
        self.logger.info(f"DATASET: {dataset_type.upper()}")
        self.logger.info(f"MODEL: {model_info.get('model_name', 'Unknown')}")
        self.logger.info(f"{'='*60}")
        
        self.logger.info("SCHEMA INTRODUCTION PROMPT:")
        self.logger.info("-" * 40)
        self.logger.info(introduction_prompt)
        
        self.logger.info("\nMODEL RESPONSE:")
        self.logger.info("-" * 40)
        self.logger.info(model_response)
        self.logger.info("="*60)
        
        # Prepare structured data for JSON log
        schema_entry = {
            "timestamp": datetime.now().isoformat(),
            "schema_key": schema_key,
            "dataset_type": dataset_type,
            "database_name": schema_info.get('database', {}).get('name', 'Unknown'),
            "model_info": model_info,
            "schema_tables": [schema.get('table_name', '') for schema in schema_info.get('schemas', [])],
            "table_count": len(schema_info.get('schemas', [])),
            "introduction_prompt": introduction_prompt,
            "model_response": model_response,
            "response_length": len(model_response),
            "prompt_length": len(introduction_prompt)
        }
        
        self.schema_responses.append(schema_entry)
        
        # Save to JSON file after each entry
        self._save_json_log()
    
    def log_conversation_summary(self, schema_key: str, total_questions: int, 
                                successful_predictions: int, failed_predictions: int,
                                avg_execution_accuracy: float):
        """
        Log summary statistics for a schema conversation
        
        Args:
            schema_key: Schema identifier
            total_questions: Total questions processed
            successful_predictions: Number of successful SQL generations
            failed_predictions: Number of failed SQL generations
            avg_execution_accuracy: Average execution accuracy for this schema
        """
        self.logger.info(f"\n{'*'*60}")
        self.logger.info(f"CONVERSATION SUMMARY - {schema_key}")
        self.logger.info(f"{'*'*60}")
        self.logger.info(f"Total questions processed: {total_questions}")
        self.logger.info(f"Successful SQL generations: {successful_predictions}")
        self.logger.info(f"Failed SQL generations: {failed_predictions}")
        self.logger.info(f"Success rate: {successful_predictions/total_questions:.3f}")
        self.logger.info(f"Average execution accuracy: {avg_execution_accuracy:.3f}")
        self.logger.info(f"{'*'*60}")
        
        # Update the last schema entry with summary stats
        if self.schema_responses:
            self.schema_responses[-1].update({
                "conversation_summary": {
                    "total_questions": total_questions,
                    "successful_predictions": successful_predictions,
                    "failed_predictions": failed_predictions,
                    "success_rate": successful_predictions/total_questions if total_questions > 0 else 0,
                    "avg_execution_accuracy": avg_execution_accuracy
                }
            })
            self._save_json_log()
    
    def _save_json_log(self):
        """Save structured logs to JSON file"""
        with open(self.json_log_file, 'w') as f:
            json.dump(self.schema_responses, f, indent=2)
    
    def get_log_file_path(self) -> str:
        """Get the path to the current log file"""
        return self.log_file_path
    
    def get_json_log_file_path(self) -> str:
        """Get the path to the current JSON log file"""
        return self.json_log_file
    
    def close(self):
        """Close the logger and finalize logs"""
        self.logger.info("="*80)
        self.logger.info("SCHEMA UNDERSTANDING SESSION ENDED")
        self.logger.info(f"Total schemas processed: {len(self.schema_responses)}")
        self.logger.info("="*80)
        
        # Final save
        self._save_json_log()
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def analyze_schema_understanding_logs(log_dir: str = "schema_logs"):
    """
    Analyze schema understanding logs to gain insights into model performance
    
    Args:
        log_dir: Directory containing schema understanding logs
    """
    print(f"\n{'='*60}")
    print("SCHEMA UNDERSTANDING ANALYSIS")
    print(f"{'='*60}")
    
    # Find all JSON log files
    json_files = glob.glob(os.path.join(log_dir, "**/schema_understanding_*.json"), recursive=True)
    
    if not json_files:
        print("No schema understanding log files found.")
        return
    
    all_schema_data = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            schema_data = json.load(f)
            all_schema_data.extend(schema_data)
    
    if not all_schema_data:
        print("No schema understanding data found.")
        return
    
    print(f"Found {len(all_schema_data)} schema understanding interactions")
    
    # Analyze by model
    model_stats = defaultdict(list)
    for entry in all_schema_data:
        model_name = entry.get('model_info', {}).get('model_name', 'Unknown')
        model_stats[model_name].append(entry)
    
    print(f"\nSchema understanding by model:")
    for model, entries in model_stats.items():
        print(f"\n  Model: {model}")
        print(f"    Total schemas processed: {len(entries)}")
        
        # Analyze response lengths
        response_lengths = [entry['response_length'] for entry in entries]
        avg_response_length = sum(response_lengths) / len(response_lengths)
        print(f"    Average schema understanding response length: {avg_response_length:.0f} chars")
        
        # Analyze conversation summaries
        summaries = [entry.get('conversation_summary', {}) for entry in entries if 'conversation_summary' in entry]
        if summaries:
            avg_success_rate = sum(s.get('success_rate', 0) for s in summaries) / len(summaries)
            avg_execution_acc = sum(s.get('avg_execution_accuracy', 0) for s in summaries) / len(summaries)
            total_questions = sum(s.get('total_questions', 0) for s in summaries)
            
            print(f"    Total questions across all schemas: {total_questions}")
            print(f"    Average SQL generation success rate: {avg_success_rate:.3f}")
            print(f"    Average execution accuracy: {avg_execution_acc:.3f}")
    
    # Analyze by dataset type
    dataset_stats = defaultdict(list)
    for entry in all_schema_data:
        dataset_type = entry.get('dataset_type', 'Unknown')
        dataset_stats[dataset_type].append(entry)
    
    print(f"\nSchema understanding by dataset:")
    for dataset, entries in dataset_stats.items():
        print(f"\n  Dataset: {dataset.upper()}")
        print(f"    Schemas processed: {len(entries)}")
        
        # Average table count per schema
        table_counts = [entry['table_count'] for entry in entries]
        avg_tables = sum(table_counts) / len(table_counts)
        print(f"    Average tables per schema: {avg_tables:.1f}")
    
    # Find most complex schemas (by table count)
    complex_schemas = sorted(all_schema_data, key=lambda x: x['table_count'], reverse=True)[:5]
    print(f"\nMost complex schemas (by table count):")
    for i, schema in enumerate(complex_schemas, 1):
        print(f"  {i}. {schema['schema_key']} - {schema['table_count']} tables")
        if 'conversation_summary' in schema:
            summary = schema['conversation_summary']
            print(f"     Success rate: {summary.get('success_rate', 0):.3f}, "
                  f"Execution accuracy: {summary.get('avg_execution_accuracy', 0):.3f}")

def compare_with_baseline():
    """
    Function to compare optimized results with baseline results
    You can implement this to load both optimized and baseline results
    and create a comparison analysis
    """
    # Load optimized results
    optimized_file = os.path.join(OUTPUT_BASE_DIR, 'optimized_evaluation_summary.json')
    
    # Load baseline results (from your previous runs)
    baseline_file = 'entriched_full_dataset_best_1/entriched_full_dataset_best_1_summary.json'
    
    if os.path.exists(optimized_file) and os.path.exists(baseline_file):
        with open(optimized_file, 'r') as f:
            optimized_results = json.load(f)
        
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPARISON")
        print(f"{'='*60}")
        
        for model_name in optimized_results:
            if model_name in baseline_results:
                opt_acc = optimized_results[model_name].get('execution_accuracy', 0)
                base_acc = baseline_results[model_name].get('execution_accuracy', 0)
                improvement = ((opt_acc - base_acc) / base_acc * 100) if base_acc > 0 else 0
                
                print(f"\nModel: {model_name}")
                print(f"  Baseline accuracy: {base_acc:.3f}")
                print(f"  Optimized accuracy: {opt_acc:.3f}")
                print(f"  Improvement: {improvement:+.1f}%")
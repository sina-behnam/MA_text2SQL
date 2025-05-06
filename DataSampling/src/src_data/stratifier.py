import json
import os
import shutil
from collections import defaultdict
import random
import math
from tqdm import tqdm

def stratify_text2sql_data(data_list, target_size, output_dir="stratified_output"):
    """
    Stratify Text2SQL data to a specific target size while:
    1. Maintaining the original difficulty distribution
    2. Ensuring uniform database representation within each difficulty level
    
    Args:
        data_list (list): List of Text2SQL instance dictionaries
        target_size (int): Target number of instances for the stratified dataset
        output_dir (str): Directory to save stratified data and files
    
    Returns:
        int: Number of instances in the stratified dataset
    """
    print("Analyzing dataset distribution...")
    
    # Step 1: Calculate original difficulty distribution
    difficulty_counts = defaultdict(int)
    total_instances = len(data_list)
    
    for instance in data_list:
        difficulty_counts[instance['difficulty']] += 1
    
    difficulty_ratios = {
        difficulty: count / total_instances 
        for difficulty, count in difficulty_counts.items()
    }
    
    # Step 2: Group by difficulty and database
    print("Grouping instances by difficulty and database...")
    grouped = defaultdict(lambda: defaultdict(list))
    
    for instance in tqdm(data_list, desc="Grouping instances"):
        difficulty = instance['difficulty']
        db_name = instance['database']['name']
        grouped[difficulty][db_name].append(instance)
    
    # Step 3: Calculate target instances for each difficulty level based on original ratios
    target_difficulty_counts = {
        difficulty: round(ratio * target_size)
        for difficulty, ratio in difficulty_ratios.items()
    }
    
    # Adjust for rounding errors to ensure the sum matches target_size
    total_after_rounding = sum(target_difficulty_counts.values())
    if total_after_rounding != target_size:
        # Add or subtract the difference from the largest difficulty
        largest_difficulty = max(target_difficulty_counts.keys(), key=lambda d: target_difficulty_counts[d])
        target_difficulty_counts[largest_difficulty] += (target_size - total_after_rounding)
    
    # Step 4: Calculate the database count for each difficulty level
    db_counts_per_difficulty = {
        difficulty: len(db_dict)
        for difficulty, db_dict in grouped.items()
    }
    
    # Step 5: Calculate target instances per database for each difficulty level
    target_per_db = {}
    feasible_target_size = target_size  # May be adjusted downward if constraints cannot be met
    
    print("Calculating optimal distribution...")
    for difficulty, count in target_difficulty_counts.items():
        db_count = db_counts_per_difficulty[difficulty]
        
        # Target instances per database (before rounding)
        target_per_db[difficulty] = count / db_count
        
        # Check if we have enough instances in each database
        min_available = min(
            len(instances) for instances in grouped[difficulty].values()
        ) if grouped[difficulty] else 0
        
        # If target is higher than available, adjust target downward
        if target_per_db[difficulty] > min_available:
            # Calculate new feasible target based on this constraint
            new_feasible = min_available * db_count / difficulty_ratios[difficulty]
            feasible_target_size = min(feasible_target_size, int(new_feasible))
    
    # If we had to adjust target size, recalculate all targets
    if feasible_target_size < target_size:
        print(f"Warning: Target size {target_size} is not feasible with given constraints.")
        print(f"Adjusted to maximum feasible size: {feasible_target_size}")
        
        # Recalculate targets based on feasible size
        target_difficulty_counts = {
            difficulty: round(ratio * feasible_target_size)
            for difficulty, ratio in difficulty_ratios.items()
        }
        
        # Adjust for rounding errors
        total_after_rounding = sum(target_difficulty_counts.values())
        if total_after_rounding != feasible_target_size:
            largest_difficulty = max(target_difficulty_counts.keys(), key=lambda d: target_difficulty_counts[d])
            target_difficulty_counts[largest_difficulty] += (feasible_target_size - total_after_rounding)
    
    # Step 6: Calculate final instances per database for each difficulty
    instances_per_db = {}
    for difficulty, count in target_difficulty_counts.items():
        db_count = db_counts_per_difficulty[difficulty]
        
        # Base count per database (may have fraction)
        base_per_db = count / db_count
        
        # Integer part and remainder
        int_per_db = int(base_per_db)
        remainder = count - (int_per_db * db_count)
        
        # Assign integer counts to each database
        instances_per_db[difficulty] = {
            db_name: int_per_db for db_name in grouped[difficulty].keys()
        }
        
        # Distribute remainder by adding 1 to some databases
        if remainder > 0:
            # Randomly select databases to receive one extra instance
            extra_dbs = random.sample(list(grouped[difficulty].keys()), remainder)
            for db_name in extra_dbs:
                instances_per_db[difficulty][db_name] += 1
    
    # Step 7: Create the stratified sample
    print("Selecting instances for stratified sample...")
    stratified_sample = []
    
    for difficulty, db_dict in grouped.items():
        for db_name, instances in db_dict.items():
            if difficulty in instances_per_db and db_name in instances_per_db[difficulty]:
                target_count = instances_per_db[difficulty][db_name]
                
                # Randomly sample target_count instances from this database
                sampled = random.sample(
                    instances, 
                    min(target_count, len(instances))
                )
                
                stratified_sample.extend(sampled)
    
    # Step 8: Create directory structure and save files
    print(f"Creating directory structure and saving {len(stratified_sample)} instances...")
    os.makedirs(output_dir, exist_ok=True)
    db_dir = os.path.join(output_dir, "databases")
    schema_dir = os.path.join(output_dir, "schemas")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(schema_dir, exist_ok=True)
    
    # Process each instance in the stratified sample
    for instance in tqdm(stratified_sample, desc="Processing instances"):
        # Create copy to modify
        instance_copy = instance.copy()
        db_name = instance['database']['name']
        
        # Create database-specific directories
        instance_db_dir = os.path.join(db_dir, db_name)
        instance_schema_dir = os.path.join(schema_dir, db_name)
        os.makedirs(instance_db_dir, exist_ok=True)
        os.makedirs(instance_schema_dir, exist_ok=True)
        
        # Update and copy database files
        new_db_paths = []
        for path in instance['database']['path']:
            filename = os.path.basename(path)
            new_path = os.path.join(instance_db_dir, filename)
            new_db_paths.append(new_path)
            # Copy database file if it exists
            if os.path.exists(path):
                shutil.copy(path, new_path)
        
        instance_copy['database']['path'] = new_db_paths
        
        # Update and copy CSV files if present
        if 'csv_files' in instance['database']:
            new_csv_paths = []
            for path in instance['database']['csv_files']:
                filename = os.path.basename(path)
                new_path = os.path.join(instance_db_dir, filename)
                new_csv_paths.append(new_path)
                # Copy CSV file if it exists
                if os.path.exists(path):
                    shutil.copy(path, new_path)
            
            instance_copy['database']['csv_files'] = new_csv_paths
        
        # Update and copy schema files
        new_schema_paths = []
        for path in instance['schemas']['path']:
            filename = os.path.basename(path)
            new_path = os.path.join(instance_schema_dir, filename)
            new_schema_paths.append(new_path)
            # Copy schema file if it exists
            if os.path.exists(path):
                shutil.copy(path, new_path)
        
        instance_copy['schemas']['path'] = new_schema_paths
        
        # Save instance as JSON file
        output_file = os.path.join(output_dir, f"instance_{instance['id']}.json")
        with open(output_file, 'w') as f:
            json.dump(instance_copy, f, indent=4)
    
    # Print a simple summary
    print("\nStratification Complete!")
    print(f"Original dataset size: {total_instances} instances")
    print(f"Stratified dataset size: {len(stratified_sample)} instances")
    print(f"Difficulty distribution in stratified dataset:")
    
    difficulty_in_stratified = defaultdict(int)
    for instance in stratified_sample:
        difficulty_in_stratified[instance['difficulty']] += 1
        
    for difficulty, count in difficulty_in_stratified.items():
        percentage = (count / len(stratified_sample)) * 100
        original_percentage = (difficulty_counts[difficulty] / total_instances) * 100
        print(f"  - {difficulty}: {count} instances ({percentage:.1f}%, original: {original_percentage:.1f}%)")
    
    return len(stratified_sample)

def analyze_and_select_instances(data_list, target_size):
    """
    Analyze dataset and select instances to create a stratified subset with uniform database distribution.
    
    Args:
        data_list (list): List of Text2SQL instance dictionaries
        target_size (int): Target number of instances for the stratified dataset
    
    Returns:
        tuple: (selected_instances, stats_dict) - Selected instances and statistics
    """
    print("Analyzing database distribution...")
    
    # Step 1: Group instances by database
    db_instances = defaultdict(list)
    
    for instance in tqdm(data_list, desc="Grouping by database"):
        db_name = instance['database']['name']
        db_instances[db_name].append(instance)
    
    # Count databases and instances per database
    num_databases = len(db_instances)
    db_counts = {db_name: len(instances) for db_name, instances in db_instances.items()}
    
    print(f"Found {num_databases} databases with following counts:")
    for db_name, count in sorted(db_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {db_name}: {count} instances")
    
    # Step 2: Calculate target instances per database
    instances_per_db = target_size // num_databases
    remainder = target_size % num_databases
    
    # Check if we have enough instances
    min_count = min(db_counts.values())
    if instances_per_db > min_count:
        print(f"Warning: Target size {target_size} requires {instances_per_db} instances per database")
        print(f"But database '{min(db_counts, key=db_counts.get)}' only has {min_count} instances")
        print(f"Adjusting target to {min_count * num_databases} instances ({min_count} per database)")
        instances_per_db = min_count
        remainder = 0
    
    # Create allocation dictionary
    allocation = {db_name: instances_per_db for db_name in db_instances.keys()}
    
    # Distribute remainder if any
    if remainder > 0:
        # Prioritize databases with more instances to minimize information loss
        sorted_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            db_name = sorted_dbs[i % len(sorted_dbs)][0]
            if allocation[db_name] < db_counts[db_name]:  # Safety check
                allocation[db_name] += 1
    
    actual_target = sum(allocation.values())
    print(f"Final allocation: {actual_target} total instances")
    for db_name, count in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {db_name}: {count} instances")
    
    # Step 3: Sample instances from each database
    print("Sampling instances...")
    stratified_sample = []
    
    for db_name, target_count in allocation.items():
        instances = db_instances[db_name]
        sampled = random.sample(instances, target_count)
        stratified_sample.extend(sampled)
    
    # Calculate statistics
    stats = {
        "original_size": len(data_list),
        "stratified_size": len(stratified_sample),
        "target_size": target_size,
        "actual_size": actual_target,
        "database_counts": db_counts,
        "database_allocation": allocation
    }
    
    # Calculate difficulty distribution statistics
    difficulty_original = defaultdict(int)
    difficulty_stratified = defaultdict(int)
    
    for instance in data_list:
        difficulty_original[instance['difficulty']] += 1
    
    for instance in stratified_sample:
        difficulty_stratified[instance['difficulty']] += 1
    
    stats["difficulty_original"] = dict(difficulty_original)
    stats["difficulty_stratified"] = dict(difficulty_stratified)
    
    # Calculate database type distribution
    db_types_original = defaultdict(int)
    db_types_stratified = defaultdict(int)
    
    for instance in data_list:
        db_type = instance['database'].get('type', 'unknown').lower()
        db_types_original[db_type] += 1
    
    for instance in stratified_sample:
        db_type = instance['database'].get('type', 'unknown').lower()
        db_types_stratified[db_type] += 1
    
    stats["db_types_original"] = dict(db_types_original)
    stats["db_types_stratified"] = dict(db_types_stratified)
    
    return stratified_sample, stats

def analyze_and_select_instances_proportional(data_list, target_size, min_instances_per_db=1):
    """
    Analyze dataset and select instances while preserving the original database distribution.
    
    Args:
        data_list (list): List of Text2SQL instance dictionaries
        target_size (int): Target number of instances for the stratified dataset
        min_instances_per_db (int): Minimum instances to include from each database
    
    Returns:
        tuple: (selected_instances, stats_dict) - Selected instances and statistics
    """
    print("Analyzing database distribution...")
    
    # Step 1: Group instances by database
    db_instances = defaultdict(list)
    
    for instance in tqdm(data_list, desc="Grouping by database"):
        db_name = instance['database']['name']
        db_instances[db_name].append(instance)
    
    # Count databases and instances per database
    num_databases = len(db_instances)
    db_counts = {db_name: len(instances) for db_name, instances in db_instances.items()}
    total_instances = len(data_list)
    
    print(f"Found {num_databases} databases with following distribution:")
    for db_name, count in sorted(db_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_instances) * 100
        print(f"  - {db_name}: {count} instances ({percentage:.1f}%)")
    
    # Step 2: Calculate proportional allocation
    db_percentages = {db_name: count / total_instances for db_name, count in db_counts.items()}
    
    # Initial allocation based on percentages
    initial_allocation = {
        db_name: max(min_instances_per_db, round(percentage * target_size))
        for db_name, percentage in db_percentages.items()
    }
    
    # Check if initial allocation exceeds available instances
    for db_name, target_count in initial_allocation.items():
        available = db_counts[db_name]
        if target_count > available:
            print(f"Warning: Requested {target_count} instances from {db_name}, but only {available} available")
            print(f"Adjusting allocation for {db_name} to {available} instances")
            initial_allocation[db_name] = available
    
    # Adjust to meet target size exactly
    total_allocated = sum(initial_allocation.values())
    adjustment_needed = target_size - total_allocated
    
    if adjustment_needed != 0:
        print(f"Adjusting allocation to match target size exactly (adjustment: {adjustment_needed})")
        
        if adjustment_needed > 0:
            # Need to add instances - prioritize databases with largest proportions
            # that have room to grow
            room_to_grow = {
                db_name: db_counts[db_name] - alloc
                for db_name, alloc in initial_allocation.items()
                if db_counts[db_name] > alloc
            }
            
            if not room_to_grow:
                print("Cannot add more instances - all databases at capacity")
            else:
                # Sort by original percentage (highest first) and add instances
                db_to_adjust = sorted(
                    room_to_grow.keys(),
                    key=lambda db: db_percentages[db],
                    reverse=True
                )
                
                for _ in range(min(adjustment_needed, sum(room_to_grow.values()))):
                    for db_name in db_to_adjust:
                        if room_to_grow[db_name] > 0:
                            initial_allocation[db_name] += 1
                            room_to_grow[db_name] -= 1
                            adjustment_needed -= 1
                            break
                            
                    if adjustment_needed == 0:
                        break
        else:
            # Need to remove instances - start with smallest proportions
            # that have more than the minimum
            can_reduce = {
                db_name: alloc - min_instances_per_db
                for db_name, alloc in initial_allocation.items()
                if alloc > min_instances_per_db
            }
            
            if not can_reduce:
                print("Cannot reduce instances - all databases at minimum")
            else:
                # Sort by original percentage (lowest first) and remove instances
                db_to_adjust = sorted(
                    can_reduce.keys(),
                    key=lambda db: db_percentages[db]
                )
                
                adjustment_needed = abs(adjustment_needed)  # Convert to positive number
                for _ in range(min(adjustment_needed, sum(can_reduce.values()))):
                    for db_name in db_to_adjust:
                        if can_reduce[db_name] > 0:
                            initial_allocation[db_name] -= 1
                            can_reduce[db_name] -= 1
                            adjustment_needed -= 1
                            break
                            
                    if adjustment_needed == 0:
                        break
    
    # Final allocation
    final_allocation = initial_allocation
    actual_target = sum(final_allocation.values())
    
    print(f"Final allocation: {actual_target} total instances")
    for db_name, count in sorted(final_allocation.items(), 
                                key=lambda x: (db_counts[x[0]] / total_instances), 
                                reverse=True):
        original_pct = (db_counts[db_name] / total_instances) * 100
        new_pct = (count / actual_target) * 100
        print(f"  - {db_name}: {count} instances ({new_pct:.1f}%, original: {original_pct:.1f}%)")
    
    # Step 3: Sample instances from each database
    print("Sampling instances...")
    stratified_sample = []
    
    for db_name, target_count in final_allocation.items():
        instances = db_instances[db_name]
        sampled = random.sample(instances, target_count)
        stratified_sample.extend(sampled)
    
    # Calculate statistics
    stats = {
        "original_size": total_instances,
        "stratified_size": len(stratified_sample),
        "target_size": target_size,
        "actual_size": actual_target,
        "database_counts": db_counts,
        "database_allocation": final_allocation,
        "database_percentages": {
            db_name: (count / total_instances) * 100 
            for db_name, count in db_counts.items()
        },
        "stratified_percentages": {
            db_name: (final_allocation[db_name] / actual_target) * 100
            for db_name in db_counts.keys()
        }
    }
    
    # Calculate difficulty distribution statistics
    difficulty_original = defaultdict(int)
    difficulty_stratified = defaultdict(int)
    
    for instance in data_list:
        difficulty_original[instance['difficulty']] += 1
    
    for instance in stratified_sample:
        difficulty_stratified[instance['difficulty']] += 1
    
    stats["difficulty_original"] = dict(difficulty_original)
    stats["difficulty_stratified"] = dict(difficulty_stratified)
    
    # Calculate database type distribution
    db_types_original = defaultdict(int)
    db_types_stratified = defaultdict(int)
    
    for instance in data_list:
        db_type = instance['database'].get('type', 'unknown').lower()
        db_types_original[db_type] += 1
    
    for instance in stratified_sample:
        db_type = instance['database'].get('type', 'unknown').lower()
        db_types_stratified[db_type] += 1
    
    stats["db_types_original"] = dict(db_types_original)
    stats["db_types_stratified"] = dict(db_types_stratified)
    
    # Calculate variance between original and stratified distributions
    db_distribution_variance = {
        db_name: abs(stats["stratified_percentages"][db_name] - stats["database_percentages"][db_name])
        for db_name in db_counts.keys()
    }
    
    stats["db_distribution_variance"] = db_distribution_variance
    stats["average_distribution_variance"] = sum(db_distribution_variance.values()) / len(db_distribution_variance)
    
    return stratified_sample, stats

def stratify_by_database_proportional(data_list, target_size, output_dir="proportional_stratified_output", min_instances_per_db=1):
    """
    Create a stratified subset that preserves the original database distribution proportions.
    
    Args:
        data_list (list): List of Text2SQL instance dictionaries
        target_size (int): Target number of instances for the stratified dataset
        output_dir (str): Directory to save stratified data and files
        min_instances_per_db (int): Minimum instances to include from each database
    
    Returns:
        int: Number of instances in the stratified dataset
    """
    # Step 1: Analyze dataset and select instances
    selected_instances, stats = analyze_and_select_instances_proportional(
        data_list, target_size, min_instances_per_db
    )
    
    # Step 2: Process and save selected instances
    processed_count = process_and_save_instances(selected_instances, output_dir)
    
    # Step 3: Print summary
    print_stratification_summary_proportional(stats)
    
    return processed_count

def process_and_save_instances(selected_instances, output_dir):
    """
    Process selected instances, copy necessary files, and save to the output directory.
    
    Args:
        selected_instances (list): List of selected Text2SQL instance dictionaries
        output_dir (str): Directory to save stratified data and files
    
    Returns:
        int: Number of processed instances
    """
    print(f"Creating directory structure and saving {len(selected_instances)} instances...")
    os.makedirs(output_dir, exist_ok=True)
    db_dir = os.path.join(output_dir, "databases")
    schema_dir = os.path.join(output_dir, "schemas")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(schema_dir, exist_ok=True)
    
    # Process each instance in the stratified sample
    for instance in tqdm(selected_instances, desc="Processing instances"):
        # Create copy to modify
        instance_copy = instance.copy()
        db_name = instance['database']['name']
        db_type = instance['database'].get('type', '').lower()
        
        # Create database-specific directories
        instance_db_dir = os.path.join(db_dir, db_name)
        instance_schema_dir = os.path.join(schema_dir, db_name)
        os.makedirs(instance_db_dir, exist_ok=True)
        os.makedirs(instance_schema_dir, exist_ok=True)
        
        # Handle database based on type
        if db_type == 'sqlite':
            # For SQLite databases, copy files and update paths
            if 'path' in instance['database']:
                new_db_paths = []
                for path in instance['database']['path']:
                    if not path:  # Skip empty paths
                        continue
                    filename = os.path.basename(path)
                    new_path = os.path.join(instance_db_dir, filename)
                    new_db_paths.append(new_path)
                    # Copy database file if it exists
                    if os.path.exists(path):
                        shutil.copy(path, new_path)
                
                instance_copy['database']['path'] = new_db_paths
            
            # Update and copy CSV files if present
            if 'csv_files' in instance['database']:
                new_csv_paths = []
                for path in instance['database']['csv_files']:
                    if not path:  # Skip empty paths
                        continue
                    filename = os.path.basename(path)
                    new_path = os.path.join(instance_db_dir, filename)
                    new_csv_paths.append(new_path)
                    # Copy CSV file if it exists
                    if os.path.exists(path):
                        shutil.copy(path, new_path)
                
                instance_copy['database']['csv_files'] = new_csv_paths
        elif db_type == 'snowflake':
            # For Snowflake databases, don't copy files since there are no local files
            # Just keep the database structure as is
            pass
        else:
            # For other database types, check if paths exist and copy if they do
            if 'path' in instance['database'] and instance['database']['path']:
                new_db_paths = []
                for path in instance['database']['path']:
                    if not path:  # Skip empty paths
                        continue
                    filename = os.path.basename(path)
                    new_path = os.path.join(instance_db_dir, filename)
                    new_db_paths.append(new_path)
                    # Copy database file if it exists
                    if os.path.exists(path):
                        shutil.copy(path, new_path)
                
                if new_db_paths:  # Only update if we have valid paths
                    instance_copy['database']['path'] = new_db_paths
        
        # Update and copy schema files (for all database types)
        if 'schemas' in instance and 'path' in instance['schemas']:
            new_schema_paths = []
            for path in instance['schemas']['path']:
                if not path:  # Skip empty paths
                    continue
                filename = os.path.basename(path)
                new_path = os.path.join(instance_schema_dir, filename)
                new_schema_paths.append(new_path)
                # Copy schema file if it exists
                if os.path.exists(path):
                    shutil.copy(path, new_path)
            
            if new_schema_paths:  # Only update if we have valid paths
                instance_copy['schemas']['path'] = new_schema_paths
        
        # Save instance as JSON file
        output_file = os.path.join(output_dir, f"instance_{instance['id']}.json")
        with open(output_file, 'w') as f:
            json.dump(instance_copy, f, indent=4)
    
    return len(selected_instances)

def print_stratification_summary_proportional(stats):
    """
    Print a summary of the proportional stratification process.
    
    Args:
        stats (dict): Statistics dictionary from the stratification process
    """
    print("\nProportional Database Stratification Complete!")
    print(f"Original dataset size: {stats['original_size']} instances")
    print(f"Stratified dataset size: {stats['stratified_size']} instances")
    
    # Print database distribution comparison
    print("\nDatabase distribution comparison (Original → Stratified):")
    print("-" * 70)
    print(f"{'Database':<25} {'Original %':<15} {'Stratified %':<15} {'Variance':<15}")
    print("-" * 70)
    
    for db_name in sorted(stats["database_percentages"].keys(), 
                          key=lambda x: stats["database_percentages"][x], 
                          reverse=True):
        orig_pct = stats["database_percentages"][db_name]
        strat_pct = stats["stratified_percentages"][db_name]
        variance = stats["db_distribution_variance"][db_name]
        
        print(f"{db_name:<25} {orig_pct:>6.1f}%{'':<8} {strat_pct:>6.1f}%{'':<8} {variance:>+6.1f}%")
    
    print("-" * 70)
    print(f"Average distribution variance: {stats['average_distribution_variance']:.2f}%")
    
    # Print difficulty distribution
    print("\nDifficulty distribution (not controlled, for information only):")
    print("Original vs Stratified:")
    
    for difficulty in sorted(stats["difficulty_original"].keys()):
        orig_count = stats["difficulty_original"][difficulty]
        orig_pct = (orig_count / stats['original_size']) * 100
        
        strat_count = stats["difficulty_stratified"].get(difficulty, 0)
        strat_pct = (strat_count / stats['stratified_size']) * 100 if stats['stratified_size'] > 0 else 0
        
        print(f"  - {difficulty}: {orig_pct:.1f}% → {strat_pct:.1f}% ({strat_pct-orig_pct:+.1f}%)")
    
    # Print database type distribution
    print("\nDatabase type distribution:")
    print("Original vs Stratified:")
    
    for db_type in sorted(stats["db_types_original"].keys()):
        orig_count = stats["db_types_original"][db_type]
        orig_pct = (orig_count / stats['original_size']) * 100
        
        strat_count = stats["db_types_stratified"].get(db_type, 0)
        strat_pct = (strat_count / stats['stratified_size']) * 100 if stats['stratified_size'] > 0 else 0
        
        print(f"  - {db_type}: {orig_pct:.1f}% → {strat_pct:.1f}% ({strat_pct-orig_pct:+.1f}%)")


def analyze_text2sql_dataset(data_list):
    """
    Perform statistical analysis on the Text2SQL dataset
    
    Args:
        data_list (list): List of Text2SQL instance dictionaries
    
    Returns:
        dict: Statistical analysis results
    """
    stats = {
        "total_instances": len(data_list),
        "difficulty_distribution": defaultdict(int),
        "database_distribution": defaultdict(int),
        "question_metrics": {
            "avg_char_length": 0,
            "avg_word_length": 0,
            "entity_presence": 0,
            "number_presence": 0,
            "negation_presence": 0,
            "comparative_presence": 0,
            "superlative_presence": 0,
            "avg_table_overlap": 0,
            "avg_column_overlap": 0
        },
        "sql_metrics": {
            "avg_char_length": 0,
            "avg_tables_count": 0,
            "avg_join_count": 0,
            "avg_where_conditions": 0,
            "avg_subquery_count": 0,
            "avg_aggregation_functions": 0,
            "tables_used": defaultdict(int)
        },
        "difficulty_metrics": {}
    }
    
    # Collect metrics
    total_question_char_length = 0
    total_question_word_length = 0
    total_entity_presence = 0
    total_number_presence = 0
    total_negation_presence = 0
    total_comparative_presence = 0
    total_superlative_presence = 0
    total_table_overlap = 0
    total_column_overlap = 0
    
    total_sql_char_length = 0
    total_tables_count = 0
    total_join_count = 0
    total_where_conditions = 0
    total_subquery_count = 0
    total_aggregation_count = 0
    
    # Collect metrics by difficulty
    difficulty_metrics = defaultdict(lambda: {
        "count": 0,
        "avg_question_length": 0,
        "avg_sql_length": 0,
        "avg_tables_count": 0,
        "avg_join_count": 0,
        "avg_where_conditions": 0,
        "avg_subquery_count": 0
    })
    
    for instance in data_list:
        # Basic distribution counts
        difficulty = instance['difficulty']
        db_name = instance['database']['name']
        
        stats["difficulty_distribution"][difficulty] += 1
        stats["database_distribution"][db_name] += 1
        
        # Question metrics
        qa = instance.get('question_analysis', {})
        total_question_char_length += qa.get('char_length', 0)
        total_question_word_length += qa.get('word_length', 0)
        total_entity_presence += 1 if qa.get('has_entities', False) else 0
        total_number_presence += 1 if qa.get('has_numbers', False) else 0
        total_negation_presence += 1 if qa.get('has_negation', False) else 0
        total_comparative_presence += 1 if qa.get('has_comparatives', False) else 0
        total_superlative_presence += 1 if qa.get('has_superlatives', False) else 0
        total_table_overlap += qa.get('table_overlap_count', 0)
        total_column_overlap += qa.get('column_overlap_count', 0)
        
        # SQL metrics
        sa = instance.get('sql_analysis', {})
        total_sql_char_length += sa.get('char_length', 0)
        total_tables_count += sa.get('tables_count', 0)
        total_join_count += sa.get('join_count', 0)
        total_where_conditions += sa.get('where_conditions', 0)
        total_subquery_count += sa.get('subquery_count', 0)
        total_aggregation_count += sa.get('aggregation_function_count', 0)
        
        # Track tables used
        for table in sa.get('tables', []):
            stats["sql_metrics"]["tables_used"][table] += 1
        
        # Difficulty-specific metrics
        difficulty_metrics[difficulty]["count"] += 1
        difficulty_metrics[difficulty]["avg_question_length"] += qa.get('char_length', 0)
        difficulty_metrics[difficulty]["avg_sql_length"] += sa.get('char_length', 0)
        difficulty_metrics[difficulty]["avg_tables_count"] += sa.get('tables_count', 0)
        difficulty_metrics[difficulty]["avg_join_count"] += sa.get('join_count', 0)
        difficulty_metrics[difficulty]["avg_where_conditions"] += sa.get('where_conditions', 0)
        difficulty_metrics[difficulty]["avg_subquery_count"] += sa.get('subquery_count', 0)
    
    # Calculate averages
    total = stats["total_instances"]
    if total > 0:
        stats["question_metrics"]["avg_char_length"] = total_question_char_length / total
        stats["question_metrics"]["avg_word_length"] = total_question_word_length / total
        stats["question_metrics"]["entity_presence"] = total_entity_presence / total * 100
        stats["question_metrics"]["number_presence"] = total_number_presence / total * 100
        stats["question_metrics"]["negation_presence"] = total_negation_presence / total * 100
        stats["question_metrics"]["comparative_presence"] = total_comparative_presence / total * 100
        stats["question_metrics"]["superlative_presence"] = total_superlative_presence / total * 100
        stats["question_metrics"]["avg_table_overlap"] = total_table_overlap / total
        stats["question_metrics"]["avg_column_overlap"] = total_column_overlap / total
        
        stats["sql_metrics"]["avg_char_length"] = total_sql_char_length / total
        stats["sql_metrics"]["avg_tables_count"] = total_tables_count / total
        stats["sql_metrics"]["avg_join_count"] = total_join_count / total
        stats["sql_metrics"]["avg_where_conditions"] = total_where_conditions / total
        stats["sql_metrics"]["avg_subquery_count"] = total_subquery_count / total
        stats["sql_metrics"]["avg_aggregation_functions"] = total_aggregation_count / total
    
    # Calculate difficulty-specific averages
    for difficulty, metrics in difficulty_metrics.items():
        count = metrics["count"]
        if count > 0:
            for key in metrics:
                if key != "count":
                    metrics[key] = metrics[key] / count
    
    stats["difficulty_metrics"] = dict(difficulty_metrics)
    
    return stats

if __name__ == "__main__":

    from data_processor2 import process

    # instances = process(
    #     dataset='bird',
    #     split='dev',
    #     dataset_path='/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/Data/Bird/dev_20240627',
    #     save_enriched=False
    # )

    
    # # Run stratification with a target size of 500 instances
    # result_size = stratify_text2sql_data(instances, target_size=200, output_dir="stratified_output")
    # print(f"Final stratified dataset size: {result_size} instances")

    instances = process(
        dataset='spider',
        split='dev',
        dataset_path='/Users/sinabehnam/Desktop/Projects/Polito/Thesis/MA_text2SQL/Data/spider_data',
        save_enriched=False
    )

    # Run stratification with a target size of 500 instances
    result_size = stratify_by_database_proportional(instances, target_size=200, output_dir="spider_stratified_output_200")
    print(f"Final stratified dataset size: {result_size} instances")
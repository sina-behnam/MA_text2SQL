import argparse
from src.src_data.data_loader import DataLoader
from src.src_data.data_processor import DataProcessor
from src.features.features_engineering import FeatureExtractor


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run data processing and feature extraction pipeline')
    
    # Add arguments
    parser.add_argument('--dataset_type', type=str, default='spider', choices=['spider', 'bird'],
                        help='Dataset type to use (spider or bird)')
    parser.add_argument('--base_dir', type=str, default='data/raw/spider',
                        help='Base directory containing the dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use (train, dev, etc.)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Limit the number of examples to load')
    parser.add_argument('--example_id', type=int, default=0,
                        help='Example ID to extract features for')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create a dataset
    dataset = DataLoader.get_dataset(args.dataset_type, base_dir=args.base_dir, 
                                    split=args.split, limit=args.limit)
    
    # Process the dataset
    processor = DataProcessor(dataset)
    processor.process_dataset()
    
    # Extract features for an example
    extractor = FeatureExtractor(dataset)
    features = extractor.extract_features(args.example_id)
    
    print(f"Extracted features for example {args.example_id}:")
    print(f"Question: {features['question']}")
    print(f"Tables: {features['tables']}")
#!/usr/bin/env python
"""
PEIRCE + LMM Pipeline Runner

This script provides a user-friendly interface for running the PEIRCE + LMM pipeline.
It allows you to run the pipeline with default settings or customize various parameters.

Usage:
    python run_pipeline.py [options]

Options:
    --model MODEL         Name of the LLM model to use (default: llama3)
    --provider PROVIDER   Provider of the LLM model (default: ollama)
    --max-iterations N    Maximum number of refinement iterations (default: 3)
    --min-urban-terms N   Minimum number of urban terms required (default: 2)
    --output-dir DIR      Directory to save the outputs (default: data/pipeline_outputs)
    --num-samples N       Number of factual statements to process (default: 3)
    --input-file FILE     Path to a custom factual statements JSON file
    --prompts-file FILE   Path to a custom creative prompts JSON file
    --verbose             Enable verbose output
    --help                Show this help message and exit

Examples:
    # Run with default settings (processes first 3 factual statements)
    python run_pipeline.py

    # Run with custom model and provider
    python run_pipeline.py --model gpt-4o --provider openai

    # Process more samples
    python run_pipeline.py --num-samples 10

    # Use custom input files
    python run_pipeline.py --input-file my_facts.json --prompts-file my_prompts.json

    # Enable verbose output
    python run_pipeline.py --verbose
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the pipeline
from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the PEIRCE + LMM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (processes first 3 factual statements)
    python run_pipeline.py

    # Run with custom model and provider
    python run_pipeline.py --model gpt-4o --provider openai

    # Process more samples
    python run_pipeline.py --num-samples 10

    # Use custom input files
    python run_pipeline.py --input-file my_facts.json --prompts-file my_prompts.json

    # Enable verbose output
    python run_pipeline.py --verbose
        """
    )
    
    parser.add_argument("--model", type=str, default="llama3",
                        help="Name of the LLM model to use (default: llama3)")
    parser.add_argument("--provider", type=str, default="ollama",
                        help="Provider of the LLM model (default: ollama)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of refinement iterations (default: 3)")
    parser.add_argument("--min-urban-terms", type=int, default=2,
                        help="Minimum number of urban terms required (default: 2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the outputs (default: data/pipeline_outputs)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of factual statements to process (default: 3)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Path to a custom factual statements JSON file")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to a custom creative prompts JSON file")
    parser.add_argument("--verbose", action="store_true",default=True,
                        help="Enable verbose output")
    
    return parser.parse_args()

def load_custom_file(file_path, file_type):
    """
    Load a custom JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        file_type (str): Type of file ('factual_statements' or 'creative_prompts')
        
    Returns:
        list: Loaded data from the JSON file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded custom {file_type} from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_type} from {file_path}: {e}")
        print(f"Using default {file_type} instead.")
        return None

def print_progress(message, verbose=False):
    """
    Print progress message if verbose mode is enabled.
    
    Args:
        message (str): Progress message
        verbose (bool): Whether to print the message
    """
    if verbose:
        print(message)

def run_pipeline_with_progress(args):
    """
    Run the pipeline with progress tracking.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        dict: Pipeline results
    """
    start_time = time.time()
    
    # Print configuration
    print("\n=== PEIRCE + LMM Pipeline Configuration ===")
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Min urban terms: {args.min_urban_terms}")
    print(f"Output directory: {args.output_dir if args.output_dir else 'data/pipeline_outputs (default)'}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Custom input file: {args.input_file if args.input_file else 'None (using default)'}")
    print(f"Custom prompts file: {args.prompts_file if args.prompts_file else 'None (using default)'}")
    print(f"Verbose mode: {'Enabled' if args.verbose else 'Disabled'}")
    
    # Initialize the pipeline
    print("\n=== Initializing Pipeline ===")
    pipeline = UrbanLMMPipeline(
        model_name=args.model,
        provider=args.provider,
        max_iterations=args.max_iterations,
        min_urban_terms=args.min_urban_terms,
        output_dir=args.output_dir
    )
    
    # Load custom files if provided
    factual_statements = None
    creative_prompts = None
    
    if args.input_file:
        factual_statements = load_custom_file(args.input_file, "factual statements")
    
    if args.prompts_file:
        creative_prompts = load_custom_file(args.prompts_file, "creative prompts")
    
    # Run the pipeline
    print("\n=== Running Pipeline ===")
    try:
        pipeline_results = pipeline.run_pipeline(
            factual_statements=factual_statements,
            creative_prompts=creative_prompts,
            num_samples=args.num_samples
        )
        
        # Calculate and print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n=== Pipeline Execution Complete ===")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Processed {len(pipeline_results['outputs'])} creative prompts")
        print(f"Results saved to: {pipeline.output_dir}")
        
        return pipeline_results
    
    except Exception as e:
        print(f"\n=== Pipeline Execution Failed ===")
        print(f"Error: {e}")
        return None

def main():
    """
    Main function to run the pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print banner
    print("\n" + "=" * 60)
    print("               PEIRCE + LMM PIPELINE RUNNER                ")
    print("=" * 60)
    
    # Run the pipeline with progress tracking
    pipeline_results = run_pipeline_with_progress(args)
    
    if pipeline_results:
        print("\nPipeline execution completed successfully.")
    else:
        print("\nPipeline execution failed. Please check the error messages above.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
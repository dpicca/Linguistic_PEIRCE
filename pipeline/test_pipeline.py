"""
Test script for the PEIRCE + LMM pipeline runner.

This script demonstrates how to use the UrbanLMMPipeline class to run the complete pipeline.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipeline.urban_lmm_pipeline import UrbanLMMPipeline

def test_pipeline_with_sample():
    """
    Test the pipeline with a small sample of factual statements.
    """
    # Create a small sample of factual statements
    sample_statements = [
        {
            "id": "S001",
            "factual_statement": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
            "topic": "science"
        },
        {
            "id": "H001",
            "factual_statement": "The Declaration of Independence was signed in 1776.",
            "topic": "history"
        }
    ]
    
    # Create output directory for test results
    output_dir = os.path.join(project_root, "data", "pipeline_test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the pipeline
    pipeline = UrbanLMMPipeline(
        model_name="gpt-4o",  # Specify the model to use
        provider=None,        # Use default provider
        max_iterations=2,     # Limit to 2 iterations for testing
        min_urban_terms=2,    # Require at least 2 urban terms
        output_dir=output_dir # Save outputs to test directory
    )
    
    # Generate creative prompts for the sample statements
    creative_prompts = pipeline.generate_creative_prompts(sample_statements)
    
    # Print the generated prompts
    print("\nGenerated Creative Prompts:")
    for prompt in creative_prompts:
        print(f"ID: {prompt['id']}")
        print(f"Type: {prompt['prompt_type']}")
        print(f"Prompt: {prompt['prompt']}")
        print()
    
    # Save the prompts to a file
    prompts_file = os.path.join(output_dir, "sample_prompts.json")
    with open(prompts_file, 'w') as f:
        json.dump(creative_prompts, f, indent=2)
    print(f"Sample prompts saved to {prompts_file}")
    
    # Note: Uncommenting the following line would run the full pipeline,
    # which may take a significant amount of time and resources
    # pipeline_results = pipeline.run_pipeline(factual_statements=sample_statements)
    
    print("\nTo run the full pipeline, use the following command:")
    print("python -m pipeline.urban_lmm_pipeline --num-samples 2")
    
    # Explain the expected output
    print("\nExpected Pipeline Output:")
    print("1. Factual statements (input)")
    print("2. Creative prompts (generated from factual statements)")
    print("3. Generated outputs (after refinement)")
    print("4. Iteration logs (showing the refinement process)")
    print("5. Execution time")
    
    print("\nEach output includes:")
    print("- Hard critique results (term mapping, semantic fidelity)")
    print("- Soft critique results (parsimony, coherence, uncertainty, etc.)")
    print("- Feedback provided for improvement")
    print("- Final refined output")

def main():
    """
    Main function to run the test.
    """
    print("=== Testing PEIRCE + LMM Pipeline ===")
    test_pipeline_with_sample()
    print("\nTest completed.")

if __name__ == "__main__":
    main()
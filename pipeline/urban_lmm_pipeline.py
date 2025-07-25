"""
Pipeline runner for the PEIRCE + LMM project.

This module implements the complete pipeline for the PEIRCE + LMM project, including:
1. Input: factual statement (id, text, topic)
2. Prompt: generated rap / urban poem prompt, linked to the id
3. Generation: LLM produces creative version (with access to the knowledge base)
4. Hard Critique: mapping of terms and checking semantic fidelity
5. Soft Critique: calculating parsimony, coherence, uncertainty, etc.
6. Feedback & Refinement: iterative improvement until thresholds are met
7. Output: final creative version with scores, logs, and mappings
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components
from generation.urban_lmm import UrbanLMMGenerator
from critique.urban_lmm_hard_critique import UrbanLMMHardCritique
from critique.urban_lmm_soft import UrbanLMMSoftCritique
from refinement.urban_lmm_refinement import UrbanLMMRefinement

class UrbanLMMPipeline:
    """
    Pipeline runner for the PEIRCE + LMM project.
    
    This class integrates all components of the PEIRCE + LMM project into a complete pipeline.
    """
    
    def __init__(self, model_name: str = "llama3", provider: Optional[str] = 'ollama',
                 max_iterations: int = 3, min_urban_terms: int = 2,
                 output_dir: Optional[str] = None):
        """
        Initialize the pipeline runner.
        
        Args:
            model_name (str): Name of the LLM model to use.
            provider (str, optional): Provider of the LLM model (e.g., 'openai', 'ollama').
            max_iterations (int): Maximum number of refinement iterations.
            min_urban_terms (int): Minimum number of urban terms required in the output.
            output_dir (str, optional): Directory to save the outputs.
        """
        self.model_name = model_name
        self.provider = provider
        self.max_iterations = max_iterations
        self.min_urban_terms = min_urban_terms
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(project_root, "data", "pipeline_outputs")
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.generator = UrbanLMMGenerator(model_name=model_name, provider=provider)
        self.hard_critique = UrbanLMMHardCritique(min_urban_terms=min_urban_terms)
        self.soft_critique = UrbanLMMSoftCritique()
        self.refinement = UrbanLMMRefinement(model_name=model_name, max_iterations=max_iterations, provider=provider)
    
    def load_factual_statements(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load factual statements from a JSON file.
        
        Args:
            file_path (str, optional): Path to the JSON file containing factual statements.
            
        Returns:
            list: List of factual statements.
        """
        if file_path is None:
            file_path = os.path.join(project_root, "data", "factual_statements.json")
        
        with open(file_path, 'r') as f:
            factual_statements = json.load(f)
        
        return factual_statements
    
    def load_creative_prompts(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load creative prompts from a JSON file.
        
        Args:
            file_path (str, optional): Path to the JSON file containing creative prompts.
            
        Returns:
            list: List of creative prompts.
        """
        if file_path is None:
            file_path = os.path.join(project_root, "data", "creative_prompts.json")
        
        with open(file_path, 'r') as f:
            creative_prompts = json.load(f)
        
        return creative_prompts
    
    def generate_creative_prompts(self, factual_statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate creative prompts for factual statements.
        
        Args:
            factual_statements (list): List of factual statements.
            
        Returns:
            list: List of creative prompts.
        """
        creative_prompts = []
        
        for statement in factual_statements:
            statement_id = statement["id"]
            factual_text = statement["factual_statement"]
            topic = statement["topic"]
            
            # Generate rap prompt
            rap_prompt = {
                "id": f"{statement_id}_RAP",
                "factual_statement_id": statement_id,
                "prompt_type": "rap",
                "prompt": f"Rewrite this fact as a rap using urban slang and hip-hop terminology: '{factual_text}'",
                "topic": topic
            }
            creative_prompts.append(rap_prompt)
            
            # Generate urban poem prompt
            poem_prompt = {
                "id": f"{statement_id}_POEM",
                "factual_statement_id": statement_id,
                "prompt_type": "urban_poem",
                "prompt": f"Rewrite this fact as a short urban poem using street language and cultural references: '{factual_text}'",
                "topic": topic
            }
            creative_prompts.append(poem_prompt)
        
        return creative_prompts
    
    def run_pipeline(self, factual_statements: Optional[List[Dict[str, Any]]] = None, 
                    creative_prompts: Optional[List[Dict[str, Any]]] = None,
                    num_samples: int = -1) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            factual_statements (list, optional): List of factual statements. If None, loads from file.
            creative_prompts (list, optional): List of creative prompts. If None, generates from factual statements.
            num_samples (int): Number of factual statements to process. If -1, processes all.
            
        Returns:
            dict: Pipeline results.
        """
        start_time = time.time()
        
        # Step 1: Input - Load or use provided factual statements
        if factual_statements is None:
            factual_statements = self.load_factual_statements()
        
        if num_samples > 0:
            factual_statements = factual_statements[:num_samples]
        
        print(f"Processing {len(factual_statements)} factual statements.")
        
        # Create a dictionary of factual statements indexed by ID
        factual_statements_dict = {statement["id"]: statement for statement in factual_statements}
        
        # Step 2: Prompt - Generate or use provided creative prompts
        if creative_prompts is None:
            creative_prompts = self.generate_creative_prompts(factual_statements)
        
        print(f"Processing {len(creative_prompts)} creative prompts.")
        
        # Initialize results
        pipeline_results = {
            "factual_statements": factual_statements,
            "creative_prompts": creative_prompts,
            "outputs": [],
            "iterations": []
        }
        
        # Process each creative prompt
        for prompt_data in creative_prompts:
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["prompt"]
            factual_statement_id = prompt_data["factual_statement_id"]
            prompt_type = prompt_data["prompt_type"]
            topic = prompt_data["topic"]
            
            factual_statement = factual_statements_dict[factual_statement_id]["factual_statement"]
            
            print(f"\nProcessing prompt {prompt_id}...")
            
            # Step 3-6: Generation, Hard Critique, Soft Critique, Feedback & Refinement
            refinement_result = self.refinement.refine(prompt_text, factual_statement)
            
            # Add metadata to the refinement result
            refinement_result["id"] = f"{prompt_id}_REFINED"
            refinement_result["prompt_id"] = prompt_id
            refinement_result["factual_statement_id"] = factual_statement_id
            refinement_result["prompt_type"] = prompt_type
            refinement_result["topic"] = topic
            
            # Add to pipeline results
            pipeline_results["outputs"].append({
                "id": f"{prompt_id}_OUTPUT",
                "prompt_id": prompt_id,
                "factual_statement_id": factual_statement_id,
                "prompt_type": prompt_type,
                "topic": topic,
                "output": refinement_result["final_output"],
                "best_output": refinement_result["best_output"],
                "best_score": refinement_result["best_score"]
            })
            
            pipeline_results["iterations"].append(refinement_result)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        pipeline_results["execution_time"] = execution_time
        
        # Save results
        self.save_pipeline_results(pipeline_results)
        
        print(f"\nPipeline execution completed in {execution_time:.2f} seconds.")
        return pipeline_results
    
    def save_pipeline_results(self, pipeline_results: Dict[str, Any]) -> None:
        """
        Save pipeline results to JSON files.
        
        Args:
            pipeline_results (dict): Pipeline results.
        """
        # Create timestamp for output files
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save factual statements
        factual_statements_file = os.path.join(self.output_dir, f"factual_statements_{timestamp}.json")
        with open(factual_statements_file, 'w') as f:
            json.dump(pipeline_results["factual_statements"], f, indent=2)
        print(f"Factual statements saved to {factual_statements_file}")
        
        # Save creative prompts
        creative_prompts_file = os.path.join(self.output_dir, f"creative_prompts_{timestamp}.json")
        with open(creative_prompts_file, 'w') as f:
            json.dump(pipeline_results["creative_prompts"], f, indent=2)
        print(f"Creative prompts saved to {creative_prompts_file}")
        
        # Save outputs
        outputs_file = os.path.join(self.output_dir, f"outputs_{timestamp}.json")
        with open(outputs_file, 'w') as f:
            json.dump(pipeline_results["outputs"], f, indent=2)
        print(f"Outputs saved to {outputs_file}")
        
        # Save iterations
        iterations_file = os.path.join(self.output_dir, f"iterations_{timestamp}.json")
        with open(iterations_file, 'w') as f:
            json.dump(pipeline_results["iterations"], f, indent=2)
        print(f"Iterations saved to {iterations_file}")
        
        # Save complete pipeline results
        pipeline_results_file = os.path.join(self.output_dir, f"pipeline_results_{timestamp}.json")
        with open(pipeline_results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        print(f"Complete pipeline results saved to {pipeline_results_file}")


def main():
    """
    Main function to run the pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the PEIRCE + LMM pipeline.")
    parser.add_argument("--model", type=str, default="llama3", help="Name of the LLM model to use.")
    parser.add_argument("--provider", type=str, default='ollama', help="Provider of the LLM model.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of refinement iterations.")
    parser.add_argument("--min-urban-terms", type=int, default=2, help="Minimum number of urban terms required.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the outputs.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of factual statements to process.")
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = UrbanLMMPipeline(
        model_name=args.model,
        provider=args.provider,
        max_iterations=args.max_iterations,
        min_urban_terms=args.min_urban_terms,
        output_dir=args.output_dir
    )
    
    # Run the pipeline
    pipeline.run_pipeline(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
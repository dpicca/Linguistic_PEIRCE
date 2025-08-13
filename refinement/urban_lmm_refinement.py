"""
Iterative refinement mechanism for the PEIRCE + LMM pipeline.
This module uses feedback from both hard and soft critiques to improve the generated outputs.
"""

import os
import sys
import json
from pathlib import Path
import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from generation.urban_lmm import UrbanLMMGenerator
from critique.urban_lmm_hard_critique import UrbanLMMHardCritique
from critique.urban_lmm_soft import UrbanLMMSoftCritique

class UrbanLMMRefinement:
    """
    Iterative refinement mechanism for the PEIRCE + LMM pipeline.
    """
    
    def __init__(self, model_name="gpt-oss", max_iterations=3, provider='ollama'):
        """
        Initialize the refinement mechanism.
        
        Args:
            model_name (str): Name of the model to use.
            max_iterations (int): Maximum number of refinement iterations.
            provider (str, optional): Provider of the model (e.g., 'openai', 'ollama').
        """
        # Load API key from config
        config_path = os.path.join(project_root, "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        api_key = config.get(model_name, {}).get('api_key', 'your_api')
        if api_key == 'your_api':
            print("Warning: Using default API key. Please update config.yaml with your API key.")
        
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.provider = provider
        self.generator = UrbanLMMGenerator(model_name=model_name, provider=provider)
        self.hard_critique = UrbanLMMHardCritique(min_urban_terms=2)
        self.soft_critique = UrbanLMMSoftCritique()
        
    def _generate_refinement_prompt(self, original_prompt, generated_output, hard_critique_result, soft_critique_result):
        """
        Generate a prompt for refinement based on critique results.
        
        Args:
            original_prompt (str): The original prompt.
            generated_output (str): The generated output.
            hard_critique_result (dict): Results from the hard critique.
            soft_critique_result (dict): Results from the soft critique.
            
        Returns:
            str: Refinement prompt.
        """
        # Combine feedback from both critiques
        feedback = []
        if hard_critique_result and "feedback" in hard_critique_result:
            feedback.extend(hard_critique_result["feedback"])
        if soft_critique_result and "feedback" in soft_critique_result:
            feedback.extend(soft_critique_result["feedback"])
        
        # Create refinement prompt
        refinement_prompt = f"""Please improve the following creative output based on the feedback provided:

Original Prompt: {original_prompt}

Current Output: {generated_output}

Feedback:
{chr(10).join(f"- {item}" for item in feedback)}

Please provide an improved version that addresses the feedback while maintaining the creative style.
"""
        return refinement_prompt
    
    def refine(self, prompt, factual_statement):
        """
        Refine the generated output through iterative critique and improvement.
        
        Args:
            prompt (str): The prompt for generation.
            factual_statement (str): The original factual statement.
            
        Returns:
            dict: Refinement results.
        """
        # Initial generation
        print("Generating initial output...")
        output = self.generator.generate(prompt)
        
        iterations = []
        best_output = output
        best_score = 0.0
        
        # Iterative refinement
        for i in range(self.max_iterations):
            print(f"Iteration {i+1}/{self.max_iterations}")
            
            # Critique the output
            print("Performing hard critique...")
            hard_critique_result = self.hard_critique.critique(factual_statement, output)
            
            print("Performing soft critique...")
            soft_critique_result = self.soft_critique.critique(factual_statement, output)
            
            # Calculate combined score
            hard_score = 1.0 if hard_critique_result["passed"] else 0.0
            soft_score = soft_critique_result["overall_score"]
            combined_score = 0.5 * hard_score + 0.5 * soft_score
            
            # Store iteration results
            iteration_result = {
                "iteration": i+1,
                "output": output,
                "hard_critique": hard_critique_result,
                "soft_critique": soft_critique_result,
                "combined_score": combined_score
            }
            iterations.append(iteration_result)
            
            # Update best output if current is better
            if combined_score > best_score:
                best_output = output
                best_score = combined_score
            
            # Check if both critiques pass
            if hard_critique_result["passed"] and soft_critique_result["passed"]:
                print("Both critiques passed. Refinement complete.")
                break
            
            # Generate refinement prompt
            refinement_prompt = self._generate_refinement_prompt(
                prompt, output, hard_critique_result, soft_critique_result
            )
            
            # Generate refined output
            print("Generating refined output...")
            output = self.generator.generate(refinement_prompt)
        
        # Return refinement results
        return {
            "original_prompt": prompt,
            "factual_statement": factual_statement,
            "final_output": output,
            "best_output": best_output,
            "best_score": best_score,
            "iterations": iterations
        }

def load_factual_statements(file_path=None):
    """
    Load factual statements from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing factual statements.
        
    Returns:
        dict: Dictionary of factual statements indexed by ID.
    """
    if file_path is None:
        file_path = os.path.join(project_root, "data", "factual_statements.json")
        
    with open(file_path, 'r') as f:
        factual_statements = json.load(f)
        
    return {statement["id"]: statement for statement in factual_statements}

def load_creative_prompts(file_path=None):
    """
    Load creative prompts from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing creative prompts.
        
    Returns:
        list: List of creative prompts.
    """
    if file_path is None:
        file_path = os.path.join(project_root, "data", "creative_prompts.json")
        
    with open(file_path, 'r') as f:
        creative_prompts = json.load(f)
        
    return creative_prompts

def save_refinement_results(refinement_results, file_path=None):
    """
    Save refinement results to a JSON file.
    
    Args:
        refinement_results (list): List of refinement results.
        file_path (str): Path to save the JSON file.
    """
    if file_path is None:
        file_path = os.path.join(project_root, "data", "refinement_results.json")
        
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(refinement_results, f, indent=2)
        
    print(f"Refinement results saved to {file_path}")

def main():
    """
    Main function to refine generated outputs.
    """
    # Load factual statements and creative prompts
    factual_statements = load_factual_statements()
    creative_prompts = load_creative_prompts()
    
    # Initialize the refinement mechanism
    refinement = UrbanLMMRefinement(max_iterations=3)
    
    # Refine outputs for each prompt
    refinement_results = []
    
    # Limit to a few examples for testing
    for prompt_data in creative_prompts[:5]:  # Limit to 5 prompts for testing
        prompt_id = prompt_data['id']
        prompt_text = prompt_data['prompt']
        factual_statement_id = prompt_data['factual_statement_id']
        prompt_type = prompt_data['prompt_type']
        topic = prompt_data['topic']
        
        factual_statement = factual_statements[factual_statement_id]["factual_statement"]
        
        print(f"\nRefining output for prompt {prompt_id}...")
        refinement_result = refinement.refine(prompt_text, factual_statement)
        
        # Add metadata to the refinement result
        refinement_result["id"] = f"{prompt_id}_REFINED"
        refinement_result["prompt_id"] = prompt_id
        refinement_result["factual_statement_id"] = factual_statement_id
        refinement_result["prompt_type"] = prompt_type
        refinement_result["topic"] = topic
        
        refinement_results.append(refinement_result)
    
    # Save refinement results
    save_refinement_results(refinement_results)
    
    # Print summary
    print("\nRefinement complete.")
    print(f"Processed {len(refinement_results)} prompts.")

if __name__ == "__main__":
    main()

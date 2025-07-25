"""
Initial generation component for the PEIRCE + LMM pipeline.
This module generates creative outputs (raps and urban poems) based on prompts,
conditioned to use terms or concepts from the urban_dict_lmm ontology.
"""

import json
import os
import sys
import random
import yaml
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from generation.abstract import GenerativeModel
from generation.model_factory import create_llm_instance

class UrbanLMMGenerator(GenerativeModel):
    """
    Generator for creative outputs using urban dictionary concepts.
    """
    
    def __init__(self, model_name="gpt-4o", urban_dict_path=None, num_samples=1, provider=None):
        """
        Initialize the generator.
        
        Args:
            model_name (str): Name of the model to use. Can include a provider prefix (e.g., "ollama:llama2").
            urban_dict_path (str): Path to the urban dictionary ontology file.
            num_samples (int): Number of samples to generate for each prompt.
            provider (str, optional): Provider to use. If None, the provider is determined from the model name.
                Valid values are 'openai' and 'ollama'.
        """
        super().__init__(model_name)
        
        # Create the appropriate LLM instance using the factory
        self.model = create_llm_instance(model_name, provider)
        self.num_samples = num_samples
        self.urban_terms = self._load_urban_terms(urban_dict_path)
        
    def _load_urban_terms(self, urban_dict_path):
        """
        Load urban dictionary terms from the ontology file.
        
        Args:
            urban_dict_path (str): Path to the urban dictionary ontology file.
            
        Returns:
            list: List of urban dictionary terms with their definitions.
        """
        if urban_dict_path is None:
            urban_dict_path = os.path.join(project_root, "data", "urban_dict_lmm.ttl")
        
        # Extract terms and definitions from the TTL file
        urban_terms = []
        try:
            with open(urban_dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract expressions and their forms
            import re
            expr_pattern = r'ex:expr_([^\s]+) a lmm:LMM_Expression[^;]*;[^;]*;[^;]*;[^;]*lmm:hasForm "([^"]+)"'
            expressions = re.findall(expr_pattern, content)
            
            # Extract senses and their definitions
            sense_pattern = r'ex:sense_([^\s]+)_1 a lmm:LMM_Sense[^;]*;[^;]*lmm:hasDefinition "([^"]+)"'
            senses = re.findall(sense_pattern, content)
            
            # Match expressions with their definitions
            expr_dict = {expr_id: form for expr_id, form in expressions}
            sense_dict = {sense_id: definition for sense_id, definition in senses}
            
            for expr_id, form in expr_dict.items():
                if expr_id in sense_dict:
                    urban_terms.append({
                        "term": form,
                        "definition": sense_dict[expr_id]
                    })
            
            # If no terms were extracted, use a fallback list
            if not urban_terms:
                urban_terms = self._get_fallback_urban_terms()
                
        except Exception as e:
            print(f"Error loading urban dictionary terms: {e}")
            urban_terms = self._get_fallback_urban_terms()
            
        return urban_terms
    
    def _get_fallback_urban_terms(self):
        """
        Get a fallback list of urban dictionary terms.
        
        Returns:
            list: List of urban dictionary terms with their definitions.
        """
        return [
            {"term": "lit", "definition": "When something is very good, exciting, or fun"},
            {"term": "flex", "definition": "To show off or boast"},
            {"term": "vibe", "definition": "A feeling or atmosphere"},
            {"term": "squad", "definition": "A group of friends"},
            {"term": "slay", "definition": "To do something extremely well"},
            {"term": "fam", "definition": "Friends or family"},
            {"term": "dope", "definition": "Cool or awesome"},
            {"term": "woke", "definition": "Being aware of social issues"},
            {"term": "savage", "definition": "Someone who does something bold without caring about consequences"},
            {"term": "ghost", "definition": "To suddenly stop communicating with someone"},
            {"term": "salty", "definition": "Being upset or bitter about something"},
            {"term": "extra", "definition": "Over the top or excessive"},
            {"term": "basic", "definition": "Someone who follows mainstream trends without originality"},
            {"term": "lowkey", "definition": "Secretly or subtly"},
            {"term": "highkey", "definition": "Obviously or openly"},
            {"term": "sus", "definition": "Suspicious or questionable"},
            {"term": "cap", "definition": "A lie or to lie"},
            {"term": "no cap", "definition": "No lie or telling the truth"},
            {"term": "fire", "definition": "Something that is amazing or excellent"},
            {"term": "bet", "definition": "Agreement or affirmation"}
        ]
    
    def _enhance_prompt_with_urban_terms(self, prompt):
        """
        Enhance the prompt with urban dictionary terms.
        
        Args:
            prompt (str): The original prompt.
            
        Returns:
            str: The enhanced prompt.
        """
        # Select a random subset of urban terms to include in the prompt
        selected_terms = random.sample(self.urban_terms, min(10, len(self.urban_terms)))
        
        # Create a string with the selected terms and their definitions
        terms_str = "\n".join([f"- {term['term']}: {term['definition']}" for term in selected_terms])
        
        # Enhance the prompt
        enhanced_prompt = f"""{prompt}

To make your response more authentic, try to incorporate some of these urban slang terms:

{terms_str}

Make sure to maintain the factual content of the original statement while using creative language.
"""
        return enhanced_prompt
    
    def generate_direct(self, prompt, **kwargs):
        """
        Generate creative outputs based on the prompt directly.
        
        Args:
            prompt (str): The prompt for generation.
            **kwargs: Additional arguments for the generation model.
            
        Returns:
            list: List of generated outputs.
        """
        # Enhance the prompt with urban dictionary terms
        enhanced_prompt = self._enhance_prompt_with_urban_terms(prompt)
        
        # Create messages for the model
        messages = [
            {"role": "system", "content": "You are a creative assistant that specializes in urban slang and cultural expressions."},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        # Generate outputs
        outputs = []
        for _ in range(self.num_samples):
            try:
                # Check if the model is an instance of Ollama
                if hasattr(self.model, 'generate_chat_completion'):
                    # Use the Ollama-specific method
                    response = self.model.generate_chat_completion(
                        messages=messages,
                        temperature=0.7,  # Higher temperature for more creative outputs
                        max_tokens=1000
                    )
                    
                    # Handle streaming response format where multiple JSON objects are returned
                    if isinstance(response, str):
                        # Split the response by newlines to handle JSON objects on separate lines
                        lines = response.strip().split('\n')
                        content_parts = []
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            try:
                                # Try to parse each line as a JSON object
                                json_obj = json.loads(line)
                                if "message" in json_obj and "content" in json_obj["message"]:
                                    content_parts.append(json_obj["message"]["content"])
                            except json.JSONDecodeError:
                                # If line parsing fails, try regex as a fallback for this line
                                try:
                                    import re
                                    json_match = re.search(r'(\{.*\})', line, re.DOTALL)
                                    if json_match:
                                        json_str = json_match.group(1)
                                        json_obj = json.loads(json_str)
                                        if "message" in json_obj and "content" in json_obj["message"]:
                                            content_parts.append(json_obj["message"]["content"])
                                except (json.JSONDecodeError, Exception):
                                    continue
                        
                        if content_parts:
                            output = "".join(content_parts)
                        else:
                            output = "Error: Could not extract content from response"
                    elif response and "message" in response:
                        # Handle single JSON object response
                        output = response["message"]["content"]
                    else:
                        output = "Error: No valid response from model"
                else:
                    # Use the OpenAI-specific method
                    response = self.model.client.chat.completions.create(
                        model=self.model_name,
                        temperature=0.7,  # Higher temperature for more creative outputs
                        frequency_penalty=0.5,  # Encourage diversity
                        max_tokens=1000,
                        messages=messages
                    )
                    output = response.choices[0].message.content
                
                outputs.append(output)
            except Exception as e:
                print(f"Error generating output: {e}")
                outputs.append(f"Error generating output: {str(e)}")
            
        return outputs
    
    def generate(self, prompt, **kwargs):
        """
        Override the generate method from GenerativeModel.
        This is a simplified version that directly calls generate_direct.
        
        Args:
            prompt (str): The prompt for generation.
            **kwargs: Additional arguments for the generation model.
            
        Returns:
            str: The generated output.
        """
        outputs = self.generate_direct(prompt, **kwargs)
        return outputs[0] if outputs else "Error generating output"

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

def save_generated_outputs(outputs, file_path=None):
    """
    Save generated outputs to a JSON file.
    
    Args:
        outputs (list): List of generated outputs.
        file_path (str): Path to save the JSON file.
    """
    if file_path is None:
        file_path = os.path.join(project_root, "data", "generated_outputs.json")
        
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(outputs, f, indent=2)
        
    print(f"Generated outputs saved to {file_path}")

def main():
    """
    Main function to generate creative outputs.
    """
    # Load creative prompts
    creative_prompts = load_creative_prompts()
    
    # Initialize the generator
    generator = UrbanLMMGenerator(num_samples=1)
    
    # Generate outputs for each prompt
    generated_outputs = []
    
    for prompt_data in creative_prompts:
        prompt_id = prompt_data['id']
        prompt_text = prompt_data['prompt']
        factual_statement_id = prompt_data['factual_statement_id']
        prompt_type = prompt_data['prompt_type']
        topic = prompt_data['topic']
        
        print(f"Generating output for prompt {prompt_id}...")
        outputs = generator.generate(prompt_text)
        
        for i, output in enumerate(outputs):
            generated_output = {
                'id': f"{prompt_id}_OUT{i+1}",
                'prompt_id': prompt_id,
                'factual_statement_id': factual_statement_id,
                'prompt_type': prompt_type,
                'topic': topic,
                'output': output
            }
            generated_outputs.append(generated_output)
    
    # Save generated outputs
    save_generated_outputs(generated_outputs)

if __name__ == "__main__":
    main()
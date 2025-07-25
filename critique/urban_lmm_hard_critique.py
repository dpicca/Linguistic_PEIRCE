"""
Hard critique component for the PEIRCE + LMM pipeline.
This module implements the Hard Critique (Formal/Symbolic Check) step of the pipeline,
which verifies whether terms/concepts from the urban_dict_lmm ontology are used in the generated outputs
and whether the semantic structure of the output maintains the key elements of the original fact.
"""

import os
import sys
import re
import json
from pathlib import Path
import spacy
from collections import Counter

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from critique.abstract import CritiqueModel

class UrbanLMMHardCritique(CritiqueModel):
    """
    Hard critique component that performs formal/symbolic checks on generated creative text.
    
    This component:
    1. Maps creative terms to concepts in the LMM/Urban Dictionary knowledge base
    2. Verifies that the creative text includes at least X terms from the knowledge base
    3. Checks that the logical structure of the factual statement is respected
    4. Provides feedback if the verification fails
    """
    
    def __init__(self, generative_model=None, urban_dict_path=None, min_urban_terms=2, prompt_dict=None, type="hard"):
        """
        Initialize the critique component.
        
        Args:
            generative_model: The generative model to use for refinement (if needed).
            urban_dict_path (str): Path to the urban dictionary ontology file.
            min_urban_terms (int): Minimum number of urban terms required in the output.
            prompt_dict (dict): Dictionary of prompts for the critique model.
            type (str): Type of critique ("hard" or "soft").
        """
        super().__init__(generative_model, prompt_dict, type)
        self.urban_terms = self._load_urban_terms(urban_dict_path)
        self.min_urban_terms = min_urban_terms
        self.nlp = spacy.load("en_core_web_sm")
        
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
                        "term": form.lower(),
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
    
    def _semantic_formalization(self, text):
        """
        Map creative terms in the text to concepts in the urban dictionary knowledge base.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            list: List of urban dictionary terms found in the text.
        """
        text_lower = text.lower()
        terms_found = []
        
        # Common words that should not be counted as urban terms
        common_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
            'would', 'could', 'should', 'shall', 'might', 'must', 'let', 'make', 'made', 'may'
        }
        
        for term_data in self.urban_terms:
            term = term_data["term"]
            # Skip common words
            if term.lower() in common_words:
                continue
                
            # Check if the term is a whole word in the text
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                terms_found.append({
                    "term": term,
                    "definition": term_data["definition"]
                })
        
        return terms_found
    
    def _extract_key_entities(self, text):
        """
        Extract key entities from the text.
        
        Args:
            text (str): The text to extract entities from.
            
        Returns:
            list: List of key entities.
        """
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append(ent.text.lower())
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            entities.append(chunk.text.lower())
        
        return list(set(entities))
    
    def _check_logical_structure(self, original_fact, generated_output):
        """
        Check if the generated output maintains the logical structure of the original fact.
        
        Args:
            original_fact (str): The original factual statement.
            generated_output (str): The generated creative output.
            
        Returns:
            tuple: (structure_preserved, missing_entities)
        """
        original_entities = self._extract_key_entities(original_fact)
        output_entities = self._extract_key_entities(generated_output)
        
        # Process entities to handle variations
        processed_original_entities = []
        for entity in original_entities:
            # Skip very short entities (likely not meaningful)
            if len(entity.strip()) <= 2:
                continue
            processed_original_entities.append(entity)
        
        # Filter out duplicates and substrings
        filtered_original_entities = []
        for i, entity1 in enumerate(processed_original_entities):
            is_substring = False
            for j, entity2 in enumerate(processed_original_entities):
                if i != j and entity1 in entity2:
                    is_substring = True
                    break
            if not is_substring:
                filtered_original_entities.append(entity1)
        
        # Check which entities are present in the output
        present_entities = []
        missing_entities = []
        
        text_lower = generated_output.lower()
        
        for entity in filtered_original_entities:
            found = False
            entity_words = set(entity.lower().split())
            
            # Check for exact match or substring match
            for output_entity in output_entities:
                if entity in output_entity or output_entity in entity:
                    found = True
                    present_entities.append(entity)
                    break
            
            # If not found, check for word overlap
            if not found:
                for output_entity in output_entities:
                    output_entity_words = set(output_entity.lower().split())
                    # If more than half of the words in the entity are in the output entity
                    if len(entity_words.intersection(output_entity_words)) >= max(1, len(entity_words) // 2):
                        found = True
                        present_entities.append(entity)
                        break
            
            # If still not found, check for presence of key terms in the full text
            if not found:
                # For entities with multiple words, check if most words are present
                if len(entity_words) > 1:
                    words_found = sum(1 for word in entity_words if word in text_lower)
                    if words_found >= max(1, len(entity_words) // 2):
                        found = True
                        present_entities.append(entity)
            
            if not found:
                missing_entities.append(entity)
        
        # Calculate preservation score
        if not filtered_original_entities:
            structure_preserved = True
        else:
            preservation_score = len(present_entities) / len(filtered_original_entities)
            structure_preserved = preservation_score >= 0.7  # At least 70% of key entities should be present
        
        return structure_preserved, missing_entities
    
    def critique(self, original_fact, generated_output, **kwargs):
        """
        Critique the generated output.
        
        Args:
            original_fact (str): The original factual statement.
            generated_output (str): The generated creative output.
            **kwargs: Additional arguments for the critique.
            
        Returns:
            dict: Critique results.
        """
        # Step 1: Semantic formalization - Map creative terms to concepts in the knowledge base
        terms_found = self._semantic_formalization(generated_output)
        
        # Step 2: Verification - Check that the creative text includes at least X terms from the knowledge base
        urban_terms_pass = len(terms_found) >= self.min_urban_terms
        
        # Step 3: Verification - Check that the logical structure of the factual statement is respected
        structure_preserved, missing_entities = self._check_logical_structure(original_fact, generated_output)
        
        # Overall pass/fail
        passed = urban_terms_pass and structure_preserved
        
        # Step 4: Feedback - If the verification fails, provide feedback for improvement
        feedback = []
        if not urban_terms_pass:
            feedback.append(f"Add more slang. The output should include at least {self.min_urban_terms} urban slang terms. Found {len(terms_found)}: {', '.join([term['term'] for term in terms_found])}")
        if not structure_preserved:
            feedback.append(f"You changed the meaning of the fact. The output should maintain the key elements of the original fact. Missing entities: {', '.join(missing_entities)}")
        
        return {
            "passed": passed,
            "urban_terms_pass": urban_terms_pass,
            "urban_term_count": len(terms_found),
            "terms_found": [term["term"] for term in terms_found],
            "structure_preserved": structure_preserved,
            "missing_entities": missing_entities,
            "feedback": feedback
        }
        
    def shutdown(self, *args, **kwargs):
        """
        Shutdown the critique model.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # No resources to clean up for this critique model
        pass


def main():
    """
    Main function to demonstrate the usage of the UrbanLMMHardCritique class.
    """
    # Example usage
    critique = UrbanLMMHardCritique(min_urban_terms=2)
    
    # Example factual statement
    original_fact = "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    
    # Example generated output with urban slang
    generated_output = "Yo, when that H2O gets lit at a hundred degrees Celsius with standard pressure vibes, it starts flexing and turns into steam, no cap!"
    
    # Critique the generated output
    result = critique.critique(original_fact, generated_output)
    
    # Print the result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
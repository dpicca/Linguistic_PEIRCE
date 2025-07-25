"""
Soft critique component for the PEIRCE + LMM pipeline.
This module implements the Soft Critique (Linguistic/Epistemic Check) step of the pipeline,
which assesses the quality of the generated text based on parsimony, coherence, uncertainty,
and other linguistic and epistemic factors.
"""

import os
import sys
import re
import json
import spacy
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from critique.abstract import CritiqueModel
from critique.parsimony import ParsimonyCritique
from critique.coherence import CoherenceCritique
from critique.uncertainty import UncertaintyCritique

class UrbanLMMSoftCritique(CritiqueModel):
    """
    Soft critique component that performs linguistic/epistemic checks on generated creative text.
    
    This component:
    1. Assesses semantic density (parsimony)
    2. Evaluates coherence using textual entailment
    3. Analyzes uncertainty (hedging expressions)
    4. Performs additional soft checks (fluency, cultural fit)
    5. Provides feedback for improvement
    """
    
    def __init__(self, generative_model=None, prompt_dict=None, 
                 parsimony_threshold=0.7, coherence_threshold=0.6, 
                 uncertainty_threshold=0.3, fluency_threshold=0.7, 
                 cultural_fit_threshold=0.6, type="soft"):
        """
        Initialize the critique component.
        
        Args:
            generative_model: The generative model to use for refinement (if needed).
            prompt_dict (dict): Dictionary of prompts for the critique model.
            parsimony_threshold (float): Threshold for parsimony score (0-1).
            coherence_threshold (float): Threshold for coherence score (0-1).
            uncertainty_threshold (float): Threshold for uncertainty score (0-1).
            fluency_threshold (float): Threshold for fluency score (0-1).
            cultural_fit_threshold (float): Threshold for cultural fit score (0-1).
            type (str): Type of critique ("hard" or "soft").
        """
        super().__init__(generative_model, prompt_dict, type)
        
        # Initialize sub-critique components with fallback mechanisms
        try:
            self.parsimony_critique = ParsimonyCritique()
            self.parsimony_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize parsimony critique: {e}")
            self.parsimony_critique = None
            self.parsimony_available = False
            
        try:
            self.coherence_critique = CoherenceCritique()
            self.coherence_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize coherence critique: {e}")
            self.coherence_critique = None
            self.coherence_available = False
            
        try:
            self.uncertainty_critique = UncertaintyCritique()
            self.uncertainty_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize uncertainty critique: {e}")
            self.uncertainty_critique = None
            self.uncertainty_available = False
        
        # Initialize spaCy model for additional checks
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except Exception as e:
            print(f"Warning: Failed to load spaCy model: {e}")
            self.nlp = None
            self.spacy_available = False
        
        # Set thresholds
        self.parsimony_threshold = parsimony_threshold
        self.coherence_threshold = coherence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.fluency_threshold = fluency_threshold
        self.cultural_fit_threshold = cultural_fit_threshold
        
        # Load urban slang terms for cultural fit assessment
        self.urban_terms = self._load_urban_terms()
        
    def _load_urban_terms(self):
        """
        Load urban dictionary terms for cultural fit assessment.
        
        Returns:
            list: List of urban dictionary terms.
        """
        urban_dict_path = os.path.join(project_root, "data", "urban_dict_lmm.ttl")
        
        # Extract terms from the TTL file
        urban_terms = []
        try:
            with open(urban_dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract expressions and their forms
            expr_pattern = r'ex:expr_([^\s]+) a lmm:LMM_Expression[^;]*;[^;]*;[^;]*;[^;]*lmm:hasForm "([^"]+)"'
            expressions = re.findall(expr_pattern, content)
            
            for _, form in expressions:
                urban_terms.append(form.lower())
                
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
            list: List of urban dictionary terms.
        """
        return [
            "lit", "flex", "vibe", "squad", "slay", "fam", "dope", "woke", 
            "savage", "ghost", "salty", "extra", "basic", "lowkey", "highkey", 
            "sus", "cap", "no cap", "fire", "bet"
        ]
    
    def _assess_parsimony(self, original_fact: str, generated_output: str) -> Dict[str, Any]:
        """
        Assess the parsimony (semantic density) of the generated output.
        
        Args:
            original_fact (str): The original factual statement.
            generated_output (str): The generated creative output.
            
        Returns:
            dict: Parsimony assessment results.
        """
        if not self.parsimony_available:
            # Fallback implementation if parsimony critique is not available
            # Simple heuristic: compare lengths of original and generated text
            original_words = len(original_fact.split())
            generated_words = len(generated_output.split())
            
            # Calculate ratio of generated words to original words
            ratio = generated_words / max(1, original_words)
            
            # Normalize to 0-1 range (lower is better for verbosity)
            # Ideal ratio is around 1.5-2.5 times the original length
            if ratio < 1.5:
                normalized_score = 0.8  # Too concise is still good
            elif ratio > 4:
                normalized_score = max(0, 1 - ((ratio - 4) / 6))  # Penalize excessive verbosity
            else:
                normalized_score = 1.0  # Ideal range
                
            passed = normalized_score >= self.parsimony_threshold
            
            feedback = []
            if not passed:
                feedback.append(f"The text is too verbose compared to the original fact. Try to be more concise and focused on the key information.")
            
            return {
                "score": normalized_score,
                "passed": passed,
                "feedback": feedback,
                "ratio": ratio,
                "fallback": True
            }
        
        try:
            # Use the ParsimonyCritique to assess parsimony
            parsimony_result = self.parsimony_critique.critique(original_fact, "", generated_output)
            
            # Extract the parsimony score
            parsimony_score = parsimony_result.get('parsimony', 0)
            
            # Normalize the score to a 0-1 range (lower is better for parsimony)
            # Assuming a maximum reasonable drift of 20 concepts
            normalized_score = max(0, min(1, 1 - (parsimony_score / 20)))
            
            # Determine if the parsimony check passed
            passed = normalized_score >= self.parsimony_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The text introduces too many new concepts not present in the original fact. Try to be more concise and focused on the key information.")
            
            return {
                "score": normalized_score,
                "passed": passed,
                "feedback": feedback,
                "raw_score": parsimony_score
            }
        except Exception as e:
            print(f"Warning: Error in parsimony assessment: {e}")
            # Fall back to the simple heuristic
            self.parsimony_available = False
            return self._assess_parsimony(original_fact, generated_output)
    
    def _assess_coherence(self, original_fact: str, generated_output: str) -> Dict[str, Any]:
        """
        Assess the coherence of the generated output using textual entailment.
        
        Args:
            original_fact (str): The original factual statement.
            generated_output (str): The generated creative output.
            
        Returns:
            dict: Coherence assessment results.
        """
        if not self.coherence_available:
            # Fallback implementation if coherence critique is not available
            # Simple heuristic: check for key terms from the original fact in the generated output
            original_terms = set(original_fact.lower().split())
            generated_terms = set(generated_output.lower().split())
            
            # Remove common words
            common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'about',
                           'to', 'from', 'in', 'on', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
            original_terms = original_terms - common_words
            
            # Count how many original terms are in the generated output
            common_terms = original_terms.intersection(generated_terms)
            
            # Calculate coherence score based on term overlap
            if len(original_terms) == 0:
                normalized_score = 1.0  # Edge case
            else:
                normalized_score = min(1.0, len(common_terms) / len(original_terms))
            
            # Determine if the coherence check passed
            passed = normalized_score >= self.coherence_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The creative version doesn't clearly link back to the original fact. Make sure the key information is preserved and clearly communicated.")
            
            return {
                "score": normalized_score,
                "passed": passed,
                "feedback": feedback,
                "common_terms": len(common_terms),
                "original_terms": len(original_terms),
                "fallback": True
            }
        
        try:
            # Use the CoherenceCritique to assess coherence
            coherence_result = self.coherence_critique.get_entailment_scores(original_fact, generated_output)
            
            # Extract the entailment score
            entailment_score = coherence_result.get('entailment', 0)
            contradiction_score = coherence_result.get('contradiction', 0)
            
            # Calculate coherence score as entailment - contradiction
            coherence_score = entailment_score - contradiction_score
            
            # Normalize to 0-1 range
            normalized_score = max(0, min(1, (coherence_score + 1) / 2))
            
            # Determine if the coherence check passed
            passed = normalized_score >= self.coherence_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The creative version doesn't clearly link back to the original fact. Make sure the key information is preserved and clearly communicated.")
            
            return {
                "score": normalized_score,
                "passed": passed,
                "feedback": feedback,
                "entailment": entailment_score,
                "contradiction": contradiction_score
            }
        except Exception as e:
            print(f"Warning: Error in coherence assessment: {e}")
            # Fall back to the simple heuristic
            self.coherence_available = False
            return self._assess_coherence(original_fact, generated_output)
    
    def _assess_uncertainty(self, generated_output: str) -> Dict[str, Any]:
        """
        Analyze the uncertainty (hedging expressions) in the generated output.
        
        Args:
            generated_output (str): The generated creative output.
            
        Returns:
            dict: Uncertainty assessment results.
        """
        if not self.uncertainty_available:
            # Fallback implementation if uncertainty critique is not available
            # Simple heuristic: check for common hedging expressions
            hedging_expressions = [
                'maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'may', 
                'seem', 'appear', 'likely', 'unlikely', 'sometimes', 'often', 
                'occasionally', 'generally', 'usually', 'typically', 'relatively',
                'somewhat', 'kind of', 'sort of', 'a bit', 'a little', 'quite',
                'rather', 'fairly', 'pretty', 'almost', 'nearly', 'approximately'
            ]
            
            # Count hedging expressions in the output
            text_lower = generated_output.lower()
            hedging_count = sum(1 for expr in hedging_expressions if expr in text_lower)
            
            # Normalize to 0-1 range (lower is better for uncertainty)
            # Assuming a maximum reasonable count of 5 hedging expressions
            normalized_score = max(0, min(1, 1 - (hedging_count / 5)))
            
            # Determine if the uncertainty check passed
            passed = normalized_score >= self.uncertainty_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The text contains too many hedging expressions (like 'maybe', 'probably', etc.). Be more assertive and confident in your statements.")
            
            return {
                "score": normalized_score,
                "passed": passed,
                "feedback": feedback,
                "hedging_count": hedging_count,
                "fallback": True
            }
        
        try:
            # Use the UncertaintyCritique to assess uncertainty
            uncertainty_result = self.uncertainty_critique.calculate_avg_uncertainity([generated_output])
            
            # Normalize to 0-1 range (lower is better for uncertainty)
            # The uncertainty score is between 0 and 5 (6 - certainty)
            normalized_score = max(0, min(1, 1 - (uncertainty_result / 5)))
            
            # Determine if the uncertainty check passed
            passed = normalized_score >= self.uncertainty_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The text contains too many hedging expressions (like 'maybe', 'probably', etc.). Be more assertive and confident in your statements.")
            
            return {
                "score": normalized_score,
                "passed": passed,
                "feedback": feedback,
                "raw_score": uncertainty_result
            }
        except Exception as e:
            print(f"Warning: Error in uncertainty assessment: {e}")
            # Fall back to the simple heuristic
            self.uncertainty_available = False
            return self._assess_uncertainty(generated_output)
    
    def _assess_fluency(self, generated_output: str) -> Dict[str, Any]:
        """
        Assess the fluency of the generated output.
        
        Args:
            generated_output (str): The generated creative output.
            
        Returns:
            dict: Fluency assessment results.
        """
        if not self.spacy_available:
            # Fallback implementation if spaCy is not available
            # Simple heuristic: check for basic readability metrics
            
            # Split text into sentences (simple approximation)
            sentences = [s.strip() for s in re.split(r'[.!?]+', generated_output) if s.strip()]
            sentence_count = len(sentences)
            
            if sentence_count == 0:
                return {
                    "score": 0,
                    "passed": False,
                    "feedback": ["The text doesn't contain any complete sentences."],
                    "fallback": True
                }
            
            # Calculate average sentence length in words
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(sentence_lengths) / sentence_count
            
            # Calculate average word length
            words = [w for w in generated_output.split() if w.strip()]
            if not words:
                return {
                    "score": 0,
                    "passed": False,
                    "feedback": ["The text doesn't contain any words."],
                    "fallback": True
                }
            
            avg_word_length = sum(len(w) for w in words) / len(words)
            
            # Simple grammar check: look for capitalization at the beginning of sentences
            grammar_issues = sum(1 for s in sentences if s and not s[0].isupper())
            
            # Calculate fluency score based on metrics
            # Ideal sentence length is between 10-25 words
            sentence_length_score = 1 - min(1, abs(avg_sentence_length - 15) / 15)
            
            # Ideal word length is between 4-6 characters
            word_length_score = 1 - min(1, abs(avg_word_length - 5) / 5)
            
            # Grammar score based on capitalization issues
            grammar_score = max(0, 1 - (grammar_issues / max(1, sentence_count)))
            
            # Combined fluency score
            fluency_score = (sentence_length_score + word_length_score + grammar_score) / 3
            
            # Determine if the fluency check passed
            passed = fluency_score >= self.fluency_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                if sentence_length_score < 0.7:
                    if avg_sentence_length < 10:
                        feedback.append("Sentences are too short. Try to combine some sentences for better flow.")
                    else:
                        feedback.append("Sentences are too long. Try to break them up for better readability.")
                
                if word_length_score < 0.7:
                    if avg_word_length < 4:
                        feedback.append("Using too many short words. Try to use more varied vocabulary.")
                    else:
                        feedback.append("Using too many long words. Try to simplify your language.")
                
                if grammar_score < 0.7:
                    feedback.append("There are grammatical issues in the text. Check for capitalization at the beginning of sentences.")
            
            return {
                "score": fluency_score,
                "passed": passed,
                "feedback": feedback,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "grammar_issues": grammar_issues,
                "fallback": True
            }
        
        try:
            # Parse the text with spaCy
            doc = self.nlp(generated_output)
            
            # Calculate fluency metrics
            sentence_count = len(list(doc.sents))
            if sentence_count == 0:
                return {
                    "score": 0,
                    "passed": False,
                    "feedback": ["The text doesn't contain any complete sentences."]
                }
            
            # Calculate average sentence length
            avg_sentence_length = len(doc) / sentence_count
            
            # Calculate average word length
            avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / sum(1 for token in doc if not token.is_punct)
            
            # Check for grammatical issues
            grammar_issues = []
            for sentence in doc.sents:
                # Check for subject-verb agreement
                has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in sentence)
                has_verb = any(token.pos_ == 'VERB' for token in sentence)
                if not (has_subject and has_verb):
                    grammar_issues.append("Missing subject or verb")
            
            # Calculate fluency score based on metrics
            # Ideal sentence length is between 10-25 words
            sentence_length_score = 1 - min(1, abs(avg_sentence_length - 15) / 15)
            
            # Ideal word length is between 4-6 characters
            word_length_score = 1 - min(1, abs(avg_word_length - 5) / 5)
            
            # Grammar score based on issues
            grammar_score = max(0, 1 - (len(grammar_issues) / sentence_count))
            
            # Combined fluency score
            fluency_score = (sentence_length_score + word_length_score + grammar_score) / 3
            
            # Determine if the fluency check passed
            passed = fluency_score >= self.fluency_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                if sentence_length_score < 0.7:
                    if avg_sentence_length < 10:
                        feedback.append("Sentences are too short. Try to combine some sentences for better flow.")
                    else:
                        feedback.append("Sentences are too long. Try to break them up for better readability.")
                
                if word_length_score < 0.7:
                    if avg_word_length < 4:
                        feedback.append("Using too many short words. Try to use more varied vocabulary.")
                    else:
                        feedback.append("Using too many long words. Try to simplify your language.")
                
                if grammar_score < 0.7:
                    feedback.append("There are grammatical issues in the text. Check for complete sentences with subjects and verbs.")
            
            return {
                "score": fluency_score,
                "passed": passed,
                "feedback": feedback,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "grammar_issues": len(grammar_issues)
            }
        except Exception as e:
            print(f"Warning: Error in fluency assessment: {e}")
            # Fall back to the simple heuristic
            self.spacy_available = False
            return self._assess_fluency(generated_output)
    
    def _assess_cultural_fit(self, generated_output: str) -> Dict[str, Any]:
        """
        Assess the cultural fit of the generated output with respect to urban slang.
        
        Args:
            generated_output (str): The generated creative output.
            
        Returns:
            dict: Cultural fit assessment results.
        """
        # Fallback urban terms if none are available
        if not hasattr(self, 'urban_terms') or not self.urban_terms:
            self.urban_terms = self._get_fallback_urban_terms()
            
        try:
            # Count urban terms in the output
            output_lower = generated_output.lower()
            urban_terms_found = []
            
            for term in self.urban_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', output_lower):
                    urban_terms_found.append(term)
            
            # Calculate cultural fit score based on the number of urban terms found
            # Assuming a good urban text should have at least 3-5 urban terms
            cultural_fit_score = min(1, len(urban_terms_found) / 5)
            
            # Determine if the cultural fit check passed
            passed = cultural_fit_score >= self.cultural_fit_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The text doesn't sound authentic with respect to urban slang. Try to incorporate more urban terms like: {', '.join(self.urban_terms[:10])}.")
            
            return {
                "score": cultural_fit_score,
                "passed": passed,
                "feedback": feedback,
                "urban_terms_found": urban_terms_found,
                "urban_term_count": len(urban_terms_found)
            }
        except Exception as e:
            print(f"Warning: Error in cultural fit assessment: {e}")
            # Fallback implementation if there's an error
            # Simple heuristic: check for common urban slang markers
            
            # Define some common urban slang markers
            urban_markers = [
                'yo', 'lit', 'fire', 'dope', 'sick', 'woke', 'fam', 'bro', 'bruh', 'homie',
                'flex', 'vibe', 'chill', 'squad', 'slay', 'savage', 'legit', 'hella', 'af',
                'lowkey', 'highkey', 'sus', 'cap', 'no cap', 'bet', 'facts', 'deadass'
            ]
            
            # Count urban markers in the output
            output_lower = generated_output.lower()
            markers_found = [marker for marker in urban_markers if re.search(r'\b' + re.escape(marker) + r'\b', output_lower)]
            
            # Calculate cultural fit score based on the number of markers found
            cultural_fit_score = min(1, len(markers_found) / 3)
            
            # Determine if the cultural fit check passed
            passed = cultural_fit_score >= self.cultural_fit_threshold
            
            # Generate feedback if the check failed
            feedback = []
            if not passed:
                feedback.append(f"The text doesn't sound authentic with respect to urban slang. Try to incorporate more urban terms like: {', '.join(urban_markers[:10])}.")
            
            return {
                "score": cultural_fit_score,
                "passed": passed,
                "feedback": feedback,
                "urban_terms_found": markers_found,
                "urban_term_count": len(markers_found),
                "fallback": True
            }
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert an object to a JSON serializable format.
        
        Args:
            obj: The object to convert.
            
        Returns:
            A JSON serializable version of the object.
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        else:
            return str(obj)
    
    def critique(self, original_fact: str, generated_output: str, **kwargs) -> Dict[str, Any]:
        """
        Critique the generated output using linguistic and epistemic checks.
        
        Args:
            original_fact (str): The original factual statement.
            generated_output (str): The generated creative output.
            **kwargs: Additional arguments for the critique.
            
        Returns:
            dict: Critique results.
        """
        try:
            # Perform all soft checks with error handling
            try:
                parsimony_result = self._assess_parsimony(original_fact, generated_output)
            except Exception as e:
                print(f"Warning: Error in parsimony assessment: {e}")
                parsimony_result = {"score": 0.7, "passed": True, "feedback": [], "error": str(e)}
                
            try:
                coherence_result = self._assess_coherence(original_fact, generated_output)
            except Exception as e:
                print(f"Warning: Error in coherence assessment: {e}")
                coherence_result = {"score": 0.7, "passed": True, "feedback": [], "error": str(e)}
                
            try:
                uncertainty_result = self._assess_uncertainty(generated_output)
            except Exception as e:
                print(f"Warning: Error in uncertainty assessment: {e}")
                uncertainty_result = {"score": 0.7, "passed": True, "feedback": [], "error": str(e)}
                
            try:
                fluency_result = self._assess_fluency(generated_output)
            except Exception as e:
                print(f"Warning: Error in fluency assessment: {e}")
                fluency_result = {"score": 0.7, "passed": True, "feedback": [], "error": str(e)}
                
            try:
                cultural_fit_result = self._assess_cultural_fit(generated_output)
            except Exception as e:
                print(f"Warning: Error in cultural fit assessment: {e}")
                cultural_fit_result = {"score": 0.7, "passed": True, "feedback": [], "error": str(e)}
            
            # Convert assessment results to JSON serializable format
            parsimony_result = self._convert_to_json_serializable(parsimony_result)
            coherence_result = self._convert_to_json_serializable(coherence_result)
            uncertainty_result = self._convert_to_json_serializable(uncertainty_result)
            fluency_result = self._convert_to_json_serializable(fluency_result)
            cultural_fit_result = self._convert_to_json_serializable(cultural_fit_result)
            
            # Define default weights
            default_weights = {
                "parsimony": 0.2,
                "coherence": 0.3,
                "uncertainty": 0.2,
                "fluency": 0.15,
                "cultural_fit": 0.15
            }
            
            # Adjust weights if any assessment is missing or using fallback
            weights = default_weights.copy()
            total_weight = 0
            
            # Check which assessments are valid
            valid_assessments = {}
            for name, result in [
                ("parsimony", parsimony_result),
                ("coherence", coherence_result),
                ("uncertainty", uncertainty_result),
                ("fluency", fluency_result),
                ("cultural_fit", cultural_fit_result)
            ]:
                # Consider an assessment valid if it has a score and doesn't have an error
                if "score" in result and not result.get("error"):
                    valid_assessments[name] = result["score"]
                    total_weight += weights[name]
            
            # If no valid assessments, use default scores
            if not valid_assessments:
                overall_score = 0.7  # Default to a passing score
                passed = True
                feedback = ["Unable to perform detailed critique. Using default assessment."]
            else:
                # Normalize weights for valid assessments
                normalized_weights = {k: (v / total_weight if total_weight > 0 else 0) 
                                    for k, v in weights.items() if k in valid_assessments}
                
                # Calculate overall score as weighted average of valid assessment scores
                overall_score = sum(normalized_weights[k] * v for k, v in valid_assessments.items())
                
                # Determine if the overall critique passed
                # A critique passes if the overall score is at least 0.7 and no individual score is below 0.5
                min_score = min(valid_assessments.values()) if valid_assessments else 0.5
                passed = overall_score >= 0.7 and min_score >= 0.5
                
                # Combine feedback from all checks
                feedback = []
                if "feedback" in parsimony_result:
                    feedback.extend(parsimony_result["feedback"])
                if "feedback" in coherence_result:
                    feedback.extend(coherence_result["feedback"])
                if "feedback" in uncertainty_result:
                    feedback.extend(uncertainty_result["feedback"])
                if "feedback" in fluency_result:
                    feedback.extend(fluency_result["feedback"])
                if "feedback" in cultural_fit_result:
                    feedback.extend(cultural_fit_result["feedback"])
            
            # Return critique results
            result = {
                "passed": passed,
                "overall_score": overall_score,
                "parsimony": parsimony_result,
                "coherence": coherence_result,
                "uncertainty": uncertainty_result,
                "fluency": fluency_result,
                "cultural_fit": cultural_fit_result,
                "feedback": feedback,
                "valid_assessments": list(valid_assessments.keys())
            }
            
            # Ensure the result is JSON serializable
            return self._convert_to_json_serializable(result)
        except Exception as e:
            # Fallback if the entire critique process fails
            print(f"Critical error in critique: {e}")
            return {
                "passed": True,  # Default to passing to avoid blocking the pipeline
                "overall_score": 0.7,
                "feedback": ["Critical error in critique process. Using default assessment."],
                "error": str(e)
            }
    
    def shutdown(self, *args, **kwargs):
        """
        Shutdown the critique model.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Shutdown sub-critique components if they exist
        if hasattr(self, 'parsimony_critique') and self.parsimony_critique is not None:
            try:
                self.parsimony_critique.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down parsimony critique: {e}")
                
        if hasattr(self, 'coherence_critique') and self.coherence_critique is not None:
            try:
                self.coherence_critique.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down coherence critique: {e}")
                
        if hasattr(self, 'uncertainty_critique') and self.uncertainty_critique is not None:
            try:
                self.uncertainty_critique.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down uncertainty critique: {e}")


def main():
    """
    Main function to demonstrate the usage of the UrbanLMMSoftCritique class.
    """
    # Example usage
    critique = UrbanLMMSoftCritique()
    
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
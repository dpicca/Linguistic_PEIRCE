from .abstract import CritiqueModel

import re  
import spacy
from typing import List

from transformers.utils import logging
logging.set_verbosity_error() 

class ParsimonyCritique(CritiqueModel):
    """Measure semantic parsimony of an explanation relative to premise/output.

    Uses a lightweight concept extraction (nouns/adjectives via spaCy) to
    estimate semantic drift between the explanation and the premise/output.

    Examples:
        >>> pc = ParsimonyCritique()
        >>> exp = "Step 1: IF A THEN B. Therefore, B."
        >>> pc.critique(premise="A implies B", hypothesis="B", explanation=exp)
        {'parsimony': ...}
    """

    def __init__(self):
        super().__init__(generative_model=None, prompt_dict=None, type="soft")
        self.spacy_model = spacy.load("en_core_web_sm")

    def shutdown(self, *args, **kwargs):
        """Shutdown and release resources, if any.

        Examples:
            >>> ParsimonyCritique().shutdown()
        """
        pass

    def parse_explanation(self, exp: str):
        """Parse explanation text to extract steps, assumptions, and summary.

        Args:
            exp (str): Explanation text containing lines like "Step i:" and "Assumption:".

        Returns:
            dict: Dictionary with keys:
                - steps (list[str]): Extracted step strings without the "Step i:" prefix.
                - assumptions (list[str]): Extracted assumption strings.
                - summary (str): The concluding line starting with "Therefore," if present; otherwise last step.

        Examples:
            >>> pc = ParsimonyCritique()
            >>> exp = "Step 1: IF A THEN B. Therefore, B."
            >>> out = pc.parse_explanation(exp)
            >>> set(out.keys()) == {"steps", "assumptions", "summary"}
            True
        """
        step_pattern = r"(Step \d+:.*?(?=\n))"
        assumption_pattern = r"(Assumption:.*?(?=\n))"

        # Extract steps and assumptions
        steps = re.findall(step_pattern, exp, re.DOTALL)
        assumptions = re.findall(assumption_pattern, exp, re.DOTALL)

        steps = [step.split(":")[1].strip() for step in steps]
        assumptions = [assumption.split(":")[1].strip() for assumption in assumptions]

        summary_match = re.search(r'Therefore,.*', exp)
        summary = summary_match.group(0) if summary_match else steps[-1]
        
        return {"steps": steps, "assumptions": assumptions, "summary": summary}

    def extract_unique_concepts(self, text: str) -> List[str]:
        """Extract unique concept tokens (nouns/adjectives) from text.

        Args:
            text (str): Input text from which to extract concepts.

        Returns:
            List[str]: Unique set of noun/adjective tokens found in the text.

        Examples:
            >>> pc = ParsimonyCritique()
            >>> sorted(pc.extract_unique_concepts("The red car is fast."))  # doctest: +ELLIPSIS
            [...]
        """
        doc = self.spacy_model(text)
        nouns_adjectives = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
        return list(set(nouns_adjectives))

    def critique(self, premise: str, hypothesis: str, explanation: str) -> dict:
        """Compute parsimony (semantic drift) between explanation and premise/output.

        Args:
            premise (str): The original factual statement or premise text.
            hypothesis (str): The generated output or hypothesis text.
            explanation (str): The explanation text with steps/assumptions.

        Returns:
            dict: A dictionary with key 'parsimony' (int) indicating drift.

        Examples:
            >>> pc = ParsimonyCritique()
            >>> pc.critique("A implies B", "B", "Step 1: IF A THEN B. Therefore, B.")
            {'parsimony': ...}
        """
        
        # 1. Extract steps from explanation
        steps = self.parse_explanation(explanation)["steps"]
        
        # 2. Extract unique concepts from hypothesis and conclusion
        premise_sent = f"{premise} {hypothesis}"
        premise_concepts = self.extract_unique_concepts(premise_sent)

        # 3. Extract unique concepts from explanation
        expl_sent = " ".join(steps)
        expl_concepts = self.extract_unique_concepts(expl_sent)

        # Calculate semantic drift as set difference
        drift = len(set(expl_concepts).difference(set(premise_concepts)))
        
        critique_output = {}
        critique_output['parsimony'] = drift
        return critique_output

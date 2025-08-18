from .abstract import CritiqueModel
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np 
from typing import Optional

from transformers.utils import logging
logging.set_verbosity_error() 
# Source: https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
# @inproceedings{nie-etal-2020-adversarial,
#     title = "Adversarial {NLI}: A New Benchmark for Natural Language Understanding",
#     author = "Nie, Yixin  and
#       Williams, Adina  and
#       Dinan, Emily  and
#       Bansal, Mohit  and
#       Weston, Jason  and
#       Kiela, Douwe",
#     booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
#     year = "2020",
#     publisher = "Association for Computational Linguistics",
# }


class CoherenceCritique(CritiqueModel):
    """Evaluate coherence via NLI-based entailment checks.

    This critique uses a pretrained NLI model to estimate entailment,
    neutrality, and contradiction probabilities between steps in an
    explanation and summarizes them into a coherence score.

    Examples:
        >>> cc = CoherenceCritique(alias="roberta")
        >>> exp = "Step 1: IF X THEN Y. Step 2: IF Y THEN Z. Therefore, Z."
        >>> cc.critique(explanation=exp)
        {'coherence': ...}
    """

    def __init__(self, alias: str = "roberta"):
        """Initialize the NLI-based coherence critique.

        Args:
            alias (str): Model alias to select a specific pretrained NLI model.
                One of {'roberta', 'albert', 'bart', 'electra', 'xlnet'}.

        Examples:
            >>> CoherenceCritique(alias="roberta")
        """
        super().__init__(generative_model=None, prompt_dict=None, type="soft")

        alias_map = {
            "roberta": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            "albert": "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
            "bart": "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            "electra": "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli",
            "xlnet": "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"
        }
        self.tokenizer = AutoTokenizer.from_pretrained(alias_map[alias])
        self.model = AutoModelForSequenceClassification.from_pretrained(alias_map[alias])


    def shutdown(self, *args, **kwargs):
        """Shutdown and release resources held by the model, if any.

        This implementation is a no-op because the Hugging Face models are
        managed by the transformers library.

        Examples:
            >>> cc = CoherenceCritique()
            >>> cc.shutdown()
        """
        pass

    def get_entailment_scores(self, premise: str, hypothesis: str) -> dict:
        """Compute NLI probabilities between a premise and a hypothesis.

        Uses a pretrained sequence classification model to return entailment,
        neutral, and contradiction probabilities.

        Args:
            premise (str): The premise sentence.
            hypothesis (str): The hypothesis sentence to evaluate against the premise.

        Returns:
            dict: A dictionary with keys 'entailment', 'neutral', 'contradiction' mapped to probabilities.

        Examples:
            >>> cc = CoherenceCritique()
            >>> cc.get_entailment_scores("Cats are animals.", "Cats are mammals.")
            {'entailment': ..., 'neutral': ..., 'contradiction': ...}
        """
        tokenized_input_seq_pair = self.tokenizer.encode_plus(
            premise, hypothesis, 
            max_length=256,
            return_token_type_ids=True, 
            truncation=True
        )

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=None)
        
        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

        return{
            "entailment": predicted_probability[0],
            "neutral": predicted_probability[1],
            "contradiction": predicted_probability[2]  
        }
    

    def parse_explanation(self, exp: str) -> dict:
        """Parse a chain-of-thought style explanation into components.

        Extracts steps ("Step i:"), assumptions ("Assumption:"), and a
        concluding summary (line starting with "Therefore,") if present.

        Args:
            exp (str): Explanation text containing steps and assumptions.

        Returns:
            dict: A dictionary with keys 'steps' (list[str]), 'assumptions' (list[str]),
                and 'summary' (str).

        Examples:
            >>> cc = CoherenceCritique()
            >>> exp = "Step 1: IF X THEN Y. Step 2: IF Y THEN Z. Therefore, Z."
            >>> cc.parse_explanation(exp)
            {'steps': [...], 'assumptions': [...], 'summary': 'Therefore, Z.'}
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



    def internal_entailment(self, steps: list) -> dict:  
        """Compute internal entailment across consecutive steps.

        For each pair of consecutive statements in the explanation (IF ... THEN ...),
        compute entailment scores and average them by label.

        Args:
            steps (list): A list of step strings, typically from parse_explanation.

        Returns:
            dict: Mean probabilities for 'entailment', 'neutral', and 'contradiction'.

        Examples:
            >>> cc = CoherenceCritique()
            >>> cc.internal_entailment(["Step 1: IF A THEN B.", "Step 2: IF B THEN C.", "Therefore, C."])
            {'entailment': ..., 'neutral': ..., 'contradiction': ...}
        """
        scores = { "entailment": [], "neutral": [], "contradiction": []}
        for step in steps[:-1]:
            try:
                pe = step.split("THEN")
                if len(pe) < 2:
                    raise ValueError(f"Malformed step, expected 'IF ... THEN ...' structure: {step}")
                p = f'{pe[0].strip("IF").strip()[:-1]}.'
                e = pe[1]
                score = self.get_entailment_scores(p, e)
                for k, v in score.items():
                    scores[k].append(v)
            except Exception as exc:
                raise ValueError(f"Failed to compute entailment for step: {step}") from exc
        scores = {k: np.mean(v) for k,v in scores.items()}
        return scores

    def critique(self, premise: Optional[str] = None, hypothesis: Optional[str] = None, explanation: str = "") -> dict:
        """Compute a coherence score from an explanation using NLI.

        Args:
            premise (Optional[str]): Optional premise text (not used directly; kept for API symmetry).
            hypothesis (Optional[str]): Optional hypothesis text (not used directly; kept for API symmetry).
            explanation (str): An explanation with steps/assumptions to analyze.

        Returns:
            dict: A dictionary with key 'coherence' containing the numeric score.

        Examples:
            >>> cc = CoherenceCritique()
            >>> exp = "Step 1: IF Rain THEN Wet. Step 2: IF Wet THEN Slippery. Therefore, Slippery."
            >>> out = cc.critique(explanation=exp)
            >>> isinstance(out["coherence"], float)
            True
        """
        critique_output = {}
        # 1. Parse explanation
        exp_dict = self.parse_explanation(explanation)

        # 2. Perform internal entailment
        scores = self.internal_entailment(exp_dict["steps"])

        # 3. Return scores
        scores.update(
           { "score": scores["entailment"] - scores["contradiction"]},
        )

        critique_output['coherence'] = scores['score']
        return critique_output

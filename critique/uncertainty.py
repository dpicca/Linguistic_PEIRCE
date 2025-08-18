from .abstract import CritiqueModel
from certainty_estimator.predict_certainty import CertaintyEstimator
import torch
import numpy as np
import re 
from typing import Optional
class UncertaintyCritique(CritiqueModel):
      
    """Estimate epistemic uncertainty from explanations using a certainty model.

    Wraps a certainty estimator that returns scores from 1 (low certainty) to 6
    (high certainty) and converts them to uncertainty values.

    Examples:
        >>> uc = UncertaintyCritique()
        >>> exp = "Step 1: Assumption: It might rain. Therefore, roads could be wet."
        >>> out = uc.critique(explanation=exp)
        >>> 'uncertainty' in out
        True
    """

    def __init__(self):
        """Initialize the uncertainty critique using a certainty estimator.

        The underlying estimator outputs certainty scores in [1, 6];
        this class converts them to uncertainty scores by computing (6 - score).

        Examples:
            >>> UncertaintyCritique()
        """
        super().__init__(generative_model=None, prompt_dict=None, type="soft")
        self.use_cuda = True if torch.cuda.is_available() else False
        self.estimator = CertaintyEstimator('sentence-level', cuda=self.use_cuda)

    def shutdown(self, *args, **kwargs):
        """Shutdown and release resources if needed.

        Examples:
            >>> UncertaintyCritique().shutdown()
        """
        pass

    def parse_explanation(self, exp: str) -> dict:
            """Parse an explanation into steps, assumptions, and summary.

            Args:
                exp (str): Explanation text with lines such as "Step i:" and "Assumption:".

            Returns:
                dict: Dictionary with 'steps' (list[str]), 'assumptions' (list[str]), and 'summary' (str).

            Examples:
                >>> uc = UncertaintyCritique()
                >>> exp = "Step 1: Assumption: It might rain. Therefore, roads could be wet."
                >>> out = uc.parse_explanation(exp)
                >>> set(out.keys()) == {'steps', 'assumptions', 'summary'}
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


    def calculate_avg_uncertainity(self, texts: list) -> float:
        """Calculate average uncertainty for a list of texts.

        The underlying estimator returns certainty in [1, 6]. We convert to
        uncertainty by computing (6 - certainty) for each text, then take the mean.

        Args:
            texts (list): A list of strings to evaluate.

        Returns:
            float: Mean uncertainty across all texts.

        Examples:
            >>> uc = UncertaintyCritique()
            >>> round(uc.calculate_avg_uncertainity(["I think so."]), 2)  # doctest: +ELLIPSIS
            ...
        """
        return np.mean([ 6 - score for score in self.estimator.predict(texts)])

    def critique(self, premise: Optional[str] = None, hypothesis: Optional[str] = None, explanation: str = "") -> dict:
         # 1. parse explanation
         exp_dict = self.parse_explanation(explanation)

         # 2. Calculate uncertainty over extracted assumptions 
         assumption_uncertainty = self.calculate_avg_uncertainity(exp_dict["assumptions"])

         # 3. Calculate uncertainty over summary 
         summary_uncertainty = self.calculate_avg_uncertainity([exp_dict["summary"]])
         
         critique_output = {}
         critique_output['uncertainty'] = (assumption_uncertainty + summary_uncertainty) / 2
         return critique_output
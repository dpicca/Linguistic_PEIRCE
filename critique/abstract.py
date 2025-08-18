from abc import ABC, abstractmethod
from typing import Optional


class CritiqueModel(ABC):
    """Abstract base class for critique models.

    Subclasses implement specific critique strategies (e.g., coherence,
    parsimony, uncertainty) that evaluate generated text against criteria.

    Args:
        generative_model: Optional handle to the text generation model used by the pipeline.
        prompt_dict (Optional[dict]): Optional prompt configuration mapping used by some critiques.
        type (Optional[str]): Category of critique (e.g., 'hard', 'soft').

    Examples:
        >>> class MyCritique(CritiqueModel):
        ...     def critique(self, text: str) -> dict:
        ...         return {"score": 1.0}
        ...     def shutdown(self):
        ...         pass
        >>> mc = MyCritique(generative_model=None, prompt_dict=None, type="soft")
        >>> mc.critique("hello")
        {'score': 1.0}
    """

    def __init__(self, generative_model,
                 prompt_dict: Optional[dict] = None,
                 type: Optional[str] = None):
        self.generative_model = generative_model
        self.prompt_dict = prompt_dict
        self.type = type

    @abstractmethod
    def critique(self, *args, **kwargs):
        """Run the critique and return structured results.

        Implementations should accept inputs relevant to the critique and
        return a dictionary of metrics/scores.

        Args:
            *args: Positional arguments defined by the concrete critique implementation.
            **kwargs: Keyword arguments defined by the concrete critique implementation.

        Returns:
            dict: A mapping of metric names to their computed values.

        Examples:
            Basic subclass pattern:

            >>> class SimpleCrit(CritiqueModel):
            ...     def critique(self, text: str) -> dict:
            ...         return {"len": len(text)}
            ...     def shutdown(self):
            ...         pass
        """
        pass

    @abstractmethod
    def shutdown(self, *args, **kwargs):
        """Release any resources held by the critique implementation.

        Args:
            *args: Implementation-specific parameters.
            **kwargs: Implementation-specific keyword parameters.

        Examples:
            >>> class NoOp(CritiqueModel):
            ...     def critique(self, *_, **__):
            ...         return {}
            ...     def shutdown(self):
            ...         print("closed")
            >>> NoOp(None).shutdown()
            closed
        """
        pass

from __future__ import annotations

from pydantic import BaseModel


class MarkdownRenderable(BaseModel):
    """Base class for all schemas usable with `call_structured`.

    Pydantic v2's BaseModel does not honor @abstractmethod via its metaclass,
    so the base raises NotImplementedError to surface accidental misuse on call.
    `call_structured` also performs an explicit subclass check that the override
    exists at the class level (see core.py).
    """

    def to_markdown(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must override to_markdown(); "
            "all schemas passed to call_structured must implement it."
        )

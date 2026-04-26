"""
-------------------------------------------------------------------------------

    Authors:    Andre Cire          | https://www.andre-cire.com/
                Selva Nadarajah     | https://www.selva-nadarajah.com/
                Parshan Pakiman     | https://parshanpakiman.github.io/
                Negar Soheili       | https://www.negar-soheili.com/
                
    GitHub:     https://github.com/self-adapting-mdp-approximations
-------------------------------------------------------------------------------

Common policy interface for weakly coupled MDP examples.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple
from wmdp import StateComponent


ActionVector = Tuple[int, ...]


class Policy(ABC):
    """Common interface for policies over joint actions."""

    @abstractmethod
    def get_action(
        self,
        period: int,
        state: Sequence[StateComponent],
    ) -> ActionVector:
        """Return a feasible joint action for the given period."""

    def __call__(self, period: int, state: Sequence[StateComponent]) -> ActionVector:
        """Alias for ``get_action`` so the policy can be called directly."""
        return self.get_action(period=period, state=state)

import abc
from typing import Set
from rivapy.tools.interfaces import FactoryObject


class BaseModel(FactoryObject):
    @abc.abstractmethod
    def udls(self)->Set[str]:
        """Return the name of all underlyings modeled

        Returns:
            Set[str]: Set of the modeled underlyings.
        """
        pass
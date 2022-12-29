# -*- coding: utf-8 -*-


from abc import \
    abstractmethod as _abstractmethod, \
    ABCMeta as _ABCMeta


class BaseObject(metaclass=_ABCMeta):
    def __init__(self, obj_id: str):
        self.__obj_id = obj_id

    @property
    def obj_id(self) -> str:
        """
        Getter for object id.

        Returns:
            str: Object id.
        """
        return self.__obj_id

    @_abstractmethod
    def _validate_derived_base_object(self):
        pass

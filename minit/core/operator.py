from .object import Object


class Operator(Object):
    def type(self):
        return type(self)

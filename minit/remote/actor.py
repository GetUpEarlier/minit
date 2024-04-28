import pickle
import socket
from typing import List, Union

from .registry import create_object, get_function, get_object, is_object
from .function import FunctionRef
from .value import Value
from .object import ObjectRef

class Actor:
    id: int

    def connect(self, address: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((address, port))
        id_bytes = self.socket.recv(8)
        self.id = int.from_bytes(id_bytes, "little")
        print(f"serving as actor {self.id}")
        return self

    def act_forever(self):
        while True:
            size_bytes = self.socket.recv(8)
            size = int.from_bytes(size_bytes, "little")
            packet = self.socket.recv(size)
            [fn, *args] = pickle.loads(packet)
            fn: FunctionRef
            args: List[Union[Value, ObjectRef]]
            assert fn.location == self.id
            for arg in args:
                if isinstance(arg, ObjectRef):
                    assert arg.location == self.id
            local_args = [
                get_object(arg.id) if isinstance(arg, ObjectRef) else arg.value for arg in args
            ]
            local_fn = get_function(fn.name)
            local_result = local_fn(*local_args)
            result_type = type(local_result)
            if is_object(result_type):
                result = ObjectRef(self.id, create_object(local_result))
            else:
                result = Value(local_result)
            packet = pickle.dumps(result)
            self.socket.sendall(len(packet).to_bytes(8, "little"))
            self.socket.sendall(packet)

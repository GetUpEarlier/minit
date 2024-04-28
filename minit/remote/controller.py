import pickle
import socket
from typing import Union

from .object import ObjectRef
from .function import FunctionRef
from .value import Value

class Controller:
    def __init__(self, address: str, port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.socket.bind((address, port))
        self.socket.listen(64)
        self.port = self.socket.getsockname()[1]
        self.peers = []

    def wait_for_connect(self, nr_actors: int):
        peers = []
        while len(peers) < nr_actors:
            connection, _address = self.socket.accept()
            id = len(peers) + len(self.peers)
            connection.sendall(id.to_bytes(8, "little"))
            peers.append(connection)
        self.peers += peers

    def invoke_function(self, function: FunctionRef, *args: Union[Value, ObjectRef]):
        """blocking invoke"""
        peer: socket.socket = self.peers[function.location]
        args_bytes = pickle.dumps((function, *args))
        peer.sendall(len(args_bytes).to_bytes(8, "little"))
        peer.sendall(args_bytes)
        result_size_bytes = peer.recv(8)
        result_size = int.from_bytes(result_size_bytes, "little")
        result_bytes = peer.recv(result_size)
        result = pickle.loads(result_bytes)
        return result

import socket
from types import FunctionType
from typing import Any, List, Union

from .channel import Channel
from .object import ObjectRef

class Controller:
    peers: List[Channel]

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
            channel = Channel(connection)
            channel.send(id)
            peers.append(channel)
        self.peers += peers

    def invoke_function(self, actor: int, function: FunctionType, *args: Union[ObjectRef, Any]):
        """blocking invoke"""
        peer = self.peers[actor]
        print(f"invoking {function.__qualname__} at {actor}")
        peer.send((function.__qualname__, *args))
        result = peer.recv()
        return result

    def __len__(self):
        return len(self.peers)

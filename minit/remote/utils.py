import pickle
import socket
from typing import Any


def recv_bytes(channel: socket.socket, size: int) -> bytes:
    received_bytes = bytearray()
    while len(received_bytes) < size:
        received_bytes += channel.recv(size - len(received_bytes))
    return bytes(received_bytes)


def send_object(channel: socket.socket, object: Any):
    packet = pickle.dumps(object)
    channel.sendall(len(packet).to_bytes(8, "little"))
    channel.sendall(packet)


def recv_object(channel: socket.socket) -> Any:
    size_bytes = recv_bytes(channel, 8)
    size = int.from_bytes(size_bytes, "little")
    packet = recv_bytes(channel, size)
    return pickle.loads(packet)

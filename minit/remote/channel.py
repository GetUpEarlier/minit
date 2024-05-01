import socket
from typing import Any
from .utils import send_object, recv_object


class Channel:
    class Future:
        _value = None

        def __init__(self, socket: "socket.socket") -> None:
            self._socket = socket

        def fetch(self):
            if self._value is None:
                self._value = recv_object(self._socket)
                assert self._value is not None

        def get(self):
            self.fetch()
            return self._value

    def __init__(self, socket: "socket.socket") -> None:
        self._socket = socket
        self._last_future = None

    def send(self, object: Any) -> None:
        if self._last_future is not None:
            self._last_future.fetch()
        send_object(self._socket, object)

    def recv(self) -> Future:
        if self._last_future is not None:
            self._last_future.fetch()
        self._last_future = Channel.Future(self._socket)
        return self._last_future

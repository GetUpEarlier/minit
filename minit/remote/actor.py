import socket
import argparse
import importlib
from typing import Any, List, Union
from multiprocessing import Process

from .channel import Channel
from .registry import create_object, get_function, get_object
from .object import ObjectRef

class Actor:
    id: int

    def connect(self, address: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((address, port))
        self.channel = Channel(self.socket)
        self.id = self.channel.recv().get()
        print(f"serving as actor {self.id}")
        return self

    def act_forever(self):
        while True:
            [fn, *args] = self.channel.recv().get()
            print(f"acting {fn} at {self.id}")
            fn: str
            args: List[Union[ObjectRef, Any]]
            for arg in args:
                if isinstance(arg, ObjectRef):
                    assert arg.location == self.id, f"{arg.location} vs {self.id}"
            local_args = [
                get_object(arg.id) if isinstance(arg, ObjectRef) else arg for arg in args
            ]
            local_fn, is_constructor = get_function(fn)
            local_result = local_fn(*local_args)
            if is_constructor:
                result = ObjectRef(self.id, create_object(local_result))
            else:
                result = local_result
            self.channel.send(result)


def act_forever(address: str, port: int, module: str):
    importlib.__import__(module)
    actor = Actor()
    actor.connect(address, port)
    actor.act_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", "-p", type=int)
    parser.add_argument("--module", "-m", type=str)
    parser.add_argument("--size", "-n", type=int)
    args = parser.parse_args()
    processes: List[Process] = []
    try:
        for _i in range(args.size):
            process = Process(target=act_forever, args=(args.host, args.port, args.module))
            processes.append(process)
            process.start()
        while True:
            for process in processes:
                process.join(timeout=0.5)
            if all(process.exitcode is not None for process in processes):
                break
    finally:
        for process in processes:
            process.kill()


if __name__ == '__main__':
    main()

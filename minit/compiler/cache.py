import hashlib
import os
import subprocess
from typing import Dict, List
import filelock


def cached_execute(commands: List[str], files: Dict[str, str]) -> str:
    hash = hashlib.md5(str(commands).encode('utf-8'))
    hash.update(str(files).encode('utf-8'))
    md5_truncate = 8
    md5 = hash.hexdigest()[:md5_truncate]
    cached_path = os.path.join(os.path.expanduser("~"), ".minit", "cached", md5)
    lock_path = f"{cached_path}.filelock"
    with filelock.FileLock(lock_path):
        os.makedirs(cached_path, exist_ok=True)
        if os.path.exists(os.path.join(cached_path, ".success")):
            return cached_path
        print(f"compiling on {cached_path}")
        os.makedirs(cached_path, exist_ok=True)
        for name, content in files.items():
            with open(os.path.join(cached_path, name), "w+") as f:
                f.write(content)
        if len(commands) > 0:
            subprocess.check_call(commands, cwd=cached_path)
        open(os.path.join(cached_path, ".success"), 'w+').close()
        print(f"compiling completed")
        return cached_path

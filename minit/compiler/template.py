from typing import Dict


def substitude(source: str, args: Dict[str, str]):
    for k, v in args.items():
        source = source.replace(f"${{{k}}}", v)
    return source

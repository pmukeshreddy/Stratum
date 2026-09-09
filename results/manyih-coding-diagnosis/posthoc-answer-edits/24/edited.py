from typing import Any


def split_two_parts(list1: list[Any], L: int)->tuple[list[Any], list[Any]]:
    return list1[:L], list1[L:]

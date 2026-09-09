def flatten_list(list1: list[object])->list[object]:
    """Flatten nested lists while preserving element order."""
    result = []
    for value in list1:
        if isinstance(value, list):
            result.extend(flatten_list(value))
        else:
            result.append(value)
    return result

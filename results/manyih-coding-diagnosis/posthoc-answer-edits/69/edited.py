def sample_nam(sample_names: list[str]):
    """Sum lengths of names with a capital first letter."""
    return sum(
        len(name)
        for name in sample_names
        if name
        if name[0].isupper()
        if all(letter.islower() for letter in name[1:])
    )

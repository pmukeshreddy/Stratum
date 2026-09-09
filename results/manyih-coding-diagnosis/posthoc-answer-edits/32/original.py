def digit_distance_nums(n1: int, n2: int) -> int:
    n1, n2=abs(n1), abs(n2)
    distance=0
    while n1 or n2:
        distance+=abs(n1%10-n2%10)
        n1//=10
        n2//=10
    return distance

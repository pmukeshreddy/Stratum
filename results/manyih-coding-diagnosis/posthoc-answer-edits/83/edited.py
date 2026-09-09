def last(r, x):
    """Return the last zero-based index of x in sorted r, or negative one if absent."""
    l=0
    h=len(r)-1
    p=-1

    while l<=h:
        m=(l+h)//2
        if r[m]==x:
            p=m
            l=m+1
        elif r[m]<x:
            l=m+1
        else:
            h=m-1

    return p

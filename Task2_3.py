def Pow(x, N):
    if N == 0:
        return 1
    if N == 1:
        return x
    else:
        if N % 2 == 0:
            temp = Pow(x, N / 2)
            return temp*temp
        else:
            temp = Pow(x, (N - 1) / 2)
            return x * temp * temp


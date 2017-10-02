
def IntegerList(list_error):
    try:
        c = [c + 0 for c in list_error]
        print(list_error)
        max_length = 0
        curr_max_lentgh = 1
        if len(list_error) == 0:
            return 0

        prev = list_error[0]

        for elem in list_error:
            if not isinstance(elem, int):
                raise TypeError
            if elem > prev:
                curr_max_lentgh += 1
            else:
                curr_max_lentgh = 1
            if curr_max_lentgh > max_length:
                max_length = curr_max_lentgh
            prev = elem
        return max_length

    except TypeError:
        print("Not correct list. List must be integer")
        return 0


print(IntegerList([1, 2.0, 3, 0, 3, 1, 2, 3, 4]))

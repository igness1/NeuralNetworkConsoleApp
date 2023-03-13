
def unique(list):
    # Initialize empty list
    unique_list = []
    for x in list:
        # Checking if value exist in unique_list, if not then value is added.
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]
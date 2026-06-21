from collections import Counter


def l1_plus_l2(l1, l2, folded=True):
    if folded:
        return [x + y for x, y in zip(l1, l2)]
    else:
        counter = Counter(l1)  # Count occurrences of elements in l1
        for element in l2:
            if counter[-element] > 0:  # Check if the negation exists
                counter[-element] -= 1  # Remove one occurrence
                if counter[-element] == 0:  # If count reaches zero, delete key
                    del counter[-element]
            else:
                counter[element] += 1  # Otherwise, add the new element
        # Reconstruct the final list while preserving duplicates
        result = []
        for key, count in counter.items():
            result.extend([key] * count)
        return sorted(result)


def l1_minus_l2(l1, l2, folded=True):
    return l1_plus_l2(l1, [-1 * x for x in l2], folded=folded)


def vector_to_bits(lst):
    result = []
    for i, val in enumerate(lst):
        if val != 0:
            if i == 0:
                print("WARNING vector_bits 0 pb")
            result.extend([-i if val < 0 else i] * abs(val))
    return result


def bits_to_vector(bits, size=2048):
    vector = [0] * size
    for idx in bits:
        if idx >= 0:  # Ensure index is within bounds
            vector[idx] += 1
        else:
            vector[-1 * idx] -= 1
    return vector

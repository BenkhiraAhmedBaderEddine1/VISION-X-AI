
def count_raised_fingers(fingers):
    """
    Count raised fingers from five Boolean values:
    thumb, index, middle, ring, little.
    """
    if len(fingers) != 5:
        raise ValueError("Expected exactly five finger states.")

    return sum(bool(finger) for finger in fingers)

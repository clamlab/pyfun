"""
string manipulation
"""

def is_interval(s):
    """
    Check if a string represents an interval in the format "[lower, upper)", "(lower, upper]", etc.
    
    An interval string should start with either '[' or '(' and end with either ']' or ')'.
    The bounds should be numeric values separated by a comma.

    Parameters:
    s (str): The input string to check.

    Returns:
    bool: True if the string is a valid interval, False otherwise.
    """


    try:
        if s[0] not in '([' or s[-1] not in ')]':
            return False
        bounds = s[1:-1].split(',')
        if len(bounds) != 2:
            return False
        lower = float(bounds[0].strip())
        upper = float(bounds[1].strip())
        return True
    except (ValueError, IndexError):
        return False
class AddOnlyDict(dict):
    """
    A dictionary subclass that prevents modification of existing keys.

    ReadOnlyDict is a dictionary where you can only add new keys, but cannot
    modify the values of existing keys. If you try to set a value for an
    existing key, a KeyError will be raised.

    Usage:
    ------
    my_dict = AddOnlyDict()
    my_dict["key1"] = "value1"  # Adds a new key-value pair
    my_dict["key2"] = "value2"  # Adds another new key-value pair
    my_dict["key1"] = "new_value1"  # Raises KeyError, because the key already exists

    Example:
    --------
    >>> d = ReadOnlyDict()
    >>> d["a"] = 1
    >>> d["a"] = 2
    KeyError: "Key 'a' already exists and cannot be modified."
    """

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Key '{key}' already exists and cannot be modified.")
        super().__setitem__(key, value)


def display_dict(dictionary, indent=0):
    """
    Display a dictionary in a neat, readable format with arbitrary depth and data types.

    This function takes a dictionary as input and prints its keys and values
    in a nested, indented format for better readability.

    Parameters
    ----------
    dictionary : dict
        A dictionary to display.
    indent : int, optional
        The indentation level (default is 0).
    """
    for key, value in dictionary.items():
        print(" " * indent + f"{key}:", end=" ")
        if isinstance(value, dict):
            print()
            display_dict(value, indent + 2)
        else:
            print(value)







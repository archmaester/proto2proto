import os


def create_dirs(path):
    """
    Create the directory structure specified in path

    Parameters
    ----------
    path : str
        A string of the entire path of which the directory
        structure is to be created.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return


def check_path(path):
    """
    Check if the path exists

    Parameters
    ----------
    path : str
        A string of the entire path of which
        the directory structure is to be created.

    Returns
    -------
    True if path exists
    False if path doesn't exist
    """
    if os.path.exists(path):
        return True
    return False

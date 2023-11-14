import os

def find_files(directory, filetypes):
    """
    Search for files in a directory with specified filetypes.

    :param directory: Path to the directory to search in.
    :param filetypes: List of file extensions to search for.
    :return: List of paths to found files.
    """
    matched_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ft) for ft in filetypes):
                matched_files.append(os.path.join(root, file))
    return matched_files


def unique_filename(directory, filename):
    """
    Generate a unique filename by appending a number if a file with the same name exists.

    :param directory: Target directory to save the file.
    :param filename: Desired filename.
    :return: A unique filename as a string.
    """
    base, extension = os.path.splitext(filename)
    counter = 1
    unique_name = filename
    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{base}_{counter}{extension}"
        counter += 1
    return unique_name

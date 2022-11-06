import os 
from glob import glob

def get_files_with_extension(file_path_with_extension):
    """
    Get all files with given extension in file path
    Args:
        file_path_with_extension: File path with extension
    Returns: 
        files: list of all files with extension
    """
    return glob(file_path_with_extension)
    
def get_folder(file):
    """
    Get folder from file path
    Args:
        file: File path
    Returns: 
        dir: Folder path
    """
    dir = os.path.dirname(file)
    return dir

def get_file_name(file):
    """
    Get basename without extension from file path
    Args:
        file: file path
    Returns: 
        base_name: Base Name with extension
        base_name: BaseName without extension
    """
    base_name = os.path.basename(file)
    base_name_without_extension = os.path.splitext(base_name)[0]
    return base_name, base_name_without_extension
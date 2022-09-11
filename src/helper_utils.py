import os 
from glob import glob

def get_files_with_extension(file_path_with_extension):
    return glob(file_path_with_extension)
    
def get_folder(file):
    dir = os.path.dirname(file)
    return dir

def get_file_name(file):
    base_name = os.path.basename(file)
    base_name_without_extension = os.path.splitext(base_name)[0]
    return base_name, base_name_without_extension
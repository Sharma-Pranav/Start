
import sys 
sys.path.append("..\\..\\src")
import os 
from helper_utils import get_files_with_extension,get_folder, get_file_name
import pandas as pd
path = os.path.join(os.getcwd(), 'data', '*', '*')
path_to_files = get_files_with_extension(path)

key_path_dictionary = {'path':[], 'folder':[],'label':[]}
label_dict = {'Coffeepot':0 , 'Dog':1, 'Elephant':2, 'Peacock':3}
for path in path_to_files:

    folder_base_name = os.path.basename(get_folder(path))
    key_path_dictionary['folder'].append(folder_base_name)
    key_path_dictionary['path'].append(path)
    key_path_dictionary['label'].append(label_dict[folder_base_name])
    
df = pd.DataFrame(key_path_dictionary)
df.to_csv('df.csv')
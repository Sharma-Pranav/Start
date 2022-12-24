import sys
sys.path.append("..\\..\\..\\src")
import os 
import pandas as pd
from helper_utils import get_folder, get_file_name, get_files_with_extension

pwd = os.getcwd()
rgb_files = get_files_with_extension(os.path.join('..', 'ISBI2016_ISIC_Part3B_Training_Data', '*.jpg'))

mapping_dict = {}
for file in rgb_files:
    dir = get_folder(file)
    _, base_file_name = get_file_name(file)
    new_name = os.path.join(dir, base_file_name + '_Segmentation.png')    
    mapping_dict[file] = new_name

df = pd.DataFrame(mapping_dict.items(), columns=['path', 'label'])
print(df.head())  
df.to_csv('mapping_segmentation_labels.csv')
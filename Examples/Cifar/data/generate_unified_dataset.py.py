from glob import glob
from unicodedata import name
import numpy as np
import os
import pickle
import cv2
import pandas as pd

image_shape = (3,32,32)
train_paths = glob(os.path.join(os.getcwd(), 'Data*'))
test_paths = glob(os.path.join(os.getcwd(), 'Test_*'))
batch_meta = glob(os.path.join(os.getcwd(), '*.meta'))

encode_byte_str_to_str = lambda x: x.decode('utf-8')
        
def unpickle(file): 
    '''
    Unpickles given file appropriately as dictionary
    Args: 
        file : path to files
    Returns:
        diction:  A dictionary containing the contents of the dictionary as is.
    '''
    with open(file, 'rb') as fo:
        diction = pickle.load(fo, encoding='bytes')
    return diction

def write_files_to_folder(consolidated_data_diction, folder):
    '''
    Writes content of data key of dictionray to conrresponding image file
    Args: 
        consolidated_data_diction : Filled dictionary of content
        folder : path of the dumping folder
    Returns:
        filepaths:  list of path to generated files
    '''

    path = os.path.join(os.getcwd(), folder )
    if not os.path.exists(path):
        os.mkdir(path)
    file_paths = []
    for i, file in enumerate(consolidated_data_diction['filenames']):
        img = consolidated_data_diction['data'][i]
        file_path = os.path.join(path, file)
        file_paths.append(file_path)
        cv2.imwrite(file_path, img)
    return file_paths

def get_dicts(paths):
    '''
    Get dictionaries from the source dictionaries  of pickled files
    Args: 
        paths : paths to pickled files
    Returns:
        consolidated_data_diction:  dictionary file containing the keys in pickled files
        key_to_str_key_dict: dictionary mapping encoded byte data to string
        str_to_byte_dict: dictionary mapping stra to encoded byte data
    '''
    for filepath in paths:
        data_dict = unpickle(filepath)
        keys = list(data_dict.keys())
        key_to_str_key_dict = {key:encode_byte_str_to_str(key) for key in keys}
        str_to_byte_dict = {key_to_str_key_dict[key]: key for key in key_to_str_key_dict.keys()}
        consolidated_data_diction = {key : [] for key in str_to_byte_dict.keys()}
        consolidated_data_diction['label_names'] = []
    return consolidated_data_diction, key_to_str_key_dict, str_to_byte_dict
    
def get_filled_consolidated_data_dictionary(consolidated_data_diction, key_to_str_key_dict, paths):
    '''
    Fill consolidated data dictionary with data from pickled files 
    Args: 
        consolidated_data_diction : Data dictionary for filling pikled data
        key_to_str_key_dict: dictionary containing mapping of keys of pickled files to appropriate strings
    Returns:
        consolidated_data_diction:  dictionary file containing the data of all given pickled files in path
    '''
    for filepath in paths:
        data_dict = unpickle(filepath)
        for key in data_dict.keys():
            data_elem = data_dict[key]
            consolidated_data_diction[key_to_str_key_dict[key]].append(data_elem)
    return consolidated_data_diction

def preprocess_consolidated_dictionary(consolidated_data_diction, image_shape):
    '''
    Preprocess the data apppropriately inside the consolidated data dictionary 
    Args: 
        consolidated_data_diction : Data dictionary for preprocessing data
        image_shape: shape of the desired image
    Returns:
        consolidated_data_diction:  dictionary file containing the preprocessed data
    '''

    for key in consolidated_data_diction.keys():
        if key =='data':
            np_elem = np.vstack(consolidated_data_diction[key])
            np_elem = np.reshape(np_elem, (np_elem.shape[0], *image_shape))
            np_elem = np_elem.transpose(0,2,3,1)
            consolidated_data_diction[key] = np_elem
        else:
            data_list = []
            for elem in consolidated_data_diction[key] :
                data_list.extend(elem)
            if key == 'filenames':
                data_list = [encode_byte_str_to_str(data) for data in data_list]
            consolidated_data_diction[key] = np.array(data_list)
    return consolidated_data_diction

def create_data_and_csv(folder, paths):
    '''
    Create data from given paths and generates corresponding csvs
    Args: 
        folder: folder where the generated files will be dumped
        paths : paths to given pixel file
        image_shape: shape of the desired image
    Returns:
        df: dataframe for created files with label and path columns
    '''
    consolidated_data_diction, key_to_str_key_dict, _ = get_dicts(test_paths)

    get_filled_consolidated_data_dictionary(consolidated_data_diction, key_to_str_key_dict, paths)
    
    consolidated_data_diction = preprocess_consolidated_dictionary(consolidated_data_diction, image_shape)
    
    consolidated_data_diction['label_names'] = [name_mapping_dict[data_] for data_ in consolidated_data_diction['labels']]
    
    file_paths = write_files_to_folder(consolidated_data_diction, folder)
    
    consolidated_data_diction['path'] = file_paths
    consolidated_data_diction['label'] = consolidated_data_diction['labels']
    
    del consolidated_data_diction['labels']
    del consolidated_data_diction['data']
    del consolidated_data_diction['batch_label']
    df = pd.DataFrame(consolidated_data_diction)
    df.to_csv(os.path.join(os.getcwd(), folder + '.csv'))
    return df

name_dict = unpickle(batch_meta[0])
for key in name_dict:
    if 'label_names'  == encode_byte_str_to_str(key):
        data = name_dict[key]
        name_mapping_dict = {i: encode_byte_str_to_str(elem) for i, elem in enumerate(data)}

create_data_and_csv('train', train_paths)
create_data_and_csv('test', test_paths)
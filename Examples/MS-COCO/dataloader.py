from helper_utils import get_cat_infos, show_image_and_annotations
import numpy as np
required_ids = ['person', 'vehicle', 'outdoor', 'animal']



def get_id_mapping_dictionary():
    (coco, total_catIDs, total_cats) = get_cat_infos()
    super_key = 'supercategory'
    id_info = []
    id_name_info = []
    for cat in total_cats:
        if cat[super_key] in required_ids:
            id_info.append(cat['id'])
            id_name_info.append(cat['name'])

    id_dictionary = {}
    id_dictionary['id'] = id_info
    id_dictionary['name'] = id_name_info
    return coco, total_catIDs, total_cats, id_dictionary


coco, total_catIDs, total_cats, id_dictionary = get_id_mapping_dictionary()

#print(total_catIDs)


catIDs = id_dictionary['id']
imgIds = coco.getImgIds(catIds=catIDs)
print('imgIds : ', imgIds)
print('catIDs : ', catIDs)
index = np.random.randint(0,len(imgIds))
#show_image_and_annotations(coco, imgIds, catIds = [] , index = index, show_annotations=True)
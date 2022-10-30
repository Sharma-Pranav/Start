# Helper Utils
#https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047

from __future__ import annotations
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


# Initialize the COCO api for instance annotations

dataDir='./COCOdataset2017' 
dataType='val'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


def get_cat_infos(annFile=annFile):
    # Load the categories in a variable
    coco=COCO(annFile)
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    return (coco, catIDs, cats)

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"


def show_image_and_annotations(coco, imgIds, catIds = [] , index = 0, show_annotations=True):#np.random.randint(0,len(imgIds))):
    img = coco.loadImgs(imgIds[index])[0]
    I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0

    # Load and display instance annotations
    plt.imshow(I)
    plt.axis('off')
    if show_annotations:
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
    plt.show()

def get_filtered_common_files(coco, filterClasses = ['laptop', 'tv', 'cell phone']):
    # Define the classes (out of the 81) which you want to see. Others will not be shown.
    # Fetch class IDs only corresponding to the filterClasses
    catIds = coco.getCatIds(catNms=filterClasses) 
    # Get all images containing the above Category IDs
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of images containing all the  classes:", len(imgIds))
    return (catIds, imgIds)

def get_all_files_in_given_categories(coco, classes):
    cat_Id_list = []
    image_info_list = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given class
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            cat_Id_list.append(catIds)
            image_info_list += coco.loadImgs(imgIds)
    else:
        cat_Id_list = coco.getCatIds()
        imgIds = coco.getImgIds()
        image_info_list = coco.loadImgs(imgIds)
        
    # Now, filter out the repeated images    
    unique_images_info_list = []
    for elem , i in range(len(image_info_list)):
        if elem not in unique_images_info_list:
            unique_images_info_list.append(elem)

    dataset_size = len(unique_images_info_list)

    print("Number of images containing the filter classes:", dataset_size)

    return cat_Id_list, unique_images_info_list, dataset_size


def generate_segmentation_mask(coco, filterClasses, imgIds, catIds, index, individual_classes =True, show_mask = True ):
    img = coco.loadImgs(imgIds[index])
    cats = coco.loadCats(catIds)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = np.zeros((img['height'],img['width']))
    for i in range(len(anns)):
        if individual_classes:
            className = getClassName(anns[i]['category_id'], cats)
            pixel_value = filterClasses.index(className)+1
            mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
        else:
            mask = np.maximum(coco.annToMask(anns[i]), mask)

    if show_mask:
        plt.imshow(mask)
        print('Non Binary Mask: ', mask.shape)

    return mask


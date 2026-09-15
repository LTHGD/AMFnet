import xml.etree.ElementTree as ET
from os import getcwd

sets=[('train'), ('val'), ('test')]

classes = ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"]


def convert_annotation(image_id, list_file):
    in_file = open('data_pre-processing/Annotations/%s.xml'%(image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

Dat='PCB'

for image_set in sets:
    image_ids = open('data_pre-processing/Train_test_val_split/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s/data_pre-processing/PCB_Train/%s_%s.txt'%(wd,Dat,image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/data_pre-processing/Image/%s.jpg'%(wd, image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()

# -*- coding: utf-8 -*-
import json
import os
import shutil

root_path = os.getcwd()
root_path = os.path.join(root_path,'datasets/BoltHead')

def voc2coco():
    import datetime
    from PIL import Image

    # 处理coco数据集中category字段。
    # 创建一个 {类名 : id} 的字典，并保存到 总标签data 字典中。
    class_name_to_id = {'BoltHead_1': 0,'BoltHead_2':1 ,'lost_BoltHead':3}
    # class_name_to_id = {'BoltNut':0, 'BoltHead':1, 'LockingWire':2, 'Coupling':3, 
    #                     'Putty':4, 'BrakePad':5, 'TailCotterPin':6, 'BrakeClamp':7, 'Axle':8,
    #                       'RadiatingRib':9, 'SingleLockingWire':10, 'Mirror':11, 'OilLevelMirror':12, 
    #                       'OilPlugB':13, 'CotterPin':14, 'MagneticBoltHolder':15, 'OilPlugS':16,
    #                         'Nameplate':17, 'Sander':18, 'Nozzle':19, 'WholeCotterPin':20, 'Rubber':21, 
    #                         'WheelTread':22, 'BrakeCylinder':23, 'Boot':24, 'TractionRod':25, 
    #                         'LockSpring':26, 'deformation':27, 'abnormal_Tape':28,
    #                           'RadiatingRid':29, 'crack_LockingWire':30}

    # 创建coco的文件夹
    if not os.path.exists(os.path.join(root_path, "coco2017")):
        os.makedirs(os.path.join(root_path, "coco2017"))
        os.makedirs(os.path.join(root_path, "coco2017", "annotations"))
        os.makedirs(os.path.join(root_path, "coco2017", "train2017"))
        os.makedirs(os.path.join(root_path, "coco2017", "val2017"))

    # 创建 总标签data
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, file_name,url, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    for name, id in class_name_to_id.items():
        data["categories"].append(
            dict(supercategory=None, id=id, name=name, )
        )

    # 处理coco数据集train中images字段。
    images_dir = os.path.join(root_path, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    images = os.listdir(images_dir)

    # 生成每个图片对应的image_id
    images_id = {}
    for idx, image_name in enumerate(images):
        images_id.update({image_name[:-4]: idx})

    # 获取训练图片
    train_img = []
    fp = open(os.path.join(root_path, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'train.txt'))
    for i in fp.readlines():
        train_img.append(i[:-1] + ".jpg")

    # 获取训练图片的数据
    for image in train_img:
        img = Image.open(os.path.join(images_dir, image))
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=image,  # 图片的文件名带后缀
                height=img.height,
                width=img.width,
                date_captured=None,
                # id=image[:-4],
                id=images_id[image[:-4]],
            )
        )

    # 获取coco数据集train中annotations字段。
    train_xml = [i[:-4] + '.xml' for i in train_img]

    bbox_id = 0
    for xml in train_xml:
        category = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        import xml.etree.ElementTree as ET
        tree = ET.parse(os.path.join(root_path, 'VOCdevkit', 'VOC2007', 'Annotations', xml))
        root = tree.getroot()
        object = root.findall('object')
        for i in object:
            category.append(class_name_to_id[i.findall('name')[0].text])
            bndbox = i.findall('bndbox')
            for j in bndbox:
                xmin.append(float(j.findall('xmin')[0].text))
                ymin.append(float(j.findall('ymin')[0].text))
                xmax.append(float(j.findall('xmax')[0].text))
                ymax.append(float(j.findall('ymax')[0].text))
        for i in range(len(category)):
            data["annotations"].append(
                dict(
                    id=bbox_id,
                    image_id=images_id[xml[:-4]],
                    category_id=category[i],
                    area=(xmax[i] - xmin[i]) * (ymax[i] - ymin[i]),
                    bbox=[xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i]],
                    iscrowd=0,
                )
            )
            bbox_id += 1
    # 生成训练集的json
    json.dump(data, open(os.path.join(root_path, 'coco2017', 'annotations', 'instances_train2017.json'), 'w'))

    # 获取验证图片
    val_img = []
    fp = open(os.path.join(root_path, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'val.txt'))
    for i in fp.readlines():
        val_img.append(i[:-1] + ".jpg")

    # 将训练的images和annotations清空，
    del data['images']
    data['images'] = []
    del data['annotations']
    data['annotations'] = []

    # 获取验证集图片的数据
    for image in val_img:
        img = Image.open(os.path.join(images_dir, image))
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=image,  # 图片的文件名带后缀
                height=img.height,
                width=img.width,
                date_captured=None,
                id=images_id[image[:-4]],
            )
        )

    # 处理coco数据集验证集中annotations字段。
    val_xml = [i[:-4] + '.xml' for i in val_img]

    for xml in val_xml:
        category = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        import xml.etree.ElementTree as ET
        tree = ET.parse(os.path.join(root_path, 'VOCdevkit', 'VOC2007', 'Annotations', xml))
        root = tree.getroot()
        object = root.findall('object')
        for i in object:
            category.append(class_name_to_id[i.findall('name')[0].text])
            bndbox = i.findall('bndbox')
            for j in bndbox:
                xmin.append(float(j.findall('xmin')[0].text))
                ymin.append(float(j.findall('ymin')[0].text))
                xmax.append(float(j.findall('xmax')[0].text))
                ymax.append(float(j.findall('ymax')[0].text))
        for i in range(len(category)):
            data["annotations"].append(
                dict(
                    id=bbox_id,
                    image_id=images_id[xml[:-4]],
                    category_id=category[i],
                    area=(xmax[i] - xmin[i]) * (ymax[i] - ymin[i]),
                    bbox=[xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i]],
                    iscrowd=0,
                )
            )
            bbox_id += 1
    # 生成验证集的json
    json.dump(data, open(os.path.join(root_path, 'coco2017', 'annotations', 'instances_val2017.json'), 'w'))
    print('| VOC -> COCO annotations transform finish.')
    print('Start copy images...')

    for img_name in train_img:
        shutil.copy(os.path.join(root_path, "VOCdevkit", "VOC2007", "JPEGImages", img_name),
                    os.path.join(root_path, "coco2017", 'train2017', img_name))
    print('| Train images copy finish.')

    for img_name in val_img:
        shutil.copy(os.path.join(root_path, "VOCdevkit", "VOC2007", "JPEGImages", img_name),
                    os.path.join(root_path, "coco2017", 'val2017', img_name))
    print('| Val images copy finish.')


if __name__ == '__main__':
    print("—" * 50)
    voc2coco()  # voc数据集转换成coco数据集
    print("—" * 50)

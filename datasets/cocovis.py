import json
import os
import random

import cv2

root_path = os.getcwd()
root_path = __file__[:-10]
SAMPLE_NUMBER = 30  # 随机挑选多少个图片检查，
class_name_to_id = {'BoltNut':0, 'BoltHead':1, 'LockingWire':2, 'Coupling':3, 
                        'Putty':4, 'BrakePad':5, 'TailCotterPin':6, 'BrakeClamp':7, 'Axle':8,
                          'RadiatingRib':9, 'SingleLockingWire':10, 'Mirror':11, 'OilLevelMirror':12, 
                          'OilPlugB':13, 'CotterPin':14, 'MagneticBoltHolder':15, 'OilPlugS':16,
                            'Nameplate':17, 'Sander':18, 'Nozzle':19, 'WholeCotterPin':20, 'Rubber':21, 
                            'WheelTread':22, 'BrakeCylinder':23, 'Boot':24, 'TractionRod':25, 
                            'LockSpring':26, 'deformation':27, 'abnormal_Tape':28,
                              'RadiatingRid':29, 'crack_LockingWire':30}

def visiual():
    # 获取bboxes
    json_file = os.path.join(root_path, 'coco2017', 'annotations', 'instances_train2017.json')  # 如果想查看验证集，就改这里
    data = json.load(open(json_file, 'r'))
    images = data['images']  # json中的image列表，

    # 读取图片
    for i in random.sample(images, SAMPLE_NUMBER):  # 随机挑选SAMPLE_NUMBER个检测
        # for i in images:                                        # 整个数据集检查
        img = cv2.imread(os.path.join(root_path, 'coco2017', 'train2017',
                                      i['file_name']))  # 改成验证集的话，这里的图片目录也需要改,train2017 -> val2017
        bboxes = []  # 获取每个图片的bboxes
        category_ids = []
        annotations = data['annotations']
        for j in annotations:
            if j['image_id'] == i['id']:
                bboxes.append(j["bbox"])
                category_ids.append(j['category_id'])   
        id_category = {}
        for  key,value in dict.items(class_name_to_id):
            id_category[value] = key
            
        # 生成锚框
        for idx, bbox in enumerate(bboxes):
            left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是，左上角坐标和右下角坐标。
            right_bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 1)  # 图像，左上角，右下坐标，颜色，粗细
            cv2.putText(img, id_category[category_ids[idx]], left_top, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            # 画出每个bbox的类别，参数分别是：图片，类别名(str)，坐标，字体，大小，颜色，粗细
        cv2.imshow('image', img)                                          # 展示图片，
        cv2.waitKey(1000)
        cv2.imwrite(os.path.join('visiual', i['file_name']), img)  # 或者是保存图片
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    print('—' * 50)
    save_path =os.path.abspath (__file__)[:-10]+"visual/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    visiual()
    print('| visiual completed.')
    print('| saved as ', os.path.join(save_path))
    print('—' * 50)

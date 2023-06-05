# 命令行执行：  python xml2voc2007.py --input_dir data --output_dir VOCdevkit
import argparse
import glob
import os
import os.path as osp
import random
import shutil
import sys

percent_train = 0.9  # 改成你想设置的训练集比例。


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default="data",
                        help="input annotated directory")  # 将保存你.jpg和.xml文件的文件夹名改为data，下边就不用动了
    parser.add_argument("--output_dir", default="VOCdevkit", help="output dataset directory")  # 输出的voc数据集目录，不用动
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    print("| Creating dataset dir:", osp.join(args.output_dir, "VOC2007"))

    # 创建保存的文件夹
    if not os.path.exists(osp.join(args.output_dir, "VOC2007", "Annotations")):
        os.makedirs(osp.join(args.output_dir, "VOC2007", "Annotations"))
    if not os.path.exists(osp.join(args.output_dir, "VOC2007", "ImageSets")):
        os.makedirs(osp.join(args.output_dir, "VOC2007", "ImageSets"))
    if not os.path.exists(osp.join(args.output_dir, "VOC2007", "ImageSets", "Main")):
        os.makedirs(osp.join(args.output_dir, "VOC2007", "ImageSets", "Main"))
    if not os.path.exists(osp.join(args.output_dir, "VOC2007", "JPEGImages")):
        os.makedirs(osp.join(args.output_dir, "VOC2007", "JPEGImages"))

    # 获取目录下所有的.jpg文件列表
    total_img = glob.glob(osp.join(args.input_dir, "*.jpg"))
    print('| Image number: ', len(total_img))

    # 获取目录下所有的joson文件列表
    total_xml = glob.glob(osp.join(args.input_dir, "*.xml"))
    print('| Xml number: ', len(total_xml))

    num_total = len(total_xml)
    data_list = range(num_total)

    num_tr = int(num_total * percent_train)
    num_train = random.sample(data_list, num_tr)

    print('| Train number: ', num_tr)
    print('| Val number: ', num_total - num_tr)

    file_train = open(
        osp.join(args.output_dir, "VOC2007", "ImageSets", "Main", "train.txt"), 'w')
    file_val = open(
        osp.join(args.output_dir, "VOC2007", "ImageSets", "Main", "val.txt"), 'w')

    for i in data_list:
        name = total_xml[i][:-4] + '\n'  # 去掉后缀'.jpg' 
        name = os.path.split(name)[1]
        if i in num_train:
            file_train.write(name)  # 因为这里的name是带着目录的，也就是name本来是：'data/1.jpg' ，去掉'data/' ，就是文件名了。
        else:
            file_val.write(name)

    file_train.close()
    file_val.close()

    if os.path.exists(args.input_dir):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                src_file = osp.join(root, file)
                if src_file.endswith(".jpg"):
                    shutil.copy(src_file, osp.join(args.output_dir, "VOC2007", "JPEGImages"))
                else:
                    shutil.copy(src_file, osp.join(args.output_dir, "VOC2007", "Annotations"))
    print('| Done!')

if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)

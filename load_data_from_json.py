import json
import cv2, os, random


def convert_bbox_to_cbox(bbox, width, height):
    return [(bbox[0]+bbox[2])/2/width, (bbox[1]+bbox[3])/2/height, (bbox[2]-bbox[0])/width, (bbox[3]-bbox[1])/height]

def get_data_lists(imgroot, jsonfile, key=None):
    f = open(jsonfile, "r")
    labelZIP = []
    for line in f.readlines():
        data = json.loads(line)
        imgname = data["image_key"]
        objs = []
        if "person" in data.keys():
            for perper in data["person"]:
                if perper["attrs"]["ignore"]=="no":
                    rect = perper["data"]
                    rect = convert_bbox_to_cbox(rect, data["width"], data["height"])
                    objs.append(rect)
            if len(objs) != 0:
                labelZIP.append([os.path.join(imgroot, imgname), objs])
    return labelZIP

def gen_BIGbatch_data(fileroot, batchsize):
    filelist= ["13386_3", "13390_3"]
    zipzip = []
    for file in filelist:
        jsonfile = os.path.join(fileroot, file, "data_000001.json")
        imgroot = os.path.join(fileroot, file, "data_000001")
        labelZip = get_data_lists(imgroot, jsonfile)
        zipzip.append(labelZip)
    return zipzip

def write_to_keras_txt(datas, txtname):
    txt = open("./kersa_txt_person.txt", "w+")
    for data in datas:
        for imgpath, bboxes in data:
            txt.write(imgpath)
            for bbox in bboxes:
                txt.write(" ")
                txt.write(str(bbox[0]))
                txt.write(",")
                txt.write(str(bbox[1]))
                txt.write(",")
                txt.write(str(bbox[2]))
                txt.write(",")
                txt.write(str(bbox[3]))
                txt.write(",")
                txt.write("0")
            txt.write("\n")
    txt.close()


import numpy as np

def get_dict(datafile):
    data = dict()
    f = open(datafile,"r")
    for line in f.readlines():
        key, value = line.strip().split("=")
        data[key]=value
    f.close()
    return data

def get_train_imgpaths(data):
    txtpath = data["train"]
    imgpaths=[]
    f = open(txtpath, "r")
    while True:
        line = f.readline().strip()
        if len(line)==0:
            break
        imgpaths.append(line)
    f.close()
    return imgpaths

def py_nms(dets, scores, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def convert_cbox_to_bbox(cbox, width, height):
    '''
    convert yolo box  to  xml box
    :param cbox: yolo_box n*4
    :param width: img`s width
    :param height: img`s height
    :return: xml_box n*4
    '''
    w = cbox[:, 2]*width
    h = cbox[:, 3]*height
    x1 = cbox[:, 0]*width - 0.5 * w
    x1 = np.where(x1<0, 0, x1)
    y1 = cbox[:, 1]*height - 0.5 * h
    y1 = np.where(y1<0, 0, y1)
    x2 = cbox[:, 0]*width + 0.5 * w
    x2 = np.where(x2>width, width-1, x2)
    y2 = cbox[:, 1]*height + 0.5 * h
    y2 = np.where(y2>height, height-1, y2)
    bbox = np.array([x1, y1, x2, y2])
    return bbox.T

def softmax(x):
    a = np.sum(np.exp(x), -1)
    b = np.stack([np.exp(x[..., i])/a for i in range(x.shape[-1])], -1)
    return b

def sigmod(x):
    return 1/(1+np.exp(-x))

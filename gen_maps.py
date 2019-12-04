import numpy as np
from config import config as cfg

def cbox_anchors_iou(cbox, anchors):
    # cbox = tf.cast(cbox, dtype=tf.float32)
    box_area = cbox[2] * cbox[3]
    # anchors = tf.cast(anchors, tf.float32)
    areas = anchors[:, 0] / cfg.imagesize * anchors[:, 1] / cfg.imagesize

    xx1 = np.maximum(0 - 0.5 * cbox[2], 0 - 0.5 * anchors[..., 0] / cfg.imagesize)
    yy1 = np.maximum(0 - 0.5 * cbox[3], 0 - 0.5 * anchors[..., 1] / cfg.imagesize)
    xx2 = np.minimum(0 + 0.5 * cbox[2], 0 + 0.5 * anchors[..., 0] / cfg.imagesize)
    yy2 = np.minimum(0 + 0.5 * cbox[3], 0 + 0.5 * anchors[..., 1] / cfg.imagesize)

    # compute the width and height of the bounding box
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + areas - inter)
    return ovr  # 3


def convert_cboxes_to_maps(batches_cboxes, anchors):
    batches_maps_13, batches_maps_26 = [], []
    for batch_cboxes in batches_cboxes:
        batch_map_13 = np.zeros((13, 13, 3, 5+cfg.classes), np.float32)
        batch_map_26 = np.zeros((26, 26, 3, 5+cfg.classes), np.float32)
        for cbox in batch_cboxes:
            cbox = np.array(cbox, np.float32)
            iou = cbox_anchors_iou(cbox, anchors)
            n = np.argmax(iou)
            if n>=3:
                x13 = int(cbox[0] * 13)
                y13 = int(cbox[1] * 13)
                batch_map_13[y13, x13, n-3, 0:4] = cbox
                batch_map_13[y13, x13, n-3, 4] = 1
                batch_map_13[y13, x13, n-3, 5] = 1
            else:
                x26 = int(cbox[0] * 26)
                y26 = int(cbox[1] * 26)
                batch_map_26[y26, x26, n, 0:4] = cbox
                batch_map_26[y26, x26, n, 4] = 1
                batch_map_26[y26, x26, n, 5] = 1
        batches_maps_13.append(batch_map_13)
        batches_maps_26.append(batch_map_26)
    return np.stack(batches_maps_13, 0), np.stack(batches_maps_26, 0)



# def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
#     '''Preprocess true boxes to training input format
#
#     Parameters
#     ----------
#     true_boxes: array, shape=(m, T, 5)
#         Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
#     input_shape: array-like, hw, multiples of 32
#     anchors: array, shape=(N, 2), wh
#     num_classes: integer
#
#     Returns
#     -------
#     y_true: list of array, shape like yolo_outputs, xywh are reletive value
#
#     '''
#     # assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
#     num_layers = len(anchors)//3 # default setting
#     anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
#
#     true_boxes = np.array(true_boxes, dtype='float32')
#     input_shape = np.array(input_shape, dtype='int32')
#     boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
#     boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
#     true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
#     true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]
#
#     m = true_boxes.shape[0]
#     grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
#     y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
#         dtype='float32') for l in range(num_layers)]
#
#     # Expand dim to apply broadcasting.
#     anchors = np.expand_dims(anchors, 0)
#     anchor_maxes = anchors / 2.
#     anchor_mins = -anchor_maxes
#     valid_mask = boxes_wh[..., 0]>0
#
#     for b in range(m):
#         # Discard zero rows.
#         wh = boxes_wh[b, valid_mask[b]]
#         if len(wh)==0: continue
#         # Expand dim to apply broadcasting.
#         wh = np.expand_dims(wh, -2)
#         box_maxes = wh / 2.
#         box_mins = -box_maxes
#
#         intersect_mins = np.maximum(box_mins, anchor_mins)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         box_area = wh[..., 0] * wh[..., 1]
#         anchor_area = anchors[..., 0] * anchors[..., 1]
#         iou = intersect_area / (box_area + anchor_area - intersect_area)
#
#         # Find best anchor for each true box
#         best_anchor = np.argmax(iou, axis=-1)
#
#         for t, n in enumerate(best_anchor):
#             for l in range(num_layers):
#                 if n in anchor_mask[l]:
#                     i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
#                     j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
#                     k = anchor_mask[l].index(n)
#                     c = true_boxes[b,t, 4].astype('int32')
#                     y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
#                     y_true[l][b, j, i, k, 4] = 1
#                     y_true[l][b, j, i, k, 5+c] = 1
#
#     return y_true
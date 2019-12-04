import tensorflow as tf
from config import config as cfg
import numpy as np


def active_map(pred_per_batch, anchors):
    b, h, w, n, _ = pred_per_batch.get_shape().as_list()
    # pred_per_batch = tf.reshape(pred_per_batch, (w, h, anchors.shape[0], -1))
    mask_x = np.tile(np.arange(0, w, 1), h).reshape([1, h, w])
    mask_y = np.tile(np.arange(0, h, 1), w).reshape([w, h]).T.reshape([1, h, w])
    pred_per_batch_x = tf.stack(
        [(tf.nn.sigmoid(pred_per_batch[..., i, 0]) + mask_x) / w for i in range(anchors.shape[0])], -1)
    pred_per_batch_y = tf.stack(
        [(tf.nn.sigmoid(pred_per_batch[..., i, 1]) + mask_y) / h for i in range(anchors.shape[0])], -1)
    pred_per_batch_w = tf.stack(
        [tf.exp(pred_per_batch[..., i, 2]) * anchors[i][0] / cfg.imagesize for i in range(anchors.shape[0])], -1)
    pred_per_batch_h = tf.stack(
        [tf.exp(pred_per_batch[..., i, 3]) * anchors[i][1] / cfg.imagesize for i in range(anchors.shape[0])], -1)
    A = tf.stack([pred_per_batch_x, pred_per_batch_y, pred_per_batch_w, pred_per_batch_h], -1)
    objs = tf.expand_dims(tf.nn.sigmoid(pred_per_batch[..., 4]), -1)
    cls = tf.nn.softmax(pred_per_batch[..., 5::], -1)
    return tf.concat((A, objs, cls), -1)

def cals_box_loss(raw_box, turth_box, w, h, anchors, obj_mask):
    '''

    :param raw_box: b, h, w, n, 4
    :param turth_box: b,h,w,n,4
    :param w:
    :param h:
    :param anchors: n, 2
    :return:
    '''
    obj_mask = tf.cast(obj_mask, tf.bool)
    mask_x = np.tile(np.arange(0, w, 1), h).reshape([1, h, w, 1])
    mask_y = np.tile(np.arange(0, h, 1), w).reshape([w, h]).T.reshape([1, h, w, 1])
    box_scale = tf.expand_dims(2- turth_box[..., 2]*turth_box[..., 3], -1)
    tx = turth_box[..., 0]*w-mask_x
    ty = turth_box[..., 1]*h-mask_y
    anchorlist = tf.cast(tf.reshape(anchors, [1,1,1,3,2]), tf.float16)
    tw = tf.log(turth_box[..., 2] / anchorlist[..., 0] * cfg.imagesize)
    tw = tf.where(obj_mask, tw, tf.zeros_like(tw))
    th = tf.log(turth_box[..., 3] / anchorlist[..., 1] * cfg.imagesize)
    th = tf.where(obj_mask, th, tf.zeros_like(th))
    x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tx, logits=raw_box[..., 0])
    y_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ty, logits=raw_box[..., 1])
    w_loss = tf.square(tw-raw_box[..., 2])
    h_loss = tf.square(th-raw_box[..., 3])
    box_loss = box_scale * tf.stack([x_loss, y_loss, w_loss, h_loss], -1)
    return box_loss

def box_boxes_iou_tf(cbox, turth_boxes):
    box_area = cbox[2] * cbox[3]
    areas = turth_boxes[..., 2] * turth_boxes[..., 3]
    xx1 = tf.maximum(cbox[0] - 0.5 * cbox[2], turth_boxes[..., 0] - 0.5 * turth_boxes[..., 2])
    yy1 = tf.maximum(cbox[1] - 0.5 * cbox[3], turth_boxes[..., 1] - 0.5 * turth_boxes[..., 3])
    xx2 = tf.minimum(cbox[0] + 0.5 * cbox[2], turth_boxes[..., 0] + 0.5 * turth_boxes[..., 2])
    yy2 = tf.minimum(cbox[1] + 0.5 * cbox[3], turth_boxes[..., 1] + 0.5 * turth_boxes[..., 3])

    # compute the width and height of the bounding box
    w = tf.maximum(tf.constant(0, dtype=tf.float16), xx2 - xx1)
    h = tf.maximum(tf.constant(0, dtype=tf.float16), yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + areas - inter)
    return ovr  # h,w,a

def cond(j, num, boxes, truth_boxes, ious):
    return j<num

def body(j, num, boxes, truth_boxes, ious):
    box = boxes[j]
    iou = box_boxes_iou_tf(box, truth_boxes) # [num of truthbox]
    iou = tf.expand_dims(iou, -1)
    ious = tf.concat([ious, iou], -1)
    j += 1
    return j, num, boxes, truth_boxes, ious


def pred_boxes_with_turth_boxes_ious(pred_feat, raw_truth, obj_mask):
    h, w, n, _ = pred_feat.get_shape().as_list()
    pred_boxes = tf.reshape(pred_feat[..., 0:4], [-1, 4])
    # obj_mask = tf.expand_dims(obj_mask, -1)
    # index = tf.where(tf.equal(obj_mask, 1))
    # reshaped_truth_boxes = tf.gather_nd(raw_truth, index)
    ious = tf.expand_dims(tf.zeros_like(raw_truth[..., 0], tf.float16), -1)
    j = tf.constant(0)
    num = tf.constant(h*w*n, tf.int32)
    j, num, boxes, truth_boxes, ious = tf.while_loop(cond, body, [j, num, pred_boxes, raw_truth[..., 0:4], ious],
                                                     shape_invariants=[j.get_shape(), num.get_shape(),
                                                                       pred_boxes.get_shape(), raw_truth[...,0:4].get_shape(),
                                                                       tf.TensorShape([h,w,n,None])])
    ious = tf.reduce_max(ious[..., 1::], -1)
    return ious

def calc_batches_losses_with_map(raw_feat, raw_truth, anchors):
    b, h, w, c = raw_feat.get_shape().as_list()
    raw_feat = tf.reshape(raw_feat, (b, h, w, 3, -1))# p04 5 cls
    pred_feat = active_map(raw_feat, anchors)
    obj_mask = raw_truth[..., 4]
    box_loss = cals_box_loss(raw_feat[..., 0:4], raw_truth[..., 0:4], w, h, anchors, obj_mask)
    box_loss = tf.reduce_sum(box_loss * tf.expand_dims(obj_mask, -1))
    cls_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=raw_truth[..., 5::], logits=raw_feat[..., 5::], name="cls_loss") * obj_mask)

    obj_loss, noobj_loss = 0., 0.
    for i in range(b):
        best_iou = pred_boxes_with_turth_boxes_ious(pred_feat[i], raw_truth[i], obj_mask[i]) # shape = [-1, num_of_truth_box]
        # best_iou = tf.reduce_max(boxes_iou, -1)
        ignore_mask = tf.where(best_iou < 0.7, tf.ones_like(best_iou), tf.zeros_like(best_iou))
        loss_mask = tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_mask[i], logits=raw_feat[i, ..., 4])
        obj_loss += tf.reduce_sum(loss_mask * obj_mask[i])
        noobj_loss += tf.reduce_sum((1 - obj_mask[i]) * loss_mask * ignore_mask)
    return noobj_loss/cfg.batchsize, obj_loss/cfg.batchsize, cls_loss/cfg.batchsize, box_loss/cfg.imagesize



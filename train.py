from Yolov3Net import *
from image_util import *
from utils import *
from loss_util import *
import numpy as np
import random
from load_data_from_json import gen_BIGbatch_data
from gen_maps import convert_cboxes_to_maps

def preprocess(datafile):
    data = get_dict(datafile)
    train_imgpaths = get_train_imgpaths(data)
    # return random.sample(train_imgpaths, cfg.batchsize)
    return train_imgpaths

def tranfrom(cboxes):
    tmps=[]
    num = []
    for cbox in cboxes:
        cbox = np.array(cbox, np.float32)
        tmp = np.zeros((cfg.targets_per_image, 5))
        if cbox.shape[0] > cfg.targets_per_image:
            tmp[:, 1::] = cbox[0:cfg.targets_per_image]
            num.append(cfg.targets_per_image)
        else:
            tmp[0:cbox.shape[0], 1::] = cbox
            num.append(cbox.shape[0])
        tmps.append(tmp)
    return np.stack(tmps, 0), np.array(num)

def readlabel(txtname):
    f= open(txtname, "r")
    labels = []
    for line in f.readlines():
        labels.append(list(map(float, line.strip().split(" "))))
    f.close()
    return labels

def get_datas(txtfile):
    f = open(txtfile, "r")
    datas=[]
    for line in f.readlines():
        imgname, txtname = line.strip().split(",")
        labels = readlabel(txtname)
        datas.append(zip(imgname, labels))
    return datas


def train(trainpath,init_model=None):
    '''
    :param tfrecordsfile:
    :return:
    '''
    datas = get_datas(trainpath)
    graph = tf.Graph()
    with graph.as_default():
        image = tf.placeholder(dtype=tf.float32, shape=[cfg.batchsize, cfg.imagesize, cfg.imagesize, 3], name='image')
        maps_13_placeholder = tf.placeholder(dtype=tf.float32, shape=[cfg.batchsize, 13, 13, 3, 6], name="maps13")
        maps_26_placeholder = tf.placeholder(dtype=tf.float32, shape=[cfg.batchsize, 26, 26, 3, 6], name="maps26")

        step = tf.Variable(0, trainable=False)

        darknet = darknet_tiny(image, cfg.anchors)
        if init_model:
            darknet.load_params(init_model)

        darknet.build(image)
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        L2_loss = darknet.get_l2_loss()
        noobj_loss_13, obj_loss_13, cls_loss_13, box_loss_13 \
            = calc_batches_losses_with_map(darknet.det1, maps_13_placeholder, cfg.anchors[3::])
        # total_loss = noobj_loss + obj_loss + cls_loss + box_loss + L2_loss
        noobj_loss_26, obj_loss_26, cls_loss_26, box_loss_26 \
            = calc_batches_losses_with_map(darknet.det2, maps_26_placeholder, cfg.anchors[0:3])

        noobj_loss = noobj_loss_13 + noobj_loss_26
        obj_loss = obj_loss_13 + obj_loss_26
        cls_loss = cls_loss_13 + cls_loss_26
        box_loss = box_loss_13 + box_loss_26
        total_loss = noobj_loss + obj_loss + cls_loss + box_loss + L2_loss
        #///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # bounaries = config.epochs
        # lr_values = [config.lr * config.lr_factor**i for i in range(len(bounaries) + 1)]
        # lr = tf.train.piecewise_constant(step, bounaries, lr_values)
        optimzer = tf.train.AdamOptimizer(1e-3)
        grads_vars = optimzer.compute_gradients(total_loss)
        # grads_vars = [(tf.clip_by_value(grad, 0, 5), var) for grad, var in grads_vars if grad is not None]
        optimzer_op = optimzer.apply_gradients(grads_vars, global_step=step)

        ## summary_op
        tf.summary.scalar('loss/total', total_loss)
        tf.summary.scalar('loss/noobj', noobj_loss)
        tf.summary.scalar('loss/obj', obj_loss)
        tf.summary.scalar('loss/clc', cls_loss)
        tf.summary.scalar('loss/box', box_loss)
        tf.summary.scalar('loss/L2', L2_loss)
        # tf.summary.image('image', image_batch)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=2)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if init_model:
            print('restored model from %s'%init_model)
            ckpt = tf.train.get_checkpoint_state(init_model)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_path = str(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(os.getcwd(), ckpt_path))
                s = int(init_model.split('-')[-1].split(".")[0])
                print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
            else:
                raise FileNotFoundError("\nCheckpoint Loading Failed! \n")
        else:
            s = 0
            print("\nInitial Train! \n")
        # write summary
        sum_writer = tf.summary.FileWriter(logdir=cfg.logdir, graph=sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        losses = 10
        while not coord.should_stop():
            s += 1
            if losses>0.01:
                batchdata = random.sample(datas, cfg.batchsize)
                imgpaths = [imgpath for imgpath, _  in batchdata]
                cboxes = [cbox for _, cbox in batchdata]
                image_array = gen_images_batch(imgpaths)
                image_array = (image_array-127.5)/128
                batches_maps_13, batches_maps_26 = convert_cboxes_to_maps(cboxes, cfg.anchors)
                feed_dict = {image: image_array, maps_13_placeholder: batches_maps_13, maps_26_placeholder: batches_maps_26}
                _, summary, total_loss_value, noobj_loss_value, obj_loss_value, cls_loss_value, box_loss_value = \
                    sess.run([optimzer_op, summary_op, total_loss,
                              noobj_loss, obj_loss, cls_loss, box_loss],
                             feed_dict=feed_dict)

                if s % 10 == 0:
                     print('step: %d  lr: %.5f  loss: %.5f  noobj_loss: %.5f  obj_loss: %.5f  cls_loss: %.5f  box_loss: %.5f'
                           % (s, 0.0, total_loss_value, noobj_loss_value, obj_loss_value, cls_loss_value, box_loss_value))
                if s % 100 == 0:
                    saver.save(sess, cfg.modelpath+'/tiny_yolo3_tf', global_step=step)
                    # darknet.save_params(sess, cfg.modelpath+'/tiny_yolo3-%d'%s, feed_dict)
                    print("model stored")
                sum_writer.add_summary(summary, global_step=s)
            else:
                break
        coord.request_stop()
        sum_writer.close()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    init_model = r'./model'
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    trainpath = "./train.txt"
    train(trainpath=trainpath,init_model=init_model)

import tensorflow as tf
import numpy as np
from config import config as cfg

def prelu(inputs, slop=0.01):
    compared = slop * inputs
    return tf.maximum(inputs, compared, name="PRelu")

class darknet_53():
    def __init__(self, inputs, testable=False, use_l2_loss=True):
        self.inputs = inputs
        self.testable = testable
        self.use_l2_loss = use_l2_loss

    def load_params(self, path=None):
        self.params = np.load(path, encoding="latin1").item()

    def get_param(self, name):
        w = tf.Variable(self.params["%s_w"%name], trainable=not self.testable, name="%s_w" % name)
        b = tf.Variable(self.params["%s_b"%name], trainable=not self.testable, name="%s_b" % name)
        if self.use_l2_loss:
            tf.add_to_collection("l2_loss", w)
        return w, b

    def init_param(self, shape, name, use_l2_loss=True):
        w = tf.Variable(tf.truncated_normal(shape, 0.0, 0.01), trainable=True, name="%s_w"%name)
        b = tf.Variable(tf.truncated_normal([shape[-1]], 0.0, 0.01), trainable=True, name="%s_b"%name)
        if use_l2_loss:
            tf.add_to_collection("l2_loss", w)
        return w, b

    def batchnorm(self, x):
        return tf.layers.batch_normalization(x)

    def conv2d(self, x, shape, strides, padding, name, active="leaky"):
        if self.testable:
            w, b = self.get_param(name)
        else:
            w, b = self.init_param(shape, name)
        x = tf.nn.conv2d(x, w, strides, padding, name=name)
        x = self.batchnorm(x)
        x = tf.nn.bias_add(x, b)
        if active=="leaky":
            return prelu(x, 0.1)
        elif active=="relu":
            return tf.nn.relu(x)
        else:
            return x

    def deconv2d(self, x, shape, strides, padding, name):
        b, h, w, c = x.get_shape().as_list()
        if self.testable:
            weights, _ = self.get_param(name)
        else:
            weights, _ = self.init_param(shape, name)
        x = tf.nn.conv2d_transpose(x, weights, [b, h*2, w*2, c], strides, padding, name=name)
        return x

    def resblock(self, x, out_shapes, name):
        x1 = self.conv2d(x, out_shapes[0], [1, 1, 1, 1], "SAME", "%s_1" % name)
        x2 = self.conv2d(x1, out_shapes[1], [1, 1, 1, 1], "SAME", "%s_2" % name)
        return x+x2

    def build(self, x):
        x = self.conv2d(x, [3, 3, 3, 32], [1, 1, 1, 1], "SAME", "conv1")
        x = self.conv2d(x, [3, 3, 32, 64], [1, 2, 2, 1], "SAME", "conv2")
        x = self.resblock(x, [[1, 1, 64, 32], [3, 3, 32, 64]], "conv3")
        print(x.get_shape().as_list())
        x = self.conv2d(x, [3, 3, 64, 128], [1, 2, 2, 1], "SAME", "conv5")
        print(x.get_shape().as_list())
        x = self.resblock(x, [[1, 1, 128, 64], [3, 3, 64, 128]], "conv6")
        x = self.resblock(x, [[1, 1, 128, 64], [3, 3, 64, 128]], "conv7")
        print(x.get_shape().as_list())
        x = self.conv2d(x, [3, 3, 128, 256], [1, 2, 2, 1], "SAME", "conv8")
        print(x.get_shape().as_list())
        x = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv9")
        x = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv10")
        x = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv11")
        x = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv12")
        x = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv13")
        x = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv14")
        conv15 = self.resblock(x, [[1, 1, 256, 128], [3, 3, 128, 256]], "conv15")
        print(conv15.get_shape().as_list())
        x = self.conv2d(conv15, [3, 3, 256, 512], [1, 2, 2, 1], "SAME", "conv16")
        print(x.get_shape().as_list())
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv17")
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv18")
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv19")
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv20")
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv21")
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv22")
        x = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv23")
        conv24 = self.resblock(x, [[1, 1, 512, 256], [3, 3, 256, 512]], "conv24")
        print(conv24.get_shape().as_list())
        x = self.conv2d(conv24, [3, 3, 512, 1024], [1, 2, 2, 1], "SAME", "conv25")
        print(x.get_shape().as_list())
        x = self.resblock(x, [[1, 1, 1024, 512], [3, 3, 512, 1024]], "conv26")
        x = self.resblock(x, [[1, 1, 1024, 512], [3, 3, 512, 1024]], "conv27")
        x = self.resblock(x, [[1, 1, 1024, 512], [3, 3, 512, 1024]], "conv28")
        x = self.resblock(x, [[1, 1, 1024, 512], [3, 3, 512, 1024]], "conv29")
        print(x.get_shape().as_list())
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # detection1
        x = self.conv2d(x, [1, 1, 1024, 512], [1, 1, 1, 1], "SAME", "conv30")
        x = self.conv2d(x, [3, 3, 512, 1024], [1, 1, 1, 1], "SAME", "conv31")
        conv32 = self.conv2d(x, [1, 1, 1024, 512], [1, 1, 1, 1], "SAME", "conv32")
        x = self.conv2d(conv32, [3, 3, 512, 1024], [1, 1, 1, 1], "SAME", "conv33")
        x = self.conv2d(x, [1, 1, 1024, 512], [1, 1, 1, 1], "SAME", "conv34")
        print(x.get_shape().as_list())
        x = self.conv2d(x, [3, 3, 512, 1024], [1, 1, 1, 1], "SAME", "conv35")
        self.det1 = self.conv2d(x, [1, 1, 1024, cfg.lastchannel], [1, 1, 1, 1], "SAME", "detection1", active="linear")
        print(self.det1.get_shape().as_list())
        # detection2
        # route1 = tf.concat([detection1, conv32], -1)
        x = self.conv2d(conv32, [1, 1, 512, 256], [1, 1, 1, 1], "SAME", "conv37")
        x = self.deconv2d(x, [3, 3, 256, 256], [1, 2, 2, 1], "SAME", "deconv1")
        x = tf.concat([x, conv24], -1)
        print(x.get_shape().as_list())
        x = self.conv2d(x, [1, 1, 768, 256], [1, 1, 1, 1], "SAME", "conv38")
        x = self.conv2d(x, [3, 3, 256, 512], [1, 1, 1, 1], "SAME", "conv39")
        x = self.conv2d(x, [1, 1, 512, 256], [1, 1, 1, 1], "SAME", "conv40")
        x = self.conv2d(x, [3, 3, 256, 512], [1, 1, 1, 1], "SAME", "conv41")
        conv42 = self.conv2d(x, [1, 1, 512, 256], [1, 1, 1, 1], "SAME", "conv42")
        print(conv42.get_shape().as_list())
        x = self.conv2d(conv42, [3, 3, 256, 512], [1, 1, 1, 1], "SAME", "conv43")
        self.det2 = self.conv2d(x, [1, 1, 512, cfg.lastchannel], [1, 1, 1, 1], "SAME", "detection2", active="linear")
        print(self.det2.get_shape().as_list())
        # detection3
        x = self.conv2d(conv42, [1, 1, 256, 128], [1, 1, 1, 1], "SAME", "conv45")
        x = self.deconv2d(x, [3, 3, 128, 128], [1, 2, 2, 1], "SAME", "deconv2")
        x = tf.concat([x, conv15], -1)
        x = self.conv2d(x, [1, 1, 384, 128], [1, 1, 1, 1], "SAME", "conv46")
        x = self.conv2d(x, [3, 3, 128, 256], [1, 1, 1, 1], "SAME", "conv47")
        x = self.conv2d(x, [1, 1, 256, 128], [1, 1, 1, 1], "SAME", "conv48")
        x = self.conv2d(x, [3, 3, 128, 256], [1, 1, 1, 1], "SAME", "conv49")
        x = self.conv2d(x, [1, 1, 256, 128], [1, 1, 1, 1], "SAME", "conv50")
        x = self.conv2d(x, [3, 3, 128, 256], [1, 1, 1, 1], "SAME", "conv51")
        self.det3 = self.conv2d(x, [1, 1, 256, cfg.lastchannel], [1, 1, 1, 1], "SAME", "detection3", active="linear")
        print(self.det3.get_shape().as_list())

    def save_params(self, sess, npy_path):
        self.params = {}
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in vars:
            if "conv" in var.name.split("/")[-1]:
                value = sess.run(var)
                self.params[var.name.split("/")[-1].split(":")[0]] = value
            elif "detection" in var.name.split("/")[-1]:
                value = sess.run(var)
                self.params[var.name.split("/")[-1].split(":")[0]] = value
            elif "deconv" in var.name.split("/")[-1]:
                value = sess.run(var)
                self.params[var.name.split("/")[-1].split(":")[0]] = value
            else:
                print(var.name)
                pass
        np.save(npy_path, self.params)

    def get_l2_loss(self):
        l2_loss = tf.get_collection("l2_loss")
        return tf.add_n([tf.nn.l2_loss(loss) for loss in l2_loss])

class darknet_tiny():
    def __init__(self, inputs, anchors, testable=False, use_l2_loss=True):
        self.inputs = inputs
        self.testable = testable
        self.use_l2_loss = use_l2_loss
        self.l2_scale = 0.0005
        self.eps = 1e-9
        self.modelpath = None
        self.anchors = anchors
        # self.params = {}

    def get_param(self, name, collect=True):
        v = tf.Variable(self.params[name], trainable=not self.testable, name=name)
        # b = tf.Variable(self.params["%s_b"%name], trainable=not self.testable, name="%s_b" % name)
        if collect:
            tf.add_to_collection("l2_loss", v)
        return v

    def init_param(self, shape, name):
        w = tf.Variable(tf.truncated_normal(shape, 0.0, 0.001), trainable=True, name="%s_w"%name)
        b = tf.Variable(tf.truncated_normal([shape[-1]], 0., 0.001), trainable=True, name="%s_b"%name)
        if self.use_l2_loss:
            tf.add_to_collection("l2_loss", w)
        return w, b

    def batchnorm(self, x, name):
        b, h, w, c = x.get_shape().as_list()
        if cfg.train_phase:
            average_mean = self.get_param("%s_bn_average_mean"%name, collect=False)
            average_varance = self.get_param("%s_bn_average_varance"%name, collect=False)
            gama = self.get_param("%s_bn_gama"%name, collect=False)
            beta = self.get_param("%s_bn_beta"%name, collect=False)
        else:
            average_mean, average_varance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
            gama = tf.Variable(tf.ones(c), name="%s_bn_gama"%name)
            beta = tf.Variable(tf.zeros(c), name="%s_bn_beta"%name)
            tf.add_to_collection("means", average_mean)
            tf.add_to_collection("varances", average_varance)

        x = tf.nn.batch_normalization(x, mean=average_mean, variance=average_varance,
                                  offset=beta, scale=gama,
                                  variance_epsilon=self.eps, name="%s_bn"%name)
        return x

    def conv2d(self, x, shape, strides, padding, name, bias=True, BN=True, active="leaky"):
        if self.modelpath:
            w = self.get_param("%s_w"%name)
            b = self.get_param("%s_b"%name)
        else:
            w, b = self.init_param(shape, name)
        # w = tf.cast(w, tf.float16)
        x = tf.nn.conv2d(x, w, strides, padding, name=name)
        if BN:
            x = self.batchnorm(x, name)
        if bias:
            x = tf.nn.bias_add(x, b)
        if active=="leaky":
            return prelu(x, 0.1)
        elif active=="relu":
            return tf.nn.relu(x)
        else:
            return x

    def max_pool(self, x, ksize, strides, name):
        x = tf.nn.max_pool(x, ksize, strides, "SAME", name=name)
        return x

    def upsample(self, x, stride, name):
        with tf.name_scope(name):
            b, h, w, c = x.get_shape().as_list()
            new_shape = (h*stride, w*stride)
            x = tf.image.resize_images(x, new_shape, method=1)
            return x

    def deconv2d(self, x, shape, strides, padding, name):
        b, h, w, c = x.get_shape().as_list()
        if self.testable:
            weights = self.get_param("%s_w"%name)
        else:
            weights, _ = self.init_param(shape, name)
        x = tf.nn.conv2d_transpose(x, weights, [b, h*2, w*2, c], strides, padding, name=name)
        return x

    def resblock(self, x, out_shapes, name):
        x1 = self.conv2d(x, out_shapes[0], [1, 1, 1, 1], "SAME", "%s_1" % name, bias=False, BN=False, active="relu")
        x2 = self.conv2d(x1, out_shapes[1], [1, 1, 1, 1], "SAME", "%s_2" % name, bias=False, BN=False, active="relu")
        return x+x2

    def active_map(self, pred_per_batch, anchors):
        b, h, w, _ = pred_per_batch.get_shape().as_list()
        pred_per_batch = tf.reshape(pred_per_batch, (b, h, w, anchors.shape[0], -1))
        mask_x = np.tile(np.arange(0, w, 1), h).reshape([1, h, w, 1])
        mask_y = np.tile(np.arange(0, h, 1), w).reshape([w, h]).T.reshape([1, h, w, 1])
        anchorlist = tf.cast(tf.reshape(anchors, [1, 1, 1, 3, 2]), tf.float32)

        tx = (tf.nn.sigmoid(pred_per_batch[..., 0]) + mask_x)/w
        ty = (tf.nn.sigmoid(pred_per_batch[..., 1]) + mask_y)/h
        tw = tf.exp(pred_per_batch[..., 2]) * anchorlist[..., 0] / cfg.imagesize
        th = tf.exp(pred_per_batch[..., 3]) * anchorlist[..., 1] / cfg.imagesize

        A = tf.stack([tx, ty, tw, th], -1)
        objs = tf.expand_dims(tf.nn.sigmoid(pred_per_batch[..., 4]), -1)
        cls = tf.nn.softmax(pred_per_batch[..., 5::], -1)
        return tf.concat((A, objs, cls), -1)

    def build(self, x):
        x = self.conv2d(x, [3, 3, 3, 16], [1, 1, 1, 1], "SAME", "conv1")
        print(x.get_shape().as_list()) # 1,416,416,16

        x = self.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "maxpool1")
        x = self.conv2d(x, [3, 3, 16, 32], [1, 1, 1, 1], "SAME", "conv2")

        x = self.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "maxpool2")
        x = self.conv2d(x, [3, 3, 32, 64], [1, 1, 1, 1], "SAME", "conv3")
        x = self.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "maxpool3")
        x = self.conv2d(x, [3, 3, 64, 128], [1, 1, 1, 1], "SAME", "conv4")
        maxpool4 = self.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "maxpool4")
        print(maxpool4.get_shape().as_list())
        x = self.conv2d(maxpool4, [3, 3, 128, 256], [1, 1, 1, 1], "SAME", "conv5")
        x = self.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "maxpool5")
        x = self.conv2d(x, [3, 3, 256, 512], [1, 1, 1, 1], "SAME", "conv6")
        x = self.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], "maxpool6")
        conv7 = self.conv2d(x, [3, 3, 512, 1024], [1, 1, 1, 1], "SAME", "conv7")

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # detecion1
        x = self.conv2d(conv7, [1, 1, 1024, 256], [1, 1, 1, 1], "SAME", "conv8")
        x = self.conv2d(x, [3, 3, 256, 512], [1, 1, 1, 1], "SAME", "conv9")
        self.det1 = self.conv2d(x, [1, 1, 512, int(cfg.lastchannel/2)], [1, 1, 1, 1], "SAME", "detection1", BN=False, active="linear")
        if self.testable:
            self.det1 = self.active_map(self.det1, self.anchors[3::])
        print(self.det1.get_shape().as_list())
        # detection2
        x = self.conv2d(conv7, [1, 1, 1024, 128], [1, 1, 1, 1], "SAME", "conv10")
        x = self.upsample(x, stride=2, name="upsample")

        # x = self.deconv2d(x, [3, 3, 128, 128], [1, 2, 2, 1], "SAME", "deconv1")
        x = tf.concat([x, maxpool4], -1)
        x = self.conv2d(x, [3, 3, 256, 256], [1, 1, 1, 1], "SAME", "conv11")
        self.det2 = self.conv2d(x, [1, 1, 256, int(cfg.lastchannel/2)], [1, 1, 1, 1], "SAME", "detection2", BN=False, active="linear")
        self.test = tf.identity(self.det2)
        if self.testable:
            self.det2 = self.active_map(self.det2, self.anchors[0:3])
        print(self.det2.get_shape().as_list())


    def get_l2_loss(self):
        l2_loss = tf.get_collection("l2_loss")
        return tf.add_n([self.l2_scale * tf.nn.l2_loss(loss) for loss in l2_loss])




if __name__ == '__main__':
    img = tf.zeros([1, 416, 416, 3], dtype=tf.float32)
    out = darknet_53(img, True)
    out.load_params("../testmodel.npy")
    out.build(img)
    l2_loss = out.get_l2_loss()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    a = sess.run(l2_loss)
    out.save_params(sess, "../testmodel.npy")
    # det = np.random.random_sample([1, 52, 52, 66])
    # # det = det.reshape([2,13,13,6,-1])
    # # indexes = np.where(det[..., 4] > 0.5)
    # det = get_det(det, cfg.anchors)
    # det = postprocess(det, 416, 416)
    # print(det.shape)
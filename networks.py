import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib as tc
FLAGS = flags.FLAGS
from nets.serenset50 import se_resnet
from nets.resnet50 import resnet_deep
from nets.resnet34 import resnet_shallow
from nets.densenet import densenet, densenetbc
from nets.resnext import resnext



class ResNet13(object):
    def __init__(self, size):
        self.channels = 3
        self.dim_hidden = FLAGS.base_num_filters
        self.dim_output = FLAGS.num_classes
        self.img_size = 84
        self.train_flag = True

    def forward(self, inp, train_flag):

        feature = tf.reshape(inp, [-1, 32, 32, 3])

        for i in range(4):
            block_name = str(i + 1)

            shortcut = tf.layers.conv2d(feature, self.dim_hidden*np.power(2, i), kernel_size=(1,1),
                                        padding='same', strides=[1,1], name=block_name + '/shortcut/conv',
                                        reuse=tf.AUTO_REUSE)

            for j in ['a', 'b']:
                feature = tf.layers.conv2d(feature, self.dim_hidden*np.power(2, i), kernel_size=(3,3),
                                           padding='SAME', strides=[1, 1], name=block_name + '/' + j + '/conv/',
                                           reuse=tf.AUTO_REUSE)

                axis = [k for k in range(len(feature.get_shape().as_list()) - 1)]
                mean, variance = tf.nn.moments(feature, axis, name='moments')
                feature = tf.nn.batch_normalization(feature, mean, variance, None, None, 1e-05)
                feature = tf.nn.relu(feature)

            feature = tf.layers.conv2d(feature, self.dim_hidden * np.power(2, i), kernel_size=(3, 3),
                                       padding='SAME', strides=[1, 1], name=block_name + '/c/conv/',
                                       reuse=tf.AUTO_REUSE)

            feature = feature + shortcut

            axis = [k for k in range(len(feature.get_shape().as_list()) - 1)]
            mean, variance = tf.nn.moments(feature, axis, name='moments')
            feature = tf.nn.batch_normalization(feature, mean, variance, None, None, 1e-05)
            feature = tf.nn.relu(feature)

            if i < 3:
                feature = tf.layers.max_pooling2d(feature, [2, 2], [2, 2], 'same')

        feature = tf.reduce_mean(feature, axis=[1, 2])

        fc1 = tf.layers.dense(feature, self.dim_output, name='dense', reuse=tf.AUTO_REUSE)
        return fc1

  
    
class ResNet(object):
    def __init__(self, size):
        self.dim_output = FLAGS.num_classes
        self.size=size


    def forward(self, inp, train_flag):
        if self.size < 50:
            hidden4 = resnet_shallow(inp, self.size, is_training=train_flag, reuse=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        else:
            hidden4 = resnet_deep(inp, self.size, is_training=train_flag, reuse=False, kernel_initializer=tf.orthogonal_initializer())
        fc1 = tf.layers.dense(hidden4, self.dim_output)
        return fc1



class SeResNet(object):
    def __init__(self, size):
        self.dim_output = FLAGS.num_classes
        self.size=size


    def forward(self, inp, train_flag):
        hidden4 = se_resnet(inp, self.size, is_training=train_flag, reuse=False, kernel_initializer=tf.orthogonal_initializer())
        fc1 = tf.layers.dense(hidden4, self.dim_output)
        return fc1



class ResNext(object):
    def __init__(self, size):
        self.dim_output = FLAGS.num_classes
        self.size=size


    def forward(self, inp, train_flag):
        hidden4 = resnext(inp, self.size, is_training=train_flag, reuse=False)
        fc1 = tf.layers.dense(hidden4, self.dim_output)
        return fc1



class DenseNet(object):
    def __init__(self, size, BC=False):
        self.dim_output = FLAGS.num_classes
        self.size=size
        self.BC = BC


    def forward(self, inp, train_flag):
        if self.BC:
            hidden4 = densenetbc(inp, self.size, is_training=train_flag, reuse=False, kernel_initializer=tf.orthogonal_initializer())
        else:
            hidden4 = densenet(inp, self.size, is_training=train_flag, reuse=False, kernel_initializer=tf.orthogonal_initializer())

        fc1 = tf.layers.dense(hidden4, self.dim_output)
        return fc1




class MobileNetV1(object):
    def __init__(self, size):
        self.dim_output = FLAGS.num_classes
        self.size = size
        self.normalizer = tc.layers.batch_norm


    def forward(self, input, train_flag):

        self.bn_params = {'is_training': train_flag}
        with tf.variable_scope('MobileNetV1'):

            i = 0
            with tf.variable_scope('init_conv'):
                self.conv1 = tc.layers.conv2d(input, num_outputs=32, kernel_size=3, stride=2,
                                               normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            # 1
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv1 = tc.layers.separable_conv2d(self.conv1, num_outputs=None, kernel_size=3, depth_multiplier=1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv1 = tc.layers.conv2d(self.dconv1, 64, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            # 2
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv2 = tc.layers.separable_conv2d(self.pconv1, None, 3, 1, 2,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv2 = tc.layers.conv2d(self.dconv2, 128, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            # 3
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv3 = tc.layers.separable_conv2d(self.pconv2, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv3 = tc.layers.conv2d(self.dconv3, 128, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 4
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv4 = tc.layers.separable_conv2d(self.pconv3, None, 3, 1, 2,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv4 = tc.layers.conv2d(self.dconv4, 256, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 5
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv5 = tc.layers.separable_conv2d(self.pconv4, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv5 = tc.layers.conv2d(self.dconv5, 256, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 6
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv6 = tc.layers.separable_conv2d(self.pconv5, None, 3, 1, 2,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv6 = tc.layers.conv2d(self.dconv6, 512, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 7_1
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv71 = tc.layers.separable_conv2d(self.pconv6, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv71 = tc.layers.conv2d(self.dconv71, 512, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 7_2
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv72 = tc.layers.separable_conv2d(self.pconv71, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv72 = tc.layers.conv2d(self.dconv72, 512, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 7_3
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv73 = tc.layers.separable_conv2d(self.pconv72, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv73 = tc.layers.conv2d(self.dconv73, 512, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 7_4
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv74 = tc.layers.separable_conv2d(self.pconv73, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv74 = tc.layers.conv2d(self.dconv74, 512, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 7_5
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv75 = tc.layers.separable_conv2d(self.pconv74, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv75 = tc.layers.conv2d(self.dconv75, 512, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 8
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv8 = tc.layers.separable_conv2d(self.pconv75, None, 3, 1, 2,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv8 = tc.layers.conv2d(self.dconv8, 1024, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            # 9
            with tf.variable_scope('dconv_block{}'.format(i)):
                i += 1
                self.dconv9 = tc.layers.separable_conv2d(self.pconv8, None, 3, 1, 1,
                                                         normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                self.pconv9 = tc.layers.conv2d(self.dconv9, 1024, 1, normalizer_fn=self.normalizer,
                                               normalizer_params=self.bn_params)

            with tf.variable_scope('prediction'):
                output = tc.layers.conv2d(self.pconv9, self.dim_output, 1, activation_fn=None)
                self.output = tf.squeeze(output, axis=[1, 2])

        return self.output



class MobileNetV2(object):
    def __init__(self, size=32):
        self.dim_output = FLAGS.num_classes
        self.size = size
        self.normalizer = tc.layers.batch_norm


    def forward(self, input, train_flag):
        self.bn_params = {'is_training': train_flag}
        with tf.variable_scope('MobileNetV2'):
            self.i = 0
            with tf.variable_scope('init_conv'):
                output = tc.layers.conv2d(input, 32, 3, 2,
                                          normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
                print(output.get_shape())
            self.output = self._inverted_bottleneck(output, 1, 16, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 24, 1)
            self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
            self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
            self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
            self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
            self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
            self.output = tc.layers.conv2d(self.output, 1280, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            # self.output = tc.layers.avg_pool2d(self.output, 8)
            self.output = tc.layers.conv2d(self.output, self.dim_output, 1, activation_fn=None)
            self.output = tf.squeeze(self.output, axis=[1, 2])

        return self.output


    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample):
        with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(self.i, up_sample_rate, subsample)):
            self.i += 1
            stride = 2 if subsample else 1
            output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1,
                                      activation_fn=tf.nn.relu6,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation_fn=tf.nn.relu6,
                                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            if input.get_shape().as_list()[-1] == channels:
                output = tf.add(input, output)
            return output



from nets.mnv3_layers import *
class MobileNetV3(object):
    def __init__(self, size=32):
        self.size = size
        self.normalizer = tc.layers.batch_norm
        self.dim_output = FLAGS.num_classes


    def forward(self, input, train_flag):

        reduction_ratio = 4
        with tf.variable_scope('mobilenetv3_larage'):
            net = conv2d_block(input, 16, 3, 2, train_flag, name='conv1_1', h_swish=True)  # size/2

            net = mnv3_block(net, 3, 16, 16, 1, train_flag, name='bneck2_1', h_swish=False, ratio=reduction_ratio,
                             se=False)

            net = mnv3_block(net, 3, 64, 24, 2, train_flag, name='bneck3_1', h_swish=False, ratio=reduction_ratio,
                             se=False)  # size/4
            net = mnv3_block(net, 3, 72, 24, 1, train_flag, name='bneck3_2', h_swish=False, ratio=reduction_ratio,
                             se=False)

            net = mnv3_block(net, 5, 72, 40, 2, train_flag, name='bneck4_1', h_swish=False, ratio=reduction_ratio,
                             se=True)  # size/8
            net = mnv3_block(net, 5, 120, 40, 1, train_flag, name='bneck4_2', h_swish=False, ratio=reduction_ratio,
                             se=True)
            net = mnv3_block(net, 5, 120, 40, 1, train_flag, name='bneck4_3', h_swish=False, ratio=reduction_ratio,
                             se=True)

            net = mnv3_block(net, 3, 240, 80, 2, train_flag, name='bneck5_1', h_swish=True, ratio=reduction_ratio,
                             se=False)  # size/16
            net = mnv3_block(net, 3, 200, 80, 1, train_flag, name='bneck5_2', h_swish=True, ratio=reduction_ratio,
                             se=False)
            net = mnv3_block(net, 3, 184, 80, 1, train_flag, name='bneck5_3', h_swish=True, ratio=reduction_ratio,
                             se=False)
            net = mnv3_block(net, 3, 184, 80, 1, train_flag, name='bneck5_4', h_swish=True, ratio=reduction_ratio,
                             se=False)

            net = mnv3_block(net, 3, 480, 112, 1, train_flag, name='bneck6_1', h_swish=True, ratio=reduction_ratio,
                             se=True)
            net = mnv3_block(net, 3, 672, 112, 1, train_flag, name='bneck6_2', h_swish=True, ratio=reduction_ratio,
                             se=True)

            net = mnv3_block(net, 5, 672, 160, 2, train_flag, name='bneck7_2', h_swish=True, ratio=reduction_ratio,
                             se=True)  # size/32
            net = mnv3_block(net, 5, 960, 160, 1, train_flag, name='bneck7_3', h_swish=True, ratio=reduction_ratio,
                             se=True)
            net = mnv3_block(net, 5, 960, 160, 1, train_flag, name='bneck7_1', h_swish=True, ratio=reduction_ratio,
                             se=True)

            net = conv2d_hs(net, 960, train_flag, name='conv8_1')
            # net = global_avg(net, 1)
            net = conv2d_NBN_hs(net, 1280, name='conv2d_NBN', bias=True)
            net = conv_1x1(net, self.dim_output, name='logits', bias=True)
            logits = flatten(net)

            return logits




from nets.shuffle import shufflenet
class ShuffleNetV1(object):
    def __init__(self, size=32):
        self.size = size
        self.dim_output = FLAGS.num_classes


    def forward(self, input, train_flag):
        out = shufflenet(input, is_training=train_flag, num_classes=self.dim_output)
        fc1 = tf.layers.dense(out, self.dim_output)
        return fc1



from nets.shuffle2 import shufflenet_V2
class ShuffleNetV2(object):
    def __init__(self, size=32):
        self.size = size
        self.dim_output = FLAGS.num_classes


    def forward(self, input, train_flag):
        out = shufflenet_V2(input, is_training=train_flag)
        fc1 = tf.layers.dense(out, self.dim_output)
        return fc1



class SqueezeNet(object):
	"""docstring for SqueezeNet"""
	def __init__(self, mode='A'):
		super(SqueezeNet, self).__init__()
		self.num_classes = FLAGS.num_classes
		self.conv_num = 0
		self.size = mode
		self.initializer = tf.contrib.layers.xavier_initializer()
		net_config = {
			'base': 128,
			'incre': 128,
			'pct33': 0.5,
			'freq': 2,
			'sr': 0.5
		}
		self.sr = net_config['sr']
		self.base = net_config['base']
		self.incre = net_config['incre']
		self.pct33 = net_config['pct33']
		self.freq = net_config['freq']

		# self.is_training
		self.keep_prob = 0

		if mode=='A':
			self.make_layer = self.make_layerA
		elif mode =='B':
			self.make_layer = self.make_layerB
		else:
			raise Exception("mode must be A or B")


	def Fiber_module(self,inputs,out_channel, is_training):
		sfilter1x1 = self.sr * out_channel
		efilter1x1 = (1-self.pct33) * out_channel
		efilter3x3 = self.pct33 * out_channel
		out = self.conv2d(inputs,sfilter1x1,kernel_size=1,stride=1, training=is_training)
		out_1 = self.conv2d(out,efilter1x1,kernel_size=1,stride=1, training=is_training)
		out_2 = self.conv2d(out,efilter3x3,kernel_size=3,stride=1, training=is_training)
		out = tf.concat([out_1,out_2],axis=-1)
		return out


	def Fiber_moduleB(self,inputs,out_channel, is_training):
		resudial = tf.identity(inputs)
		sfilter1x1 = self.sr * out_channel
		efilter1x1 = (1-self.pct33) * out_channel
		efilter3x3 = self.pct33 * out_channel
		out = self.conv2d(inputs,sfilter1x1,kernel_size=1,stride=1, training=is_training)
		out_1 = self.conv2d(out,efilter1x1,kernel_size=1,stride=1,relu=False, training=is_training)
		out_2 = self.conv2d(out,efilter3x3,kernel_size=3,stride=1,relu=False, training=is_training)
		out = tf.concat([out_1,out_2],axis=-1)
		return tf.nn.relu(tf.add(resudial,out))


	def conv2d(self,inputs,out_channel,kernel_size,stride,relu=True, training=True):
		out = tf.layers.conv2d(inputs,filters=out_channel,kernel_size=kernel_size,strides=stride,padding='same',
			kernel_initializer=self.initializer,name='conv_'+str(self.conv_num))
		self.conv_num+=1
		out = tf.layers.batch_normalization(out, training=training)
		return tf.nn.relu(out) if relu else out


	def forward(self,inputs, is_training):
		input_width = inputs.shape[-2]

		out = self.conv2d(inputs,out_channel=96,kernel_size=7,stride=2, training=is_training)
		out = tf.layers.max_pooling2d(out,pool_size=3,strides=2,padding='same',name='maxpool_0')
		out = self.make_layer(out, is_training)
		out = self.conv2d(out,out_channel=1000,kernel_size=1,stride=1, training=is_training)

		pool_size,stride = (input_width //16),(input_width//16)

		out = tf.layers.average_pooling2d(out,pool_size=(pool_size,pool_size),strides=(stride,stride),name='avg_pool_0')
		out = tf.layers.flatten(out,name='flatten')
		out = tf.layers.dropout(out,rate=self.keep_prob,name='dropout', training=is_training)
		predicts = tf.layers.dense(out,units=self.num_classes,kernel_initializer=self.initializer,name='fc')
		# softmax_out = tf.nn.softmax(predicts,name='output')

		return predicts
	

	def make_layerA(self,inputs, training):
		max_pool_loc = [4,8]
		pool_num = 1
		for i in range(2,10):
			out_channel = self.base+self.incre*((i-2)//self.freq)
			inputs = self.Fiber_module(inputs,out_channel, training)
			if i in max_pool_loc:
				inputs = tf.layers.max_pooling2d(inputs,pool_size=3,strides=2,padding='same',name='maxpool_'+str(pool_num))
				pool_num+=1
		return inputs


	def make_layerB(self,inputs, training):
		max_pool_loc = [4,8]
		pool_num = 1
		resudial_loc = [3,5,7,9]
		for i in range(2,10):
			out_channel = self.base+self.incre*((i-2)//self.freq)
			if i in resudial_loc:
				inputs = self.Fiber_moduleB(inputs,out_channel, training)
			else:
				inputs = self.Fiber_module(inputs,out_channel, training)
			if i in max_pool_loc:
				inputs = tf.layers.max_pooling2d(inputs,pool_size=3,strides=2,padding='same',name='maxpool_'+str(pool_num))
				pool_num+=1
		return inputs

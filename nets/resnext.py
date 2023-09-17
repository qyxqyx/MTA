import tensorflow as tf
import numpy as np 

# class group_layer(Layer):
#     def __init__(self, layer, filters, kernel_size, strides, act=tf.nn.relu, cardinality=8, is_train=True, w_init=None, b_init=None, name='group_layer'):
#         Layer.__init__(self, layer=layer, name=name)
#         self.inputs = layer.outputs

#         ans = []
#         per_filter = filters // cardinality
#         for i in range(cardinality):
#             split_conv = tf.gather(self.inputs, tf.range(i * cardinality, (i+1)*cardinality), axis=-1)
#             split_conv = tl.layers.InputLayer(split_conv, name=name+'_input_'+str(i))
#             conv1 = conv2d(split_conv, filters=per_filter, kernel_size=kernel_size, strides=strides, w_init=w_init, b_init=b_init, name=name+'_group_conv_'+str(i))
#             conv1 = bn(conv1, is_train=is_train, name=name+'_group_bn_'+str(i), act=act)
#             ans.append(conv1)

#         self.outputs = tf.concat(ans, axis=-1)
        


def conv2d(input_tensor, filters, kernel_size, strides, act=None, padding='SAME', w_init=None, b_init=None, name=None):
    return tf.layers.conv2d(input_tensor, filters=filters, kernel_size=kernel_size, strides=strides, activation=act, padding=padding, kernel_initializer=w_init, bias_initializer=b_init, name=name)

def bn(input_tensor, is_train, name, act=None):
    out = tf.layers.batch_normalization(input_tensor, training=is_train, name=name)
    if act != None:
        out = act(out)
    return out


def group_conv(input_tensor, filters, kernel_size, strides, act=tf.nn.relu, cardinality=8, is_train=True, w_init=None, b_init=None, name='group_n'):
    ans = []
    per_filter = filters // cardinality
    for i in range(cardinality):
        split_conv = tf.gather(input_tensor, tf.range(i * cardinality, (i+1)*cardinality), axis=-1)
        # split_conv = tl.layers.InputLayer(split_conv, name=name+'_input_'+str(i))
        conv1 = conv2d(split_conv, filters=per_filter, kernel_size=kernel_size, strides=strides, w_init=w_init, b_init=b_init, name=name+'_group_conv_'+str(i))
        conv1 = bn(conv1, is_train=is_train, name=name+'_group_bn_'+str(i), act=act)
        ans.append(conv1)

    out = tf.concat(ans, axis=-1)
    return out


def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, cardinality,  kernel_initializer=None):
    filters1, filters2, filters3 = filters

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'

    x = conv2d(input_tensor, filters1, (1, 1), strides=(1,1), padding='SAME', name=conv_name_1, w_init=kernel_initializer)
    x = bn(x, is_train=is_training, name=bn_name_1, act=tf.nn.relu)

    group_name_2 = 'group' + str(stage) + '_' + str(block) + '_3x3'

    x = group_conv(x, filters2, kernel_size=(3,3), strides=(1,1), act=tf.nn.relu, cardinality=cardinality, is_train=is_training, name=group_name_2)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
    x = conv2d(x, filters3, (1,1), strides=(1,1), name=conv_name_3, padding='SAME', w_init=kernel_initializer)
    x = bn(x, is_train=is_training, name=bn_name_3)

    x = x + input_tensor
    return tf.nn.relu(x)


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, cardinality, strides=(2, 2), kernel_initializer=None):
    filters1, filters2, filters3 = filters

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'
    x = conv2d(input_tensor, filters1, (1, 1), strides=strides, padding='SAME', name=conv_name_1, w_init=kernel_initializer)
    x = bn(x, is_train=is_training, name=bn_name_1, act=tf.nn.relu)

    group_name_2 = 'group' + str(stage) + '_' + str(block) + '_3x3'

    x = group_conv(x, filters2, kernel_size=(3,3), strides=(1,1), act=tf.nn.relu, cardinality=cardinality, is_train=is_training, name=group_name_2)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
    x = conv2d(x, filters3, (1,1), strides=(1,1), name=conv_name_3, padding='SAME', w_init=kernel_initializer)
    x = bn(x, is_train=is_training, name=bn_name_3, act=tf.nn.relu)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
    bn_name_4   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_shortcut'
    shortcut = conv2d(input_tensor, filters3, (1,1), strides=strides, padding='SAME', name=conv_name_4, w_init=kernel_initializer)
    shortcut = bn(shortcut, is_train=is_training, name=bn_name_4)

    x = x + shortcut
    return tf.nn.relu(x)


def get_resnext(input_tensor, block, is_training, reuse, cardinality, kernel_initializer=None):
    # 3, 4, 16, 3
    with tf.variable_scope('scope', reuse=reuse):
        # x = tl.layers.InputLayer(input_tensor, name='inputs')
        x = conv2d(input_tensor, 16, (3,3), strides=(1,1), padding='SAME', w_init=kernel_initializer, name='face_conv1_1/3x3_s1')
        x = bn(x, is_train=is_training, name='face_bn1_1/3x3_s1', act=tf.nn.relu)

        x = conv_block_2d(x, 3, [32, 32, 64], stage=2, block='face_1a', strides=(1,1), cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
        for first_block in range(block[0] - 1):
            x = identity_block2d(x, 3, [64, 64, 64], stage='1b_{}'.format(first_block), block='face_{}'.format(first_block), cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

        x = conv_block_2d(x, 3, [64, 64, 128], stage=3, block='face_2a', cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
        for second_block in range(block[1] - 1):
            x = identity_block2d(x, 3, [32, 32, 128], stage='2b_{}'.format(second_block), block='face_{}'.format(second_block), cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

        x = conv_block_2d(x, 3, [128, 128, 256], stage=4, block='face_3a' , cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
        for third_block in range(block[2] - 1):
            x = identity_block2d(x, 3, [64, 64, 256], stage='3b_{}'.format(third_block), block='face_{}'.format(third_block), cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

        x = conv_block_2d(x, 3, [256, 256, 512], stage=5, block='face_4a', cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
        for fourth_block in range(block[3] - 1):
            x = identity_block2d(x, 3, [128, 128, 512], stage='4b_{}'.format(fourth_block), block='face_{}'.format(fourth_block), cardinality=cardinality, is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

        # pooling_output = tf.layers.max_pooling2d(x4, (7,7), strides=(1,1), name='mpool2')
        print('before gap: ', x)

        pooling_output = tf.reduce_mean(x, name='gap', axis=[1, 2])

    return pooling_output



def resnext(input_tensor, size, is_training, reuse, cardinality=32, kernel_initializer=tf.orthogonal_initializer()):
    if size == 50:
        blocks = [3,4,6,3]
    elif size == 26:
        blocks = [2,2,2,2]
    elif size == 14:
        blocks = [1,1,1,1]
    else:
        pass

    return get_resnext(input_tensor, blocks, is_training, reuse, cardinality=cardinality, kernel_initializer=kernel_initializer)




# def resnext50(input_tensor, is_training, reuse, cardinality, kernel_initializer=None):
#     return get_resnet(input_tensor, [3,4,6,3], is_training, reuse, cardinality, kernel_initializer)
#
# def resnext110(input_tensor, is_training, reuse, cardinality, kernel_initializer=None):
#     return get_resnet(input_tensor, [3,4,23,3], is_training, reuse, cardinality, kernel_initializer)
#
# def resnext152(input_tensor, is_training, reuse, cardinality, kernel_initializer=None):
#     return get_resnet(input_tensor, [3,8,36,3], is_training, reuse, cardinality, kernel_initializer)
#


if __name__ == '__main__':
    example_data = [np.random.rand(32, 32, 3)]
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = resnext152(x, is_training=True, reuse=False, cardinality=32)
    print(y)

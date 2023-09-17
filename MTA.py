import tensorflow as tf
from tensorflow.python.platform import flags
from networks import ResNet, SeResNet, ResNext, DenseNet, MobileNetV1, MobileNetV2, MobileNetV3, \
    ShuffleNetV1, ShuffleNetV2, SqueezeNet, ResNet13

FLAGS = flags.FLAGS

class Meta_Transfer_Attack:
    def __init__(self, dim_input=1, dim_output=1):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.eps = tf.placeholder_with_default(FLAGS.eps, ())

        self.MSM = ResNet13(0)

        print('The substitute model is',self.MSM.__class__.__name__)

        self.source_nets = [ResNet(18), ResNet(34), ResNet(10), SeResNet(14),
                                  SeResNet(26), SeResNet(50), MobileNetV1(0), MobileNetV2(0)]
        self.target_nets = [MobileNetV3(0), ShuffleNetV1(0), ShuffleNetV2(0), SqueezeNet('A'), SqueezeNet('B')]

        self.loss_func = tf.nn.softmax_cross_entropy_with_logits_v2

        shape = [None, 32, 32, 3]
        self.image = tf.placeholder(tf.float32, shape=shape)
        shape = [None, FLAGS.num_classes]
        self.label = tf.placeholder(tf.float32, shape=shape)

        self.target_net_vars = {}
        self.target_vars = []
        self.target_loaders = {}
        self.test_clean_outputs = {}
        self.test_attack_outputs = {}


    def construct_training_graph(self, update_steps):
        attack = self.image
        with tf.variable_scope('Substitute', reuse=tf.AUTO_REUSE) as training_scope:
            for i in range(update_steps):
                output = self.MSM.forward(attack, True)
                if i == 0:
                    self.MSM_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Substitute')
                loss = self.loss_func(logits=output, labels=self.label)
                grad = tf.gradients(loss, attack)[0]
    
                abs_grad = tf.abs(grad)
                l1_norm = tf.reduce_sum(abs_grad, axis=[1, 2, 3], keep_dims=True)
                grad_1 = grad / l1_norm
    
                mean_abs_grad = tf.reduce_mean(abs_grad, axis=[1, 2, 3], keep_dims=True)
                norm_one_grad = grad / mean_abs_grad
                grad_atan = tf.atan(norm_one_grad) * (2 / 3.1415926)
    
                grad_sign = tf.sign(grad)
                norm_grad = grad_1 + 0.01 * grad_sign + 0.01 * grad_atan
    
                attack = attack + self.eps * norm_grad  / update_steps
                attack = tf.clip_by_value(attack, 0.0, 1.0)
        
            self.training_attack = attack
            
            distortion = attack - self.image
            l1_distortion = tf.abs(distortion)
            l_inf_distortion = tf.reduce_max(l1_distortion, axis=[1, 2, 3])
            self.training_l_inf_distortion_all = tf.reduce_mean(l_inf_distortion)

        self.source_loss1 = 0                     # loss on the clean images
        self.source_accuracy1 = 0
        self.source_loss2 = 0                     # loss on the adversarial images
        self.source_accuracy2 = 0

        with tf.variable_scope('Target', reuse=tf.AUTO_REUSE) as scope:
            for net in self.source_nets:
                net_name = net.__class__.__name__
                net_size = net.size
                with tf.variable_scope(net_name + str(net_size), reuse=tf.AUTO_REUSE) as scope:
                    output1 = net.forward(self.image, False)
                    accuracy1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(output1), 1), tf.argmax(self.label, 1))
                    self.source_accuracy1 += accuracy1
                    loss1 = self.loss_func(logits=output1, labels=self.label)
                    self.source_loss1 += tf.reduce_mean(loss1)

                scope = 'Target/' + net_name + str(net_size)
                vars = tf.get_collection(key=tf.GraphKeys.VARIABLES, scope=scope)
                self.target_net_vars[scope] = vars
                self.target_vars.extend(vars)
                self.target_loaders[scope] = tf.train.Saver(vars, max_to_keep=0)

                with tf.variable_scope(net_name + str(net_size), reuse=tf.AUTO_REUSE) as scope:
                    output2 = net.forward(self.training_attack, False)
                    accuracy2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(output2), 1), tf.argmax(self.label, 1))
                    self.source_accuracy2 += accuracy2
                    loss2 = self.loss_func(logits=output2, labels=self.label)
                    self.source_loss2 += tf.reduce_mean(loss2)

        self.source_accuracy1 = self.source_accuracy1 / len(self.source_nets)
        self.source_loss1 = self.source_loss1 / len(self.source_nets)
        self.source_accuracy2 = self.source_accuracy2 / len(self.source_nets)
        self.source_loss2 = self.source_loss2 / len(self.source_nets)


    def construct_testing_graph(self, update_steps):
        attack = self.image
        with tf.variable_scope('Substitute', reuse=tf.AUTO_REUSE) as training_scope:

            for i in range(update_steps):
                output = self.MSM.forward(attack, True)
                if 'MSM_vars' not in dir(self):
                    if i == 0:
                        self.MSM_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Substitute')
                        
                loss = self.loss_func(logits=output, labels=self.label)

                grad = tf.gradients(loss, attack)[0]
                norm_grad = tf.sign(grad)

                attack = attack + self.eps * norm_grad / update_steps
                attack = tf.clip_by_value(attack, 0.0, 1.0)

            distortion = attack - self.image
            l1_distortion = tf.abs(distortion)
            l_inf_distortion = tf.reduce_max(l1_distortion, axis=[1, 2, 3])
            self.testing_l_inf_distortion_all = tf.reduce_mean(l_inf_distortion)


        self.target_loss1 = 0
        self.target_loss2 = 0
        self.target_accuracy1 = 0
        self.target_accuracy2 = 0
        with tf.variable_scope('Target', reuse=tf.AUTO_REUSE) as scope:
            for net in self.target_nets:
                net_name = net.__class__.__name__
                net_size = net.size
                with tf.variable_scope(net_name + str(net_size), reuse=tf.AUTO_REUSE) as scope:
                    output1 = net.forward(self.image, False)
                    self.test_clean_outputs[net_name + str(net_size)] = tf.nn.softmax(output1)
                    accuracy1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(output1), 1),
                                                            tf.argmax(self.label, 1))
                    self.target_accuracy1 += accuracy1
                    loss1 = self.loss_func(logits=output1, labels=self.label)
                    self.target_loss1 += tf.reduce_mean(loss1)
        
                scope = 'Target/' + net_name + str(net_size)
                vars = tf.get_collection(key=tf.GraphKeys.VARIABLES, scope=scope)
                self.target_net_vars[scope] = vars
                self.target_vars.extend(vars)
                self.target_loaders[scope] = tf.train.Saver(vars, max_to_keep=0)
       
                if 'SqueezeNet' in scope:
                    net.conv_num = 0
 
                with tf.variable_scope(net_name + str(net_size), reuse=tf.AUTO_REUSE) as scope:
                    output2 = net.forward(attack, False)
                    self.test_attack_outputs[net_name + str(net_size)] = tf.nn.softmax(output2)
                    accuracy2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(output2), 1), tf.argmax(self.label, 1))
                    self.target_accuracy2 += accuracy2
                    loss2 = self.loss_func(logits=output2, labels=self.label)
                    self.target_loss2 += tf.reduce_mean(loss2)

        self.target_loss1 = self.target_loss1 / len(self.target_nets)
        self.target_loss2 = self.target_loss2 / len(self.target_nets)
        self.target_accuracy1 = self.target_accuracy1 / len(self.target_nets)
        self.target_accuracy2 = self.target_accuracy2 / len(self.target_nets)

        

    def construct_optimizing_graph(self):
        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        gvs = optimizer.compute_gradients(-self.source_loss2, self.MSM_vars)
        self.gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
        self.metatrain_op = optimizer.apply_gradients(self.gvs)


    def load_target_models(self, sess):
        if FLAGS.train:
            target_models = self.source_nets
        else:
            target_models = self.target_nets

        for model in target_models:
            model_name = model.__class__.__name__ + str(model.size)
            if model.__class__.__name__ == 'ResNet':
                model_path = '../../target_models/ResNet' + str(model.size)
                model_name = 'ResNet' + str(model.size)

            elif model.__class__.__name__ == 'SeResNet':
                model_path = '../../target_models/SeResNet' + str(model.size)
                model_name = 'SeResNet' + str(model.size)

            elif model.__class__.__name__ == 'DenseNet':
                if not model.BC:
                    model_path = '../../target_models/DenseNet' + str(model.size)
                    model_name = 'DenseNet' + str(model.size)
                else:
                    model_path = '../../target_models/DenseNet' + str(model.size) + 'BC'
                    model_name = 'DenseNet' + str(model.size)


            elif model.__class__.__name__ == 'SqueezeNet':
                model_path = '../../target_models/SqueezeNet' + str(model.size)
                model_name = 'SqueezeNet' + str(model.size)

            elif model.__class__.__name__ == 'MobileNetV1':
                model_path = '../../target_models/Mobile1'
            elif model.__class__.__name__ == 'MobileNetV2':
                model_path = '../../target_models/Mobile2'
            elif model.__class__.__name__ == 'MobileNetV3':
                model_path = '../../target_models/Mobile3'
            elif model.__class__.__name__ == 'ShuffleNetV1':
                model_path = '../../target_models/Shuffle1'
            elif model.__class__.__name__ == 'ShuffleNetV2':
                model_path = '../../target_models/Shuffle2'
            else:
                pass
            print(model_path)
            self.target_loaders['Target/'+model_name].restore(sess, model_path)






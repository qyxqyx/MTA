import tensorflow as tf
import datetime
from MTA import Meta_Transfer_Attack
from tensorflow.python.platform import flags
from dataset import Dataset_CF
from train_eval import train, test

FLAGS = flags.FLAGS

flags.DEFINE_integer('train_iterations', 47000, 'number of training iterations.')
flags.DEFINE_integer('num_classes', 10, 'number of classes')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_float('meta_lr', 0.001, 'the learning rate of the MSM')
flags.DEFINE_string('network', 'ResNet13', 'network name')
flags.DEFINE_integer('base_num_filters', 64, 'number of filters for the network')
flags.DEFINE_bool('data_aug', True, 'Whether or not to use data augmentation')

flags.DEFINE_string('logdir', 'logs/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')

flags.DEFINE_float('eps_c', 1600, 'the initial epsilon for the Customized PGD')
flags.DEFINE_integer('T_train', 7, 'number of iteration of the Customized PGD')
flags.DEFINE_float('eps', 15, 'epsilon for PGD')
flags.DEFINE_integer('T_test', 10, 'number of iteration of PGD')
flags.DEFINE_integer('attack_decay_iter', 4000, 'the interval that eps_c should be decayed')

FLAGS.eps_c/=255
FLAGS.eps/=255


def main():
    FLAGS.logdir = 'logs/Cifar' + str(FLAGS.num_classes) + '/'

    data_generator = Dataset_CF()
    model = Meta_Transfer_Attack()
    
    if FLAGS.train:
        print('create training graph')
        model.construct_training_graph(FLAGS.T_train)
        model.construct_optimizing_graph()
    else:
        print('create testing graph')
        model.construct_testing_graph(FLAGS.T_test)

    substitute_vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='Substitute')
    saver = tf.train.Saver(substitute_vars, max_to_keep=0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    experiment_settings = str(FLAGS.network)
    experiment_settings += '.lr_' + str(FLAGS.meta_lr)
    experiment_settings += '.nfs_' + str(FLAGS.base_num_filters)
    experiment_settings += '.DA_' + str(FLAGS.data_aug)[0]
    experiment_settings += '.alr_' + str(FLAGS.eps_c) + '.up_' + str(FLAGS.T_train)
    experiment_settings += '.adecay_' + str(FLAGS.attack_decay_iter)
 
    resume_itr = 0

    tf.global_variables_initializer().run()
    print('loading target models')
    model.load_target_models(sess=sess)

    if FLAGS.resume:
        if FLAGS.test_iter > 0:
            model_file = FLAGS.logdir + experiment_settings + '/model' + str(FLAGS.test_iter)
        else:
            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + experiment_settings)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)


    if FLAGS.train:
        train(model, saver, sess, experiment_settings, data_generator, resume_itr)
    else:
        import os
        models = os.listdir(FLAGS.logdir + experiment_settings)
        model_epochs = []
        for model_file in models:
            if 'model' in model_file and 'index' in model_file:
                i = model_file.find('del')
                j = model_file.find('.')
                model_epoch = model_file[i + 3:j]
                model_epochs.append(int(model_epoch))

        model_epochs.sort()
        Min_acc = 1.0
        max_epoch = 0
        for epoch in model_epochs:
            if epoch >= 1000:
                model_file = FLAGS.logdir + experiment_settings + '/model' + str(epoch)
                saver.restore(sess, model_file)
                print(str(datetime.datetime.now())[:-7], "testing model: " + model_file)
                mean_acc = test(model, sess, data_generator, epoch)
                if mean_acc < Min_acc:
                    Min_acc = mean_acc
                    max_epoch = epoch
                print('----------min_acc:', Min_acc, '-----------max_model:', max_epoch)

            else:
                pass


if __name__ == "__main__":
    main()






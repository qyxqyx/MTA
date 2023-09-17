import numpy as np
import datetime
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def train(model, saver, sess, experiment_settings, data_generator, resume_itr=0):
    PRINT_INTERVAL = 50
    CHECKPOINT_INTERVAL = 200


    print(experiment_settings)
    print('Done initializing, starting training.')

    source_losses1, source_accs1, source_losses2, source_accs2, l_inf_distortions_all = [], [], [], [], []

    for itr in range(resume_itr, FLAGS.train_iterations):
        feed_dict = {model.meta_lr: FLAGS.meta_lr}

        batch_images, batch_labels, _ = data_generator.get_batch_data(FLAGS.batch_size, train=True)
        feed_dict[model.image] = batch_images
        feed_dict[model.label] = batch_labels
 
        eps_c = FLAGS.eps_c * (0.9 ** int(itr / FLAGS.attack_decay_iter))
        if int(itr % FLAGS.attack_decay_iter) < 2:
            print('change the attack step size to:' + str(eps_c) + ', ----------------------------')
    
        feed_dict[model.eps] = eps_c

        input_tensors = [model.metatrain_op]
        input_tensors.extend([model.source_loss1, model.source_accuracy1])
        input_tensors.extend([model.source_loss2, model.source_accuracy2])
        input_tensors.extend([model.training_l_inf_distortion_all])

        result = sess.run(input_tensors, feed_dict)

        source_losses1.append(result[1])
        source_accs1.append(result[2])
        source_losses2.append(result[3])
        source_accs2.append(result[4])
        l_inf_distortions_all.append(result[5])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ':::' + str('%.4f' %np.mean(source_losses1)) + ', ' + str('%.4f' %np.mean(source_accs1))
            print_str += ', ' + str('%.4f' %np.mean(source_losses2)) + ', ' + str('%.4f' %np.mean(source_accs2))
            print_str += ':::' + str('%.4f' %np.mean(l_inf_distortions_all))

            print(str(datetime.datetime.now())[:-7], print_str)
            source_losses1, source_accs1, source_losses2, source_accs2, l_inf_distortions_all = [], [], [], [], []
            
        if (itr!=0) and itr % CHECKPOINT_INTERVAL == 0:
            model_name = FLAGS.logdir + '/' + experiment_settings + '/model' + str(itr)
            saver.save(sess, model_name)
        
    saver.save(sess, FLAGS.logdir + '/' + experiment_settings +  '/model' + str(itr))
    
    

def test(model, sess, data_generator, itr):
    target_losses1, target_accs1, target_losses2, target_accs2, l_inf_distortions_all = [], [], [], [], []
    
    val_clean_outputs = {}
    val_attack_outputs = {}
    val_labels = []
    
    val_end = False
    while not val_end:
        
        val_batch_images, val_batch_labels, val_end = data_generator.get_batch_data(FLAGS.batch_size, train=False)
        val_labels.append(val_batch_labels)
        
        val_feed_dict = {model.meta_lr: 0}
        val_feed_dict[model.image] = val_batch_images
        val_feed_dict[model.label] = val_batch_labels
        val_feed_dict[model.eps] = FLAGS.eps
        
        input_tensors = [model.target_loss1, model.target_loss2]
        input_tensors.extend([model.testing_l_inf_distortion_all])
        input_tensors.extend([model.test_clean_outputs, model.test_attack_outputs])
        
        result = sess.run(input_tensors, val_feed_dict)
        target_losses1.append(result[0])
        target_losses2.append(result[1])
        l_inf_distortions_all.append(result[2])
        
        clean_output = result[3]
        for key, value in clean_output.items():
            if key not in val_clean_outputs.keys():
                val_clean_outputs[key] = [value]
            else:
                val_clean_outputs[key].append(value)
        
        attack_output = result[4]
        for key, value in attack_output.items():
            if key not in val_attack_outputs.keys():
                val_attack_outputs[key] = [value]
            else:
                val_attack_outputs[key].append(value)
    
    val_labels = np.concatenate(val_labels, axis=0)
    val_labels = np.argmax(val_labels, axis=1)
    val_clean_accuracy = {}
    val_attack_accuracy = {}
    attack_accuracies = []
    clean_accuracies = []
    print_clean_accuracy_str =  'Validation  clean accuracy' + str(itr)
    print_attack_accuracy_str = 'Validation attack accuracy' + str(itr)
    print_attack_success_str =  'Validation attack  success' + str(itr)
    
    for key in val_clean_outputs.keys():
        clean_value = np.concatenate(val_clean_outputs[key], axis=0)
        clean_predict = np.argmax(clean_value, axis=1)
        clean_predict_right = val_labels == clean_predict
        clean_accuracy = np.mean(clean_predict_right)
        
        val_clean_accuracy[key] = clean_accuracy
        clean_accuracies.append(clean_accuracy)
        print_clean_accuracy_str += ': ' + str('%.4f' % clean_accuracy)
        
        attack_value = np.concatenate(val_attack_outputs[key], axis=0)
        attack_predict = np.argmax(attack_value, axis=1)
        attack_predict_right = val_labels == attack_predict
        attack_right_on_clean_right = attack_predict_right * clean_predict_right
        attack_right_on_clean_right_rate = np.sum(attack_right_on_clean_right) / np.sum(clean_predict_right)
        
        val_attack_accuracy[key] = attack_right_on_clean_right_rate
        attack_accuracies.append(attack_right_on_clean_right_rate)
        print_attack_accuracy_str += ': ' + str('%.4f' % attack_right_on_clean_right_rate)
        print_attack_success_str += ': ' + str('%.4f' % (1-attack_right_on_clean_right_rate) )
        
    mean_acc_clean = np.mean(clean_accuracies)
    std_acc_clean = np.std(clean_accuracies)
    ci95_clean = 1.96 * std_acc_clean / np.sqrt(len(clean_accuracies))
    mean_acc_attack = np.mean(attack_accuracies)
    std_acc_attack = np.std(attack_accuracies)
    ci95_attack = 1.96 * std_acc_attack / np.sqrt(len(attack_accuracies))
    
    print_clean_accuracy_str += ': ' + str('%.4f' % mean_acc_clean) + ', ' + str('%.4f' % ci95_clean)
    print_attack_accuracy_str += ': ' + str('%.4f' % mean_acc_attack) + ', ' + str('%.4f' % ci95_attack)
    
    print_str = 'Validation ' + str(itr)
    print_str += ':::' + str('%.4f' % np.mean(target_losses1)) + ', ' + str('%.4f' % np.mean(target_losses2)) \
                 + ':::' + str('%.4f' % (np.mean(l_inf_distortions_all)*255))
    
    print('------------------------------------------', itr)
    print(print_str)
    print(print_clean_accuracy_str)
    print(print_attack_accuracy_str)
    print(print_attack_success_str)
    print('------------------------------------------')
    
    return mean_acc_attack







"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
import tensorflow as tf

def lr_scheduler(current_epoch, total_epochs):
    global max_lr, transition_epoch
    max_lr = max_lr
    min_lr = max_lr/transition_epoch

    # annealing lr get from tf log graph and change an_start_lr
    an_start_lr = 0.08
    an_end_lr = 0.001

    an_start_epochs = 24
    an_end_epochs = total_epochs

    global transition_epoch
    if current_epoch <= transition_epoch:
        m = (max_lr - min_lr) / (transition_epoch - 0)
        new_lr = m * current_epoch + min_lr

    elif current_epoch < an_start_epochs:
        m = (max_lr - an_start_lr) / (transition_epoch - an_start_epochs)
        new_x = (current_epoch - an_start_epochs)
        new_lr = m * new_x + an_start_lr
    else:
        m = (an_end_lr - an_start_lr) / (an_end_epochs - an_start_epochs)
        new_x = (current_epoch - an_end_epochs)
        new_lr = m * new_x + an_end_lr

    tf.summary.scalar('learning rate', data=new_lr, step=current_epoch)
    return new_lr
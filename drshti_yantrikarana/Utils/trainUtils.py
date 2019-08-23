"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
import tensorflow as tf
from config import model_config


def OneCycleLrScheduler(current_epoch, ):
    total_epochs = model_config['epochs']
    print(f"!!Total Epochs used in LR schedule is {total_epochs}")
    max_lr = model_config['maxLr']
    min_lr = model_config['minLr']
    transition_epoch = model_config['transitionEpoch']
    print(f"!!Using transition epoch of {transition_epoch} in LR schedule")
    print(f"!!Using max_lr of {max_lr} in LR schedule")
    # min_lr = max_lr/transition_epoch

    # annealing lr get from tf log graph and change an_start_lr
    an_start_lr = model_config['an_start_lr']
    an_end_lr = model_config['an_end_lr']

    an_start_epochs = model_config['an_start_epochs']
    an_end_epochs = total_epochs

    # global transition_epoch
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
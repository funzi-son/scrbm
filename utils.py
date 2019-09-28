from tensorflow.python.ops.nn_ops import softmax
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
import os

def length(x):
    # Lengths of each sequences
    mask = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2)) 
    lens = tf.cast(tf.reduce_sum(mask,reduction_indices=1),tf.int32)
    return lens

def accuracy(o,y,length,mask):
    corrects = tf.equal(o,y)
    corrects = tf.cast(corrects,tf.float32)
    corrects*=mask

    corrects = tf.reduce_sum(corrects,reduction_indices=1)
    corrects/= tf.cast(length,tf.float32)
    return tf.reduce_mean(corrects)

def cross_entropy_with_logits(o,y,length):
    #o: distribution vectors
    #y: one-hot vector
    x_entr = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=o)

    x_entr = tf.reduce_sum(x_entr,reduction_indices=1)
    x_entr /= tf.cast(length,tf.float32)
    return tf.reduce_mean(x_entr)


def negative_log_likelihood(o,y,length):
    #o: 
    #y: one hot vector
    o_ = tf.exp(o)
    llh= tf.reduce_sum(o*y,axis=2) - tf.log(tf.reduce_sum(o_,axis=2))

    llh = tf.reduce_sum(llh,reduction_indices=1)
    llh/=tf.cast(length,tf.float32)
    return -tf.reduce_mean(llh)

def save_process(ckp_name,log_data):
    if ckp_name:
        np.savetxt(ckp_name+"_process.out",log_data,delimiter=',')

def load_process(ckp_name):
    if os.path.isfile(ckp_name+"_process.out"):
        data_log = np.loadtxt(ckp_name+"_process.out", delimiter=',')
        #lr,epoch,max_vld_val,per_dec_count
        print(len(data_log.shape))
        return data_log
    else:
        return 0,0,0,0
    return lr,epoch,max_vld_val,per_dec_count

def evaluate(model,session,dataset,nllh,pred,eval_type="validation"):
    eval_acc  = 0
    eval_nllh = 0
    eval_f1   = 0
    scount = 0
    while True:
        if eval_type=="train":
            x_,y_ = dataset.next()
        if eval_type=="validation":
            x_,y_ = dataset.next_valid()
        elif eval_type=="test":
            x_,y_ = dataset.next_test()
        if x_ is None:
            break

        l = x_.shape[1]
        pred_ = session.run(pred,{model.x:x_,model.l:[l]})
        nllh_ = session.run(nllh,{model.x:x_,model.y:y_,model.l:[l]})

        y_    = np.argmax(y_,axis=2)
        
        eval_nllh+= nllh_

        acc_ = eval_accuracy(pred_,y_)
        f1_  = f1(pred_,y_)

        eval_acc += acc_
        eval_f1  += f1_
        scount+=1
        
    
    return eval_nllh/scount, eval_acc/scount, eval_f1/scount

def eval_accuracy(pred,y_):
    return np.mean(pred==y_)

def f1(pred,y_):
    return f1_score(y_[0,:],pred[0,:],average="micro")
    

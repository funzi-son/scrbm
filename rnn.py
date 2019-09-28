'''
The Recurrent Neural Networks
Son N. Tran
sontn.fz@gmail.com
'''
import time
import os
import numpy as np
import tensorflow as tf

from utils import *

if int(tf.__version__[0])==1:
    BasicRNNCell = tf.contrib.rnn.BasicRNNCell
    LSTMCell = tf.contrib.rnn.LSTMCell
    GRUCell = tf.contrib.rnn.GRUCell
else:
    BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
    LSTMCell = tf.nn.rnn_cell.LSTMCell
    GRUCell = tf.nn.rnn_cell.GRUCell
    
class RNN(object):
    def __init__(self,conf,dataset,model_type='rnn'):
        self.conf = conf
        self.model_type = model_type
        self.dataset = dataset
        self.max_len = dataset.get_max_len()
        self.ckp_name = conf.ckp_file

    def build_model(self):
        hidNum = self.conf.hidNum
        visNum = self.dataset.sensor_num()
        labNum = self.dataset.total_combined_acts()
        
        self.x = tf.placeholder(tf.float32,[None,self.max_len,visNum])
        self.y = tf.placeholder(tf.float32,[None,self.max_len,labNum])
        self.lr = tf.placeholder(tf.float32,shape=[]) 
        # This is useful for batch
        mask  = tf.sign(tf.reduce_max(tf.abs(self.y),reduction_indices=2))
        lens = length(self.x)
    
        # Choose for cell type
        if self.conf.cell_type     == "BasicRNN":
            mycell  = BasicRNNCell
        elif self.conf.cell_type   == "LSTM":
            mycell  = LSTMCell
        elif self.conf.cell_type   == "GRU":
            mycell = GRUCell        
        else:
            raise ValueError("cell type is not specified!!!")
        
        s,_ = tf.nn.dynamic_rnn(
            mycell(hidNum),
            self.x,
            dtype=tf.float32,
            sequence_length=lens
        )

        s = tf.reshape(s,[-1,hidNum])
        
        with tf.variable_scope("softmax_layer"):
            weights = tf.get_variable("softmax_w",[hidNum,labNum],initializer=tf.truncated_normal_initializer(stddev=1e-1))
            biases  = tf.get_variable("softmax_b",[labNum],initializer=tf.constant_initializer(0.0))
            
        o = tf.matmul(s,weights)+biases
        o = tf.reshape(o,[-1,self.max_len,labNum])
        pred = tf.argmax(o,2)

        #acc = accuracy(pred,tf.argmax(self.y,2),lens,mask)
        nllh = negative_log_likelihood(o,self.y,lens,mask)
        if self.conf.obj_func ==   "xen": # Cross entropy
            cost = cross_entropy_with_logits(o,self.y,lens,mask)
        elif self.conf.obj_func == "llh": # log-likelihood
            cost = nllh
            
        ##### regularization --> dont use now
#        l2 =  self.conf.weight_decay*sum(tf.nn.l2_loss(tf_var)
#                                            for tf_var in tf.trainable_variables()
#                                            if not ("Bias" in tf_var.name or "softmax_b" in tf_var.name))
#        cost += l2

        return nllh,cost,pred
    
    def run(self):
        with tf.Graph().as_default():            
            # Build graph
            nllh,loss,pred = self.build_model()
            if tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint process found, loading hyper-params...")
                lr,epoch,max_vld_val,per_dec_count = load_process(self.ckp_name)
                if lr==0:
                    lr = self.conf.lr
            else:
                lr  = self.conf.lr
                epoch = max_vld_val =  per_dec_count = 0

            if self.conf.opt=="sgd":
                assert (self.conf.batch_size == 1),"Batch size must be 1!"
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                # use self.lr for manual decay during learning
            elif self.conf.opt=="mini-gd":
                assert (self.conf.batch_size > 1),"Batch size must be bigger than 1!"
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                # use self.lr for manual decay during learning
            elif self.conf.opt=="adam":
                optimizer = tf.train.AdamOptimizer(self.conf.lr)
            elif self.conf.opt=="rmsprop":
                optimizer = tf.train.RMSPropOptimizer(self.conf.lr)
                            
            tvars    = tf.trainable_variables()
            grads    = tf.gradients(loss,tvars) 
            train_op = optimizer.apply_gradients(zip(grads,tvars))
    
            init     = tf.global_variables_initializer()
            saver    = tf.train.Saver()

            # If not running with GPU
            if self.conf.computation=="single_thread":
                config = tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1
                    # device_count = {'CPU': 1}
                )
                session = tf.Session(config=config)
            else:
                session = tf.Session()

            ######## Training
            running = True
            total_err = 0
            
            if tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint found, loading checkpoint...")
                saver.restore(session,self.ckp_name)
            else:
                session.run(init)
                
            while running:
                if not self.dataset.eof:
#                    start_time = time.time()
                    batch_x,batch_y = self.dataset.next_seq_vec_batch(batch_size=self.conf.batch_size)
                    inpdict = {self.x:batch_x,self.y:batch_y,self.lr:lr}
                    _,err = session.run([train_op,loss],inpdict)
                    total_err += err
                else:
                    epoch +=1
                    self.dataset.rewind()
                    total_err = 0
                    # Evaluate on evaluation set
                    vld_nllh, vld_acc, vld_f1 = evaluate(self,session,self.dataset,nllh,pred)
                    if self.conf.eval_metric=="nllh":# Negative Log-likelihood
                        vld_val = vld_nllh
                    elif self.conf.eval_metric=="accuracy":
                        vld_val = vld_acc
                    elif self.conf.eval_metric=="f1":
                        vld_val = vld_f1
                        
                    #EARLY STOPPING
                    if vld_val>=max_vld_val:
                        max_vld_val = vld_val
                        saver.save(session,self.ckp_name)
                        per_dec_count = 0
                    else:
                        per_dec_count +=1
                        if self.conf.opt=="sgd":                            
                            lr = lr/(1+self.conf.LR_DECAY_VAL)                                                                           
                    print("[Epoch %d] aac:%.5f nllh:%.5f f1:%.5f (%s):%f %d"  % (epoch,vld_acc,vld_nllh,vld_f1,self.conf.eval_metric,max_vld_val,per_dec_count))                    
                    save_process(self.ckp_name,lr,epoch,max_vld_val,per_dec_count)
                    
                    if per_dec_count >= self.conf.DEC_NUM_4_EARLY_STOP or epoch >= self.conf.MAX_ITER:
                        running = False

            ##### Evaluation on validation set
            saver.restore(session,self.ckp_name)
            vld_nllh, vld_acc, vld_f1 = evaluate(self,session,self.dataset,nllh,pred)            
            ##### Evaluation on test set
            if self.dataset.is_test:                
                tst_nllh, tst_acc, tst_f1 = evaluate(self,session,self.dataset,nllh,pred,eval_type="test")
            else:
                tst_nllh =  tst_acc =  tst_f1 = 0
                
            return vld_acc,vld_nllh,vld_f1,tst_acc,tst_nllh,tst_f1

    """
    def evaluate(self,session,acc,eval_type="validation",batch_size=1):
        avg_acc = 0
        scount = 0
        while True:
            if eval_type=="validation":
                x_,y_ = self.dataset.valid_seq_vec_batch(batch_size)
            elif eval_type=="test":
                x_,y_ = self.dataset.test_seq_vec_batch(batch_size)
            if x_ is None:
                break
            acc_ = session.run([acc],{self.x:x_,self.y:y_})
            avg_acc += acc_[0]*x_.shape[0]
            scount+=x_.shape[0]
        #print(scount)
        return avg_acc/scount            
    """

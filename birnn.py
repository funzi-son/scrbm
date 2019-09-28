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
    
class BiRNN(object):
    def __init__(self,conf,dataset):
        self.conf = conf
        self.dataset = dataset
        self.max_len = dataset.get_max_len()
        self.ckp_name = conf.ckp_file

    def build_model(self):
        hidNum = self.conf.hidNum
        visNum = self.dataset.sensor_num()
        labNum = self.dataset.total_combined_acts()
        
        self.x = tf.placeholder(tf.float32,[1,None,visNum])
        self.y = tf.placeholder(tf.float32,[1,None,labNum])
        self.l = tf.placeholder(tf.int32,shape=[1]) 
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

        cell_fw = mycell(hidNum)
        cell_bw = mycell(hidNum)
        (bi_h,_) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            self.x,
            dtype=tf.float32,
            sequence_length=self.l
        )
        
        s = tf.concat(bi_h,2)
        s = tf.reshape(s,[-1,2*hidNum])

        
        with tf.variable_scope("softmax_layer"):
            weights = tf.get_variable("softmax_w",[2*hidNum,labNum],initializer=tf.truncated_normal_initializer(stddev=1e-1))
            biases  = tf.get_variable("softmax_b",[labNum],initializer=tf.constant_initializer(0.0))
            
        o = tf.matmul(s,weights)+biases
        o = tf.reshape(o,[1,-1,labNum])
        pred = tf.argmax(o,2)
    
        nllh = negative_log_likelihood(o,self.y,self.l)
        if self.conf.obj_func=="xen": #Cross entropy
            cost = cross_entropy_with_logits(o,self.y,self.l)
        elif self.conf.obj_func=="llh":
            cost = nllh
        else:
            raise ValueError("Opt type must be specified")
                
        
        return nllh,cost,pred

    def run(self):
        with tf.Graph().as_default():
            #Build graph
            nllh,loss,pred = self.build_model()
            if tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint process found, loading hyper-params...")
                lr,epoch,max_vld_val,max_epoch,per_dec_count = load_process(self.ckp_name)
                if lr==0:
                    lr = self.conf.lr
            else:
                lr  = self.conf.lr
                epoch = max_vld_val=  per_dec_count = 0

            if self.conf.opt=="sgd":
                assert (self.conf.batch_size == 1),"Batch size must be 1!"
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
                # use self.lr for manual decay during learning
            elif self.conf.opt=="gd":
                assert (self.conf.batch_size > 1),"Batch size must be bigger than 1!"
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
                # use self.lr for manual decay during learning
            elif self.conf.opt=="adam":
                optimizer = tf.train.AdamOptimizer(lr)
            elif self.conf.opt=="rmsprop":
                optimizer = tf.train.RMSPropOptimizer(lr)
            elif self.conf.opt=="nadam":
                optimizer = tf.contrib.opt.NadamOptimizer(lr)
            elif self.conf.opt=="adadelta":
                optimizer = tf.train.AdadeltaOptimizer(lr)
            elif self.conf.opt=="adagrad":
                optimizer = tf.train.AdagradOptimizer(lr)
                
            
            tvars   = tf.trainable_variables()
            grads =tf.gradients(loss,tvars)            
            train_op = optimizer.apply_gradients(zip(grads,tvars))
            
            
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            # If not running with GPU
            if self.conf.computation=="single_thread":
                config = tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1
                    # device_count = {"CPU": 1}
                )
                session = tf.Session(config=config)
            else:
                session = tf.Session()

            ######## Training
            running = True
            total_err=0
            if tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint found, loading checkpoint...")
                saver.restore(session,self.ckp_name)
            else:
                session.run(init)

            c = 0
            while running:
                batch_x,batch_y = self.dataset.next()                
                if batch_x is not None:
                    inpdict = {self.x:batch_x,self.y:batch_y,self.l:[batch_x.shape[1]]}             
                    _,err = session.run([train_op,loss],inpdict)
                    
                    #if np.isnan(err) or np.isinf(err):
                    #    raise ValueError("Nan error")
                    total_err += err
                    c+=1
                else:
                    epoch +=1
                    print("%d sample trained"%c)
                    c=0                    
                    
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
                        max_epoch   = epoch
                        if self.ckp_name:
                            saver.save(session,self.ckp_name)                            
                        per_dec_count = 0
                    else:
                        per_dec_count +=1
                        if self.conf.opt=="sgd":
                            lr = lr/(1+self.conf.LR_DECAY_VAL)

                    print("[Epoch %d] err:%.5f aac:%.5f nllh:%.5f f1:%.5f (%s):%f %d"  % (epoch,total_err,vld_acc,vld_nllh,vld_f1,self.conf.eval_metric,max_vld_val,per_dec_count))
                    total_err = 0
                    
                    save_process(self.ckp_name,[lr,epoch,max_vld_val,max_epoch,per_dec_count])
                    
                    if per_dec_count >= self.conf.DEC_NUM_4_EARLY_STOP or epoch >= self.conf.MAX_ITER:
                        running = False

            ##### Evaluation on validation set
            if self.ckp_name:
                saver.restore(session,self.ckp_name)
                
            vld_nllh, vld_acc, vld_f1 = evaluate(self,session,self.dataset,nllh,pred)            
            ##### Evaluation on test set
            if self.dataset.is_test:                
                tst_nllh, tst_acc, tst_f1 = evaluate(self,session,self.dataset,nllh,pred,eval_type="test")
            else:
                tst_nllh =  tst_acc =  tst_f1 = 0
            return vld_acc,vld_nllh,vld_f1,tst_acc,tst_nllh,tst_f1,max_epoch           

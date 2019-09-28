'''
The Recurrent Neural Networks
Son N. Tran
sontn.fz@gmail.com
'''
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import tensorflow as tf
import time

from rtdrbm_cell import *
from utils import *

class RTDRBM(object):
    def __init__(self,conf,dataset,model_type="rtdrbm"):
        self.conf = conf
        self.model_type = model_type
        self.dataset = dataset
        self.max_len = dataset.get_max_len()
        print(self.max_len)
        self.ckp_name = conf.ckp_file
        
    def build_model(self):
        hidNum = self.conf.hidNum
        visNum = self.dataset.sensor_num()
        labNum = self.dataset.total_combined_acts()

        inp_ftr_len = visNum
        dtype = tf.float32
        
        self.x  = tf.placeholder(dtype,[1,self.max_len,inp_ftr_len])
        self.y  = tf.placeholder(tf.float32,[1,self.max_len,labNum])    # for training
        self.lr = tf.placeholder(tf.float32,shape=[])
        
        mask  =  tf.sign(tf.reduce_max(tf.abs(self.y),reduction_indices=2))                 
        lens = length(self.x)

        if self.conf.cell_type=="BasicRTDRBM":
            cell = BasicRTDRBMCell(visNum,labNum,hidNum)
        elif self.conf.cell_type=="RTDRBM":
            cell = RTDRBMCell(visNum,labNum,hidNum)
        
        o,_ = tf.nn.dynamic_rnn(
                cell,
                [self.x,self.y],
                dtype=tf.float32,
                sequence_length=lens
            )

        #pred = tf.argmax(o,2)
        
        #scope = vs.get_variable_scope()
        #print(scope.name)
        #scope.reuse_variables()
        pred,_= tf.nn.dynamic_rnn(
            cell,
            [self.x],
            dtype=tf.float32,
            sequence_length=lens
        )
        pred = tf.argmax(pred,2)
        
        nllh = negative_log_likelihood(o,self.y,lens,mask)
        if self.conf.obj_func=="xen": #Cross entropy
            cost = cross_entropy_with_logits(o,self.y,lens,mask)
        elif self.conf.obj_func=="llh":
            cost = nllh
        else:
            raise ValueError("Opt type must be specified")
        

        #acc = accuracy(pred,tf.argmax(self.y,2),lens,mask)                
        """ Regularization 
        l2 =  self.conf.weight_decay*sum(tf.nn.l2_loss(tf_var)
                                            for tf_var in tf.trainable_variables()
                                            if not ("Bias" in tf_var.name or "softmax_b" in tf_var.name))
        cost += l2
        """
        return nllh,cost,pred
    def run(self):
        with tf.Graph().as_default():
            #Build graph
            nllh,loss,pred = self.build_model()
            if tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint process found, loading hyper-params...")
                lr,epoch,max_vld_val,per_dec_count = load_process(self.ckp_name)
                if lr==0:
                    lr = self.conf.lr
            else:
                lr  = self.conf.lr
                epoch = max_vld_val=  per_dec_count = 0

            if self.conf.opt=="sgd":
                assert (self.conf.batch_size == 1),"Batch size must be 1!"
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                # use self.lr for manual decay during learning
            elif self.conf.opt=="gd":
                assert (self.conf.batch_size > 1),"Batch size must be bigger than 1!"
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                # use self.lr for manual decay during learning
            elif self.conf.opt=="adam":
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.conf.opt=="rmsprop":
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif self.conf.opt=="nadam":
                optimizer = tf.contrib.opt.NadamOptimizer(self.lr)
            elif self.conf.opt=="adadelta":
                optimizer = tf.train.AdadeltaOptimizer(self.lr)
            elif self.conf.opt=="adagrad":
                optimizer = tf.train.AdagradOptimizer(self.lr)
                
            
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
            
            while running:
                if not self.dataset.eof:
                    batch_x,batch_y = self.dataset.next_seq_vec_batch(batch_size=self.conf.batch_size)
                    inpdict = {self.x:batch_x,self.y:batch_y,self.lr:lr}                
                    _,err = session.run([train_op,loss],inpdict)
                    
                    #if np.isnan(err) or np.isinf(err):
                    #    raise ValueError("Nan error")
                    total_err += err
                else:
                    epoch +=1
                    print(total_err)
                    self.dataset.rewind()
                    
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
                    total_err = 0
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
            
                    
            #return 0,0,0,0,0,0

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
            if self.conf.predict_type=="rnn":
                avg_acc += acc_[0]*x_.shape[0]
                scount+=x_.shape[0]
            elif self.conf.predict_type=="viterbi":
                assert (x_.shape[0] ==1),"Cannot apply to batch"
                seq_len = np.sum(np.sign(np.amax(x_,axis=2)),axis=1).astype(int)
                avg_acc = traceback_acc(acc_[0],y_,seq_len)
                scount+=1
                #print(scount)
            #break
        return avg_acc/scount
    
"""

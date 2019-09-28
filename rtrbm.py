'''
The Recurrent Neural Networks
Son N. Tran
sontn.fz@gmail.com
'''
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import tensorflow as tf
import time

from rtrbm_cell import *
from utils import *

class RTRBM(object):
    def __init__(self,conf,dataset,model_type="rtdrbm"):
        self.conf = conf
        self.model_type = model_type
        self.dataset = dataset
        self.max_len = dataset.get_max_len()
        self.ckp_name = conf.ckp_file

        assert (not (conf.quick_eval and conf.predict_type=="viterbi")),"Cannot apply to batch for viterbi"
        
    def build_model(self):
        hidNum = self.conf.hidNum
        visNum = self.dataset.sensor_num()
        labNum = self.dataset.total_combined_acts()

        inp_ftr_len = self.dataset.inp_ftr_len() if self.conf.sparse_input else visNum
        dtype = tf.int32 if self.conf.sparse_input else tf.float32
        
        self.x = tf.placeholder(dtype,[1,self.max_len,inp_ftr_len])
        self.y = tf.placeholder(tf.float32,[1,self.max_len,labNum])    # for training

        mask  =  tf.sign(tf.reduce_max(tf.abs(self.y),reduction_indices=2))
                 
        lens = length(self.x)

        cell = RTRBMCell(visNum,labNum,hidNum)
        
        o,_ = tf.nn.dynamic_rnn(
                cell,
                [self.x,self.y],
                dtype=tf.float32,
                sequence_length=lens
            )
        
        if self.conf.predict_type=="rnn":
            scope = vs.get_variable_scope()
            scope.reuse_variables()
            pred,_= dynamic_rnn(
                cell,
                [self.x],
                dtype=tf.float32,
                sequence_length=lens
            )
            pred = tf.argmax(pred,2)
        elif self.conf.predict_type=="viterbi":
            scope = vs.get_variable_scope()
            scope.reuse_variables()
            #cell.set_viterbi()
            pred,_ = viterbi(
                cell,[self.x],
                dtype=tf.float32,
                sequence_length=lens
            )
            #### TODO traceback to get most likely sequence
            #print(pred.get_shape()) --> 1xmax_lenxlab_size
            #print(pred.get_shape())
            #print(self.y.get_shape())
            #input("")
        else:
            raise ValueError("predict_type must be identified")

        if self.conf.opt_type=="xen": #Cross entropy
            cost = cross_entropy_with_logits(o,self.y,lens,mask)
        elif self.conf.opt_type=="llh":
            cost = negative_log_likelihood(o,self.y,lens,mask)
        else:
            raise ValueError("Opt type must be specified")
        
        if self.conf.predict_type=="rnn":  
            acc = accuracy(pred,tf.argmax(self.y,2),lens,mask)
        elif self.conf.predict_type=="viterbi":
            acc = pred
        else:
            raise ValueError("Predict type must be specified")
        
        """ Regularization 
        l2 =  self.conf.weight_decay*sum(tf.nn.l2_loss(tf_var)
                                            for tf_var in tf.trainable_variables()
                                            if not ("Bias" in tf_var.name or "softmax_b" in tf_var.name))
        cost += l2
        """
        return acc,cost    
    def run(self):
        with tf.Graph().as_default():
            #Build grseaph
            acc,loss = self.build_model()
            if tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint process found, loading hyper-params...")
                lr,epoch,max_vld_acc,per_dec_count,lr_decay_count = load_process(self.ckp_name)
                if lr==0:
                    lr = self.conf.lr
            else:
                lr  = self.conf.lr
                epoch = max_vld_acc=  per_dec_count = lr_decay_count = 0
                
            optimizer = tf.train.GradientDescentOptimizer(lr)
            
            tvars   = tf.trainable_variables()
            grads =tf.gradients(loss,tvars) 
            train_op = optimizer.apply_gradients(zip(grads,tvars))
            
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
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
                    #start_time = time.time()
                    batch_x,batch_y = self.dataset.next_seq_vec_batch(batch_size=self.conf.batch_size)
                    _,err = session.run([train_op,loss],{self.x:batch_x,self.y:batch_y})
                    
                    if np.isnan(err) or np.isinf(err):
                        raise ValueError("Nan error")
                    total_err += err
                    #print(time.time()-start_time)
                    ### Now set the first row of weights to be zeros
                else:
                    epoch +=1
                    self.dataset.rewind()    
                    # Evaluate on evaluation set
                    if self.conf.quick_eval:# Evaluate the whole set
                        vld_x,vld_y = self.dataset.valid_seq_vec_dat()
                        [vld_acc] = session.run([acc],{self.x:vld_x,self.y:vld_y})
                    else: # evaluate each samples
                        vld_acc = self.evaluate(session,acc,eval_type="validation")
                    #EARLY STOPPING
                    if vld_acc>=max_vld_acc:
                        max_vld_acc = vld_acc
                        saver.save(session,self.ckp_name)                    
                    else:
                        per_dec_count +=1
                        if per_dec_count >= self.conf.NUM_DEC_4_LR_DECAY:
                            saver.restore(session,self.ckp_name)
                            per_dec_count = 0
                            lr_decay_count+=1
                            lr = lr/(1+self.conf.LR_DECAY_VAL)
                    
                    print("[Epoch %d: %.5f] %.5f %.5f %.5f %d %d"% (epoch,total_err,lr,vld_acc,max_vld_acc,per_dec_count,lr_decay_count))
                    save_process(self.ckp_name,lr,epoch,max_vld_acc,per_dec_count,lr_decay_count)
                    
                    total_err = 0
                    if lr_decay_count >= self.conf.MAX_LR_DECAY or epoch >= self.conf.MAX_ITER:
                        running = False
                        
            ##### Testing if exists
            if self.dataset.is_test:
                saver.restore(session,self.ckp_name)
                if self.conf.quick_eval:# Evaluate the whole set
                    tst_x,tst_y = self.dataset.test_seq_vec_dat()
                    [tst_acc] = session.run([acc],{self.x:tst_x,self.y:tst_y})
                else:
                    tst_acc = self.evaluate(session,acc,eval_type="test")     
        
            return max_vld_acc,tst_acc
        
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
        return avg_acc/scount

def traceback_acc(trace,y_,seq_len):
    trace = np.squeeze(trace)
    max_len,lab_size = trace.shape
    y_ = np.argmax(y_,axis=2)
    y_pred = np.zeros(y_.shape)
    last = np.argmax(trace[seq_len-1,int(lab_size/2):-1])

    trace = trace[:,:int(lab_size/2)]
    y_pred[0,seq_len[0]-1] = last
    for i in range(seq_len[0]-1,0,-1):
        y_pred[0,i-1] =  trace[i,int(y_pred[0,i])]

#    print(last)
#    print(y_pred[:,:seq_len[0]])
#    print(trace)
#    input("")
    return np.mean(np.equal(y_[:,:seq_len[0]],y_pred[:,:seq_len[0]]).astype(float))
    

'''
The Recurrent Neural Networks
Son N. Tran
sontn.fz@gmail.com
'''
from tensorflow.python.ops import variable_scope as vs
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from lstm_drbm_cell import *
from ae_rnn_cell import *
from utils import *

if int(tf.__version__[0])==1:
    BasicRNNCell = tf.contrib.rnn.BasicRNNCell
    LSTMCell = tf.contrib.rnn.LSTMCell
    GRUCell = tf.contrib.rnn.GRUCell
else:
    BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
    LSTMCell = tf.nn.rnn_cell.LSTMCell
    GRUCell = tf.nn.rnn_cell.GRUCell


class LSTM_SCRBM(object):
    def __init__(self,conf,dataset,model_type="rtdrbm"):
        self.conf = conf
        self.model_type = model_type
        self.dataset = dataset
        self.max_len = dataset.get_max_len()
    
        self.ckp_name = conf.ckp_file
        
    def build_model(self):
        hidNum = self.conf.hidNum
        visNum = self.dataset.sensor_num()
        labNum = self.dataset.total_combined_acts()

        inp_ftr_len = visNum
        dtype = tf.float32
        
        self.x  = tf.placeholder(dtype,[1,None,visNum])
        self.y  = tf.placeholder(tf.float32,[1,None,labNum])    # for training
        self.l  = tf.placeholder(tf.float32,shape=[1])        

        if self.conf.cell_type     == "BasicRNN":
            if self.conf.activation=="sigmoid":
                activation = sigmoid
            else:
                activation = tanh                
            cell  = BasicRNNCell(hidNum,activation=activation)
            cell_base   = "RNN"
        elif self.conf.cell_type   == "LSTM":
            cell  = LSTMCell(hidNum)
            cell_base   = "RNN"            
        else:
            raise ValueError("cell type is not specified!!!")
            
        s,_ = tf.nn.dynamic_rnn(
            cell,
            self.x,
            dtype=tf.float32,
            sequence_length=self.l
        )
        
        s = tf.slice(s,[0,-1,0],[1,1,hidNum])
                
        with tf.variable_scope("softmax_layer"):
            weights = tf.get_variable("softmax_w",[hidNum,labNum],initializer=tf.truncated_normal_initializer(stddev=1e-1))
            biases  = tf.get_variable("softmax_b",[labNum],initializer=tf.constant_initializer(0.0))

        o = tf.matmul(s,weights)+biases
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = o,labels=self.y))
        pred = tf.argmax(o,axis=1)

        return cost,pred
    
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
                    vld_rec, vld_pre, vld_f1 = evaluate(self,session,pred)
                    if self.conf.eval_metric=="recall":# Negative Log-likelihood
                        vld_val = vld_nllh
                    elif self.conf.eval_metric=="precision":
                        vld_val = vld_acc
                    elif self.conf.eval_metric=="f1":
                        vld_val = vld_f1                                        
                    
                    #EARLY STOPPING
                    if vld_val>=max_vld_val:
                        max_vld_val = vld_val
                        if self.ckp_name:
                            saver.save(session,self.ckp_name)                            
                        per_dec_count = 0
                    else:
                        per_dec_count +=1
                        if self.conf.opt=="sgd":
                            lr = lr/(1+self.conf.LR_DECAY_VAL)

                    print("[Epoch %d] rec:%.5f pre:%.5f f1:%.5f"  % (epoch,vld_rec,vld_pre,vld_f1))
                    total_err = 0
                    
                    save_process(self.ckp_name,[lr,epoch,max_vld_val,per_dec_count])
                    
                    if per_dec_count >= self.conf.DEC_NUM_4_EARLY_STOP or epoch >= self.conf.MAX_ITER:
                        running = False

            ##### Evaluation on validation set
            if self.ckp_name:
                saver.restore(session,self.ckp_name)
                
            vld_rec, vld_pre, vld_f1 = evaluate(self,session,pred)            
            ##### Evaluation on test set
            if self.dataset.is_test:                
                tst_rec, tst_pre, tst_f1 = evaluate(self,session,pred,eval_type="test")
            else:
                tst_rec =  tst_pre =  tst_f1 = 0
                
            return vld_rec,vld_pre,vld_f1,tst_rec,tst_pre,tst_f1
            
def act_type(t):
    if t=="sigmoid":
        return sigmoid
    else:
        return tanh

def evaluate(self,session,pred,eval_type="validation"):    
    ys = []
    os = []
    while True:
        if eval_type=="validation":
            x_,y_ = self.dataset.next_valid()
        elif eval_type=="test":
            x_,y_ = self.dataset.next_test()
        if x_ is None:
            break
        pred = session.run([pred],{self.x:x_,self.y:y_})
        ys.append(np.argmax(y_,axis=1))
        os.append(pred)
    rec += recall_score(ys,os)
    pre += precision_score(ys,os)
    f1  += f1_score(ys,os,average="micro")            

    return rec,pre,f1
    

    

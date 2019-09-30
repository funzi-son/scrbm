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
        elif self.conf.cell_type   == "GRU":
            cell = GRUCell(hidNum)
            cell_base   = "RNN"
        elif self.conf.cell_type == "AE":                
            cell = AERNNCell(hidNum,labNum,f=act_type(self.conf.f),g=act_type(self.conf.g))
            cell_base = "RBM"
        elif self.conf.cell_type == "AEsoft":
            cell = AERNNSmxCell(hidNum,labNum,f=act_type(self.conf.f),g=act_type(self.conf.g))
            cell_base = "RBMsoft"
        elif self.conf.cell_type == "AELSTM":
            cell = AELSTMCell(hidNum,labNum)
            cell_base = "RBM"
        elif self.conf.cell_type   == "RDRBM":
            cell = RDRBMCell(hidNum,labNum)
            cell_base   = "RBM"
        elif self.conf.cell_type   == "DRBM":
            cell = DRBMCell(hidNum,labNum)
            cell_base   = "RBM"
        elif self.conf.cell_type =="RDRBM":
            cell = RDRBMCell(hidNum,labNum)
            cell_base  = "RBM"
        elif self.conf.cell_type== "DRBMsoft":
            cell = DRBMSmxCell(hidNum,labNum)
            cell_base  = "RBMsoft"
        elif self.conf.cell_type   == "LSTMDRBM":
            cell = LSTM_DRBMCell(hidNum,labNum,gate=self.conf.gate_use)
            cell_base   = "RBM"
        elif self.conf.cell_type   == "LSTMDRBMsoft":
            cell = LSTM_DRBMSmxCell(hidNum,labNum,gate=self.conf.gate_use)
            cell_base   = "RBMsoft"
        else:
            raise ValueError("cell type is not specified!!!")

        if cell_base == "RBM":
            o,_ = tf.nn.dynamic_rnn(
                cell,
                [self.x,self.y],
                dtype=tf.float32,
                sequence_length= self.l
            )

            pred,_= tf.nn.dynamic_rnn(
                cell,
                [self.x],
                dtype=tf.float32,
                sequence_length= self.l
            )
            pred = tf.argmax(pred,2)
            
        elif cell_base == "RBMsoft":
            o,_ = tf.nn.dynamic_rnn(
                cell,
                self.x,
                dtype=tf.float32,
                sequence_length= self.l
            )
            pred = tf.argmax(o,2)            
            
        elif cell_base=="RNN":
            s,_ = tf.nn.dynamic_rnn(
                cell,
                self.x,
                dtype=tf.float32,
                sequence_length=self.l
            )

            s = tf.reshape(s,[-1,hidNum])
            with tf.variable_scope("softmax_layer"):
                weights = tf.get_variable("softmax_w",[hidNum,labNum],initializer=tf.truncated_normal_initializer(stddev=1e-1))
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
                    #print("%d sample trained"%c)
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

                    #print("[Epoch %d] err:%.5f aac:%.5f nllh:%.5f f1:%.5f (%s):%f %d"  % (epoch,total_err,vld_acc,vld_nllh,vld_f1,self.conf.eval_metric,max_vld_val,per_dec_count))
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
            
def act_type(t):
    if t=="sigmoid":
        return sigmoid
    else:
        return tanh

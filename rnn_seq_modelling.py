'''
The Recurrent Neural Networks for sequence modelling
Son N. Tran
sontn.fz@gmail.com
'''
import time
import os
import sys
import numpy as np
import tensorflow as tf

from utils import length, negative_log_likelihood,save_process,load_process

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

        # This is negative log likelihood loss
        cost = negative_log_likelihood(o,self.y,lens,mask)

        # This is sequence2sequence loss
        """
        cost = tf.contrib.seq2seq.sequence_loss(
            o,
            tf.argmax(self.y,axis=2),
            tf.ones([self.conf.batch_size, self.conf.num_steps], dtype=tf.float32), average_across_timesteps=False,average_across_batch=True)
        cost = tf.reduce_sum(cost)
        """
        
        # regularization --> dont use now
        """
        l2 =  self.conf.weight_decay*sum(tf.nn.l2_loss(tf_var)
                                            for tf_var in tf.trainable_variables()
                                            if not ("Bias" in tf_var.name or "softmax_b" in tf_var.name))
        cost += l2
        """
        
        return cost
    
    def run(self):
        with tf.Graph().as_default():            
            # Build graph
            loss = self.build_model()
            if self.conf.save_ckpt and tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint process found, loading hyper-params...")
                lr,epoch,min_vld_val,per_dec_count = load_process(self.ckp_name)
                if lr==0:
                    lr = self.conf.lr
            else:
                lr  = self.conf.lr
                epoch =  per_dec_count = 0
                min_vld_val = sys.maxsize
                
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
            
            if self.conf.save_ckpt and tf.train.checkpoint_exists(self.ckp_name):
                print("Checkpoint found, loading checkpoint...")
                saver.restore(session,self.ckp_name)
            else:
                session.run(init)

            start_time = time.time()
            log = np.empty((0,4))
            while running:
                batch_x,batch_y = self.dataset.next_seq_vec_batch()
                if batch_x is not None:
                    _,err = session.run([train_op,loss],{self.x:batch_x,self.y:batch_y,self.lr:lr})
                    total_err += err
                else:
                    count = 0
                    epoch +=1
                    # Evaluate on evaluation set
                    trn_val = nllh_evaluate(self,session,loss,eval_type="train")
                    vld_val = nllh_evaluate(self,session,loss)

                    elapsed_time = time.time() - start_time

                    #logging
                    log = np.append(log,[[epoch,elapsed_time,trn_val,vld_val]],axis=0)
                    np.savetxt(self.conf.progrs_log,log,delimiter=',')
                    #EARLY STOPPING
                    if vld_val<min_vld_val:
                        min_vld_val = vld_val
                        saver.save(session,self.ckp_name)
                        per_dec_count = 0
                    else:
                        per_dec_count +=1
                        if self.conf.opt=="sgd":                            
                            lr = lr/(1+self.conf.LR_DECAY_VAL)

                    
                    print("[Epoch %d] err = %.5f nllh: %.5f  %d"  % (epoch,trn_val,min_vld_val,per_dec_count))
                    total_err = 0
                    save_process(self.ckp_name,lr,epoch,min_vld_val,per_dec_count)
                    
                    if per_dec_count >= self.conf.DEC_NUM_4_EARLY_STOP or epoch >= self.conf.MAX_ITER:
                        running = False

            ##### Evaluation on validation set
            if self.conf.save_ckpt:
                saver.restore(session,self.ckp_name)
            #vld_nllh, vld_acc, vld_f1 = evaluate(self,session,self.dataset,nllh,pred)            
            ##### Evaluation on test set
            if self.dataset.is_test:                
                tst_nllh = nllh_evaluate(self,session,loss,eval_type="test")
                
            return min_vld_val,tst_nllh

def nllh_evaluate(model,session,nnlh,eval_type="validation"):
    avg_val = 0
    scount = 0
    while True:
        if eval_type=="validation":
            x_,y_ = model.dataset.valid_seq_vec_batch()
        elif eval_type=="test":
            x_,y_ = model.dataset.test_seq_vec_batch()
        elif eval_type=="train":
            x_,y_ = model.dataset.next_seq_vec_batch()
            
        if x_ is None:
            break
        val = session.run(nnlh,{model.x:x_,model.y:y_})
        #print(val)
        avg_val += val*x_.shape[0]
        scount+=x_.shape[0]
    return avg_val/scount            

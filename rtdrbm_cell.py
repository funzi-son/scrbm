from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.nn_ops import softmax
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops

from tensorflow.python.ops import rnn_cell_impl

import numpy as np
import tensorflow as tf

print(tf.__version__)
if int(tf.__version__[0])==1:
    print(tf.__version__[2])
    if int(tf.__version__[2]) == 0:
        from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
    else:
        from tensorflow.python.ops.rnn_cell_impl import  RNNCell

else:
    from tensorflow.python.ops.rnn_cell import RNNCell

class BasicRTDRBMCell(RNNCell):
    """ Basic RTDRBMCell with dense input """
    def __init__(self,input_size,label_size,hidden_size,activation=sigmoid,reuse=None):
        self._input_size = input_size
        self._label_size = label_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._viterbi = False
    @property
    def state_size(self):
        if self._viterbi:
            return self._hidden_size+1
        else:
            return self._hidden_size
    
    @property
    def output_size(self):
        if self._viterbi:
            return 2*self._label_size
        else:
            return self._label_size

    @property
    def label_size(self):
        return self._label_size
    
    def set_viterbi(self):
        self._viterbi = True
        
    def __call__(self,inputs,state,scope=None):
        with vs.variable_scope(scope or "basic_rtdrbm_cell"):
            output,probs = _basic_linear(inputs,state,self._input_size,self._label_size,self._hidden_size)
            state = self._activation(output)
        return probs,state
    
class RTDRBMCell(RNNCell):
    """ Basic RTDRBMCell with dense input """
    def __init__(self,input_size,label_size,hidden_size,activation=sigmoid,reuse=None):
        #super(RTDRBMCell,self).__init__(_reuse=reuse)
        self._input_size = input_size
        self._label_size = label_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._viterbi = False

    @property
    def state_size(self):
        if self._viterbi:
            return self._hidden_size+1
        else:
            return self._hidden_size
    
    @property
    def output_size(self):
        if self._viterbi:
            return 2*self._label_size
        else:
            return self._label_size

    @property
    def label_size(self):
        return self._label_size
    
    def set_viterbi(self):
        self._viterbi = True
        
    def __call__(self,inputs,state,scope=None):
        with vs.variable_scope(scope or "rtdrbm_cell"):
            output,probs = _linear(inputs,state,self._input_size,self._label_size,self._hidden_size)
            state = self._activation(output)
            
        return probs,state
        


def _basic_linear(inputs,state,input_size,label_size,hidden_size):
    # inputs[0] -> x
    # inputs[1] -> y
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        #initializer=tf.truncated_normal_initializer(stddev=5e-2)
        try:
            Wxh = vs.get_variable("Wxh",[input_size,hidden_size],dtype=tf.float32)
            Wyh = vs.get_variable("Wyh",[label_size,hidden_size],dtype=tf.float32)
            Whh = vs.get_variable("Whh",[hidden_size,hidden_size],dtype=tf.float32)
            yb  = vs.get_variable("yb",[label_size,1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            hb  = vs.get_variable("hb",[hidden_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        except ValueError:
            outer_scope.reuse_variables()
            Wxh = vs.get_variable("Wxh")
            Wyh = vs.get_variable("Wyh")
            Whh = vs.get_variable("Whh")
            yb  = vs.get_variable("yb")
            hb  = vs.get_variable("hb")

    I = nn_ops.bias_add(tf.matmul(inputs[0],Wxh)+ tf.matmul(state,Whh),hb)
    logits = tf.stack([I + tf.nn.embedding_lookup(Wyh,[i]) for i in range(label_size)])
    #print(labs.get_shape())
    logits = tf.reduce_sum(tf.log(1+tf.exp(logits)),axis=2)+yb
    # Check check check probs = probs + tf.log(yb)
    logits = tf.transpose(logits - tf.reduce_max(logits,axis=0))
    
    if len(inputs)==1:
        output = I + tf.matmul(softmax(logits),Wyh)
    elif len(inputs)==2:
        output = I + tf.matmul(inputs[1],Wyh)

    return output,logits


def _linear(inputs,state,input_size,label_size,hidden_size):
    # inputs[0] -> x
    # inputs[1] -> y
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        #initializer=tf.truncated_normal_initializer(stddev=5e-2)
        try:
            Wxh = tf.get_variable("Wxh",[input_size,hidden_size],dtype=tf.float32)
            Wyh = vs.get_variable("Wyh",[label_size,hidden_size],dtype=tf.float32)
            Whh = vs.get_variable("Whh",[hidden_size,hidden_size],dtype=tf.float32)
            Why = vs.get_variable("Why",[hidden_size,label_size],dtype=tf.float32) 
            yb  = vs.get_variable("yb",[label_size,1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            hb  = vs.get_variable("hb",[hidden_size],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        except ValueError:
            outer_scope.reuse_variables()
            Wxh = tf.get_variable("Wxh")
            Wyh = vs.get_variable("Wyh")
            Whh = vs.get_variable("Whh")
            Why = vs.get_variable("Why") 
            yb  = vs.get_variable("yb")
            hb  = vs.get_variable("hb")


    I = nn_ops.bias_add(tf.matmul(inputs[0],Wxh)+tf.matmul(state,Whh),hb)
    ybt = nn_ops.bias_add(tf.matmul(state,Why),tf.squeeze(yb))

    logits = tf.stack([I + tf.nn.embedding_lookup(Wyh,[i]) for i in range(label_size)])
    #print(labs.get_shape())

    logits = tf.reduce_sum(tf.log(1+tf.exp(logits)),axis=2) + tf.transpose(ybt)
    # Check check check probs = probs + tf.log(yb) 
    logits = tf.transpose(logits - tf.reduce_max(logits,axis=0))
    
    if len(inputs)==1:
        output = I + tf.matmul(softmax(logits),Wyh)
    elif len(inputs)==2:
        output = I + tf.matmul(inputs[1],Wyh)

    return output,logits

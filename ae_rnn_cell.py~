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
from tensorflow.python.ops import init_ops
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

from rnn_cell_impl import _linear
from rnn_cell_impl import LSTMStateTuple

class AERNNCell(RNNCell):
    def __init__(self,num_units,num_labs,
                 f=sigmoid,g=sigmoid,reuse=None):
        self._num_units = num_units
        self._num_labs = num_labs
        self._factivation = f
        self._gactivation = g
        
    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_labs

    @property
    def label_size(self):
        return self._num_labs
        
    def __call__(self,inputs,state):
        x = inputs[0]
        input_size = x.get_shape()[1]
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope,reuse=tf.AUTO_REUSE) as unit_scope:
                    #initializer=tf.truncated_normal_initializer(stddev=5e-2)
            Wxh = vs.get_variable("Wxh",[input_size,self._num_units],dtype=tf.float32)
            Why = vs.get_variable("Wyh",[self._num_units,self._num_labs],dtype=tf.float32)
            Whh = vs.get_variable("Whh",[self._num_units,self._num_units],dtype=tf.float32)
            yb  = vs.get_variable("yb",[self._num_labs],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            hb  = vs.get_variable("hb",[self._num_units],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            
            i = nn_ops.bias_add(math_ops.matmul(inputs[0],Wxh)+ math_ops.matmul(state,Whh),hb)
                        
            output = nn_ops.bias_add(math_ops.matmul(self._factivation(i),Why),yb)
            
            if len(inputs)==1:
                y =  softmax(output)
            elif len(inputs)==2:
                y = inputs[1]
                
            i_ = math_ops.matmul(y,array_ops.transpose(Why))
        new_state = i + i_
        new_state = self._gactivation(new_state)
        
        return output,new_state

class AERNNSmxCell(RNNCell):
    def __init__(self,num_units,num_labs,
                 f=sigmoid,g=sigmoid,reuse=None):
        self._num_units = num_units
        self._num_labs = num_labs
        self._factivation = f
        self._gactivation = g
        
    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_labs

    @property
    def label_size(self):
        return self._num_labs
    

    
    def __call__(self,inputs,state):
        input_size = inputs.get_shape()[1]
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope,reuse=tf.AUTO_REUSE) as unit_scope:
                    #initializer=tf.truncated_normal_initializer(stddev=5e-2)
            Wxh = vs.get_variable("Wxh",[input_size,self._num_units],dtype=tf.float32)
            Why = vs.get_variable("Wyh",[self._num_units,self._num_labs],dtype=tf.float32)
            Whh = vs.get_variable("Whh",[self._num_units,self._num_units],dtype=tf.float32)
            yb  = vs.get_variable("yb",[self._num_labs],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            hb  = vs.get_variable("hb",[self._num_units],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            
            i = nn_ops.bias_add(math_ops.matmul(inputs,Wxh)+ math_ops.matmul(state,Whh),hb)
                        
            output = nn_ops.bias_add(math_ops.matmul(self._factivation(i),Why),yb)
                    
            y =  softmax(output)
            
                
            i_ = math_ops.matmul(y,array_ops.transpose(Why))
        new_state = i + i_
        new_state = self._gactivation(new_state)
        
        return output,new_state

class AEGRUCell(RNNCell):
    def __init__(self):
        print("TODO")



class AEGRUSmxCell(RNNCell):
    def __init__(self):
        print("TODO")


class AELSTMCell(RNNCell):
    def __init__(self,num_units,num_labs,
                 use_peepholes  = False, cell_clip = None,
                 initializer = None, num_proj = None, proj_clip = None,
                 forget_bias = 1.0, state_is_tuple = True,
                 activation     = None, reuse=None):

        """ 
        Arguments' details
        """

        super(AELSTMCell,self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%: Using concatenated state is slower. Use state_is_tuple=True",self)

        self._num_units      = num_units
        self._num_labs       = num_labs
        self._use_peepholes  = use_peepholes
        self._cell_clip      = cell_clip
        self._initializer    = initializer
        self._num_proj       = num_proj
        self._proj_clip      = proj_clip
        self._forget_bias    = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation     = activation or math_ops.tanh

        
        if num_proj:
            raise ValueError("Not supported")
        else:
            self._state_size = (
                LSTMStateTuple(num_units,num_units)
                if state_is_tuple else 2*num_units)
            self._output_size = num_labs

           
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._num_labs

    def call(self,inputs,state):
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid  = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev,h_prev) = state
        else:
            c_prev = array_ops.slice(state,[0,0],[-1,self._num_units])
            h_prev = array_ops.slice(state,[0,self._num_units],[-1,num_proj])

        x = inputs[0]
        dtype = x.dtype
        input_size = x.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from get_shape()") # [-1]?
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope,reuse=tf.AUTO_REUSE,initializer=self._initializer) as unit_scope:
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            lstm_matrix = _linear([x,h_prev],4*self._num_units,bias=True) 
            i,j,f,o = array_ops.split(value=lstm_matrix, num_or_size_splits=4,axis=1)

            
            if self._use_peepholes:
                raise ValueError("Not supported yet")
            
            c = (sigmoid(f + self._forget_bias)*c_prev + sigmoid(i)*self._activation(j))

            if self._cell_clip is not None:
                raise ValueError("Not supported yet")
            
            m = sigmoid(o)*self._activation(c)

            Why  = vs.get_variable("Why",[self._num_units,self._num_labs])
            by   = vs.get_variable("by",[self._num_labs])

            output = nn_ops.bias_add(math_ops.matmul(m,Why),yb)

            if len(inputs)==1:
                y =  softmax(output)
            elif len(inputs)==2:
                y = inputs[1]
                
            i_ = math_ops.matmul(y,array_ops.transpose(Why))

            m+=i_
            
            new_state = (LSTMStateTuple(c,m) if self._state_is_tuple else array_ops.concat([c,m],1))
            
        return output, new_state

class AELSTMSmxCell(RNNCell):
    def __init__(self):
        print("TODO")


# SCRBM

## This code is for the Sequence Classification Restricted Boltzmann Machine

Prerequisites:

+ Python 3.5+

+ Tensorflow 1.8+

+ Scikitlearn 0.20+

Run code:

python orc_example.py <cell> <learning rate> <hidden units> <optimiser> <objective func>

where
+ cell: [BasicRNN, GRU, LSTM, AE, AESoft, AELSTM, DRBM, RDRBM,LSTMDRBM,LSTMDRBMSoft]
+ learning rate
+ hidden units: number of hidden units
+ optimiser: [sgd, adam, rmsprop, nadam, adadelta, adagrad]
+ objective func: objective functions (xen: cross-entropy, llh: (negative) log-likelihood)

If there is a problem running the code please contact: sontn.fz@gmail.com or sn.tran@utas.edu.au

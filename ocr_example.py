"""
Test LSTM_SCRBM on OCR
"""
import os
import sys
import numpy as np

from ocr import OCR
from lstm_scrbm import LSTM_SCRBM

HOME = os.path.expanduser("~")

EXP_DIR = './ckpt'
DAT_DIR = '.'
FOLD_NUM = 10


argv_len = len(sys.argv)
ofunc    ="llh"
cell_type= "BasicRNN"

####
gate        = "combine"  # For AELSTM,RBMLSTM
activation  = "tanh"     # For RNN
f_act       = "tanh"     # for ppRNN (AERNN)
g_act       = "sigmoid"  # for ppRNN

if(len(sys.argv)>=5):
    print(sys.argv)
    cell_type= sys.argv[1]
    lrs      = [float(sys.argv[2])]
    hidNums  = [int(sys.argv[3])]
    opttype  = sys.argv[4]
    if argv_len >=6:
        ofunc = sys.argv[5]
else:
    lrs       = [0.001]
    hNum      = 10
    opttype   = "adam"
        
class Config():
    lr        = 0.001
    hidNum    = 0
    MAX_ITER  = 500
    DEC_NUM_4_EARLY_STOP = 20
    LR_DECAY_VAL = 0.015
    batch_size = 1
    
    if opttype=="gd":
        raise ValueError("Batch learning is not supported yet")
    
    cell_type   = "BasicRNN"
    gate_use    = ""
    activation  = ""
    f           = ""
    g           = ""
    
    obj_func    = ofunc
    opt         = opttype
    eval_metric = "accuracy"
    computation = None
    
def main():
    hidNum = hidNums[0]
    lr     = lrs[0]
    run(hidNum,lr)         

def run(hidNum,lr):
    conf        = Config()
    conf.lr     = lr
    conf.hidNum = hidNum
    conf.cell_type = cell_type
    
    if "LSTMDRBM" in conf.cell_type:
        conf.gate_use  = gate     # For LSTM-DRBM only
        
    if conf.cell_type=="BasicRNN":
        conf.activation = activation

    if "AE" in conf.cell_type:
        conf.f = f_act
        conf.g = g_act
    
    print("running ..." + conf.cell_type +"_"+ conf.obj_func + " "+ conf.opt
          + " hidNum=%d lr=%.5f"%(conf.hidNum,conf.lr))

    result_dir = (EXP_DIR                  
                  + "/"  + conf.opt
                  + "_"  + conf.cell_type  + "_"  + conf.gate_use  + "_"+ conf.activation  + "_" + conf.f  + "_" + conf.g
                  + "_"+ conf.obj_func +  "_h" + str(conf.hidNum)
                  + "_b" + str(conf.batch_size) +"_"+str(lr))

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
        
    result_log = result_dir+'/result_log.csv'    
    if os.path.isfile(result_log):
        print("exist file "+ result_log)
        return
    
    acc = []
    for fold in range(FOLD_NUM):
        print("Fold %d, training ..."%(fold+1))
        result_fold_log = result_dir+"/fold_"+str(fold+1)+"_log.csv"
        if os.path.isfile(result_fold_log):
            continue
        conf.ckp_file= result_dir + '/fold_'+str(fold+1)+'.ckpt'
        dataset = OCR(DAT_DIR,fold)
        model = LSTM_SCRBM(conf,dataset)
        vld_acc,vld_nllh,vld_f1,tst_acc,tst_nllh,tst_f1,_ = model.run()    
        acc.append([vld_acc,tst_acc])
        print("[Fold %d] : valid acc:%.5f test acc:%.5f" %(fold+1,vld_acc,tst_acc))       

    acc = np.mean(np.array(acc),axis=0)
    print("validation acc: %.5f  test acc: %.5f" % (acc[0],acc[1]))
    #Save to CSV File
    print("Saving results ...")
    np.savetxt(result_log,acc,delimiter=',')
    # delete all checkpoints
    print("Clear checkpoint graph ...")
    os.remove(os.path.join(result_dir,"checkpoint"))
    ckpt_files = os.listdir(result_dir)
    for f in ckpt_files:
        if ".ckpt." in f:
            os.remove(os.path.join(result_dir,f))
if __name__=="__main__":
    main()

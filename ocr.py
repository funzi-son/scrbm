"""
OCR data reading - python3
Son N. Tran
sontn.fz@gmail.com
"""

import pickle
import gzip

import numpy as np

class OCR(object):
    def __init__(self,data_path,fold=0):
        self.fold = fold
        # Reading pickle file in python2 format
        with gzip.open(data_path+'/data.pkl.gz','rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.data_dict = data_dict = u.load()
            self.n_classes = self.data_dict['n_classes']
            
            self.train_data = data_dict['X']['train'][fold]
            self.train_label = data_dict['y']['train'][fold]

            self.valid_data = data_dict['X']['verify'][fold]
            self.valid_label = data_dict['y']['verify'][fold]
            
            self.test_data = data_dict['X']['test'][fold]
            self.test_label = data_dict['y']['test'][fold]


            self.max_len = 0
            self.max_len = self.get_max_len()

        
        self.trn_inx = self.vld_inx = self.tst_inx = 0
        self.all_classes = np.eye(self.n_classes)
        self.is_test = True
    def get_max_len(self):
        if self.max_len ==0:
            for d in self.train_data+self.valid_data+self.test_data:
                if self.max_len < d.shape[0]:
                    self.max_len = d.shape[0]
        return self.max_len

    
    def total_combined_acts(self):
        '''Get number of classes '''
        return self.data_dict['n_classes']

    def sensor_num(self):
        ''' Get input size '''
        return self.train_data[0].shape[1]

    def next(self):
        if self.trn_inx >= len(self.train_data):
            self.trn_inx = 0
            return None,None

        x = self.train_data[self.trn_inx][np.newaxis,:,:]
        y = self.train_label[self.trn_inx]
        if y.size==1:
            y = [y]
        y = self.all_classes[y,:][np.newaxis,:,:]
        self.trn_inx+=1

        return x,y
        
    
    def next_valid(self):
        if self.vld_inx >= len(self.valid_data):
            self.vld_inx = 0
            return None,None

        x = self.valid_data[self.vld_inx][np.newaxis,:,:]
        y = self.valid_label[self.vld_inx]
        if y.size==1:
            y = [y]
        y = self.all_classes[y,:][np.newaxis,:,:]
        self.vld_inx+=1
        
        return x,y

    def next_test(self):
        if self.tst_inx >= len(self.test_data):
            self.tst_inx = 0
            return None,None

        x = self.test_data[self.tst_inx][np.newaxis,:,:]
        y = self.test_label[self.tst_inx]
        if y.size==1:
            y = [y]
        y = self.all_classes[y,:][np.newaxis,:,:]
        self.tst_inx+=1

        return x,y

    
    def rewind(self):
        self.trn_inx = 0
    
if __name__=="__main__":
    data = OCR('/home/tra161/WORK/Data/rtdrbm/data/ocr',fold=0)
    data.next()

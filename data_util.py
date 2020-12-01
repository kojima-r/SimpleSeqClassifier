import os
import numpy as np
import logging
import torch
import json


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, train=True, previous_n_step=None, after_n_step=None):
        super(SeqDataset, self).__init__()
        self.data = data
        self.previous_n_step=previous_n_step #None
        self.after_n_step=after_n_step #None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        k,v=self.data[idx]
        obj=np.load("./"+v["mel"][0])
        y=int(v["label"])
        s=obj.shape[0]
        if self.previous_n_step is not None:
            if s-self.previous_n_step>1:
                obj=obj[:s-self.previous_n_step,:]
                s=s-self.previous_n_step
            else:
                obj=obj
        if self.after_n_step is not None:
            obj=obj[:self.after_n_step,:]
            s=obj.shape[0]
        return obj, s, y


def load_dataset(filename, config, logger=None):
    if logger is None: logger=logging.getLogger(__name__)
    obj=json.load(open(filename))
    return  list(obj.items())

if __name__ == '__main__':
    config={}
    data=load_dataset("../ROP_1h_28w/rop_data_npy/dataset.json",config)
    dataset=SeqDataset(data)
    print(dataset,config)
    dataset[[1,2,3]]


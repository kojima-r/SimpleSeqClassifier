import os
import numpy as np
import logging
import torch
import json


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, train=True):
        super(SeqDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    """
    def __getitem__(self, idx):
        # out_obs, out_input, out_state=None,None,None
        out_obs=[]
        out_step=[]
        d=0

        for i in idx:
            k,v=self.data[i]
            obj=np.load("../"+v["mel"][0])
            s=obj.shape[0]
            d=obj.shape[1]
            out_obs.append(obj)
            out_step.append(s)
        max_s=np.max(out_step)
        out=np.zeros((len(out_obs),max_s,d))
        for i,o in enumerate(out_obs):
            out[i,:out_step[i],:]=o
        print(out.shape)
        return out, out_step
    """
    def __getitem__(self, idx):
        k,v=self.data[idx]
        obj=np.load("../"+v["mel"][0])
        y=int(v["label"])
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


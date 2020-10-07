import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_util import SeqDataset, load_dataset
import json
import logging
import numpy as np
import os

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, h_lstm_dim, out_dim, activation=F.relu, scale=0.1):
        super(SimpleMLP, self).__init__()
        linears=[]
        prev_d=in_dim
        if h_dim is None:
            h_dim=[]
        for d in h_dim:
            linears.append(self.get_layer(prev_d,d))
            prev_d=d
        self.linears = nn.ModuleList(linears)
        self.activation = activation

        self.lstm=nn.LSTM(input_size = prev_d,
            hidden_size = h_lstm_dim,
            batch_first = True)
        
        self.linear_out=self.get_layer(h_lstm_dim, out_dim)

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        nn.init.kaiming_uniform_(l.weight)
        return l

    def forward(self, x, s):
        for i in range(len(self.linears)):
            x = self.activation(self.linears[i](x))
        x=torch.nn.utils.rnn.pack_padded_sequence(x,s,batch_first=True,enforce_sorted=False)
        x, (hidden, cell) =self.lstm(x)
        x, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(x,batch_first =True) 
        x = self.activation(x)
        x = self.linear_out(x)

        return x


class LossLogger:
    def __init__(self):
        self.loss_history=[]
        self.loss_dict_history=[]

    def start_epoch(self):
        self.running_loss = 0
        self.running_loss_dict = {}
        self.running_count = 0

    def update(self, loss, loss_dict):
        self.running_loss += loss.detach().to('cpu')
        self.running_count +=1
        for k, v in loss_dict.items():
            if k in self.running_loss_dict:
                self.running_loss_dict[k] += v.detach().to('cpu')
            else:
                self.running_loss_dict[k] = v.detach().to('cpu')

    def end_epoch(self,mean_flag=True):
        if mean_flag:
            self.running_loss /= self.running_count
            for k in self.running_loss_dict.keys():
                self.running_loss_dict[k] /=  self.running_count
        self.loss_history.append(self.running_loss)
        self.loss_dict_history.append(self.running_loss_dict)

    def get_dict(self, prefix="train"):
        result={}
        key="{:s}-loss".format(prefix)
        val=self.running_loss
        result[key]=float(val)
        for k, v in self.running_loss_dict.items():
            if k[0]!="*":
                m = "{:s}-{:s}-loss".format(prefix, k)
            else:
                m = "*{:s}-{:s}".format(prefix, k[1:])
            result[m]=float(v)
        return result

    def get_msg(self, prefix="train"):
        msg = []
        for key,val in self.get_dict(prefix=prefix).items():
            m = "{:s}: {:.3f}".format(key,val)
            msg.append(m)
        return "  ".join(msg)

    def get_loss(self):
            return self.running_loss



from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    xs, steps, ys = [], [], []
    for x, step, y in batch:
        xs.append(torch.tensor(x))
        ys.append(y)
        steps.append(step)
    bx = pad_sequence(xs,batch_first=True)
    return bx, torch.tensor(steps), torch.tensor(ys)


class SeqClassifier:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model.to(device)
        self.device=device


    def _compute_batch_loss(self, batch, epoch):
        obs, lengths, ys = batch
        metrics = {}
        out=self.model(obs, lengths)
        ###
        seq_error=False
        if seq_error:
            loss_f = nn.CrossEntropyLoss()
            ys_tile=ys.unsqueeze(1).repeat(1,out.size()[1])
            n_label=out.size()[2]
            out_loss=loss_f(out.view(-1,n_label),ys_tile.view(-1))
            loss=out_loss
            _, predicted = torch.max(out.view(-1,n_label), 1)
            acc = (predicted == ys_tile.view(-1)).to(dtype=torch.float64).mean()
        else:
            loss_f = nn.CrossEntropyLoss()
            out_d=[out[i,lengths[i]-1,:] for i in range(out.size()[0])]
            out=torch.stack(out_d)
            out_loss=loss_f(out,ys)
            loss=out_loss
            _, predicted = torch.max(out, 1)
            acc = (predicted == ys).to(dtype=torch.float64).mean()

        ###
        ###
        ###
        loss_dict = {"*acc":acc}
        return loss, loss_dict

    def save(self,path):
        torch.save(self.model.state_dict(), path)

    def load(self,path):
        state_dict=torch.load(path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def save_ckpt(self, epoch, loss, optimizer, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

    def load_ckpt(self, path):
        ckpt=torch.load(path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def fit(self, train_data, valid_data):
        config = self.config
        batch_size = config["batch_size"]
        trainset = SeqDataset(train_data, train=True)
        validset = SeqDataset(valid_data, train=False)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, timeout=20, collate_fn=collate_fn
        )
        validloader = DataLoader(
            validset, batch_size=batch_size, shuffle=False, num_workers=4, timeout=20, collate_fn=collate_fn
        )
        optimizer = optim.Adam(
            self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )

        train_loss_logger = LossLogger()
        valid_loss_logger = LossLogger()
        prev_valid_loss=None
        best_valid_loss=None
        patient_count=0
        for epoch in range(config["epoch"]):
            train_loss_logger.start_epoch()
            valid_loss_logger.start_epoch()
            for i, batch in enumerate(trainloader, 0):
                optimizer.zero_grad()
                batch=[el.to(self.device) for el in batch]
                loss, loss_dict = self._compute_batch_loss(batch,epoch)
                train_loss_logger.update(loss, loss_dict)
                loss.backward()
                optimizer.step()
                del loss
                del loss_dict

            for i, batch in enumerate(validloader, 0):
                batch=[el.to(self.device) for el in batch]
                loss, loss_dict = self._compute_batch_loss(batch,epoch)
                valid_loss_logger.update(loss, loss_dict)
            train_loss_logger.end_epoch()
            valid_loss_logger.end_epoch()
            ## Early stopping
            l=valid_loss_logger.get_loss()
            if np.isnan(l):
                break
            if prev_valid_loss is None or l < prev_valid_loss:
                patient_count=0
            else:
                patient_count+=1
            prev_valid_loss=l
            ## check point
            check_point_flag=False
            if best_valid_loss is None or l < best_valid_loss:
                os.makedirs(config["result_path"]+f"/model/", exist_ok=True)
                path = config["result_path"]+f"/model/model.{epoch}.checkpoint"
                self.save_ckpt(epoch, l, optimizer, path)
                path = config["result_path"]+f"/model/best.checkpoint"
                self.save_ckpt(epoch, l, optimizer, path)
                path = config["result_path"]+f"/model/best.result.json"
                fp = open(path, "w")
                res=train_loss_logger.get_dict("train")
                res.update(valid_loss_logger.get_dict("valid"))
                json.dump(
                    res,
                    fp,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
                check_point_flag=True
                best_valid_loss=l

            ## print message
            ckpt_msg = "*" if check_point_flag else ""
            msg="\t".join(["[{:4d}] ".format(epoch + 1),
                train_loss_logger.get_msg("train"),
                valid_loss_logger.get_msg("valid"),
                "({:2d})".format(patient_count),
                ckpt_msg,])
            logger = logging.getLogger("logger")
            logger.info(msg)
        return train_loss_logger, valid_loss_logger

def get_default_config():
    config = {}
    # data and network
    # training
    config["epoch"] = 10
    config["patience"] = 5
    config["batch_size"] = 100
    #config["activation"] = "relu"
    #config["optimizer"] = "sgd"
    ##
    config["learning_rate"] = 1.0e-3
    # dataset
    config["train_valid_ratio"] = 0.2
    # save/load model
    config["load_model"] = None
    config["result"] = None
    config["weight_decay"] = 0.01
    config["hidden_layer_f"] = [32]
    config["hidden_layer_g"] = [32]
    config["hidden_layer_h"] = [32]
    # generate json
    return config



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")

    config=get_default_config()
    config.update(json.load(open("config.json")))
    data=load_dataset("../ROP_1h_28w/rop_data_npy/dataset.json",config)
    valid_num=int(len(data)*0.2)
    train_data=data[:len(data)-valid_num]
    valid_data=data[len(data)-valid_num:]
    model=SimpleMLP(40,[32],32,3)
    clf=SeqClassifier(config, model, device="cpu")

    clf.fit(train_data,valid_data)


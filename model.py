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
from sklearn.metrics import roc_curve,auc,accuracy_score,roc_auc_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, h_lstm_dim, out_dim, att_mode, activation=F.relu, bidirectional=False,num_lstm_layers=1):
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
            bidirectional = bidirectional,
            num_layers =num_lstm_layers,
            batch_first = True)
        if bidirectional:
            h_lstm_dim=h_lstm_dim*2
        self.linear_out=self.get_layer(h_lstm_dim, out_dim)
        self.att_mode=att_mode
        if self.att_mode:
            self.linear_att_k=self.get_layer(h_lstm_dim, h_lstm_dim)
            self.linear_att_v=self.get_layer(h_lstm_dim, out_dim)
            self.linear_att_q=self.get_layer(h_lstm_dim, h_lstm_dim)
 

    def get_layer(self,in_d,out_d):
        l=nn.Linear(in_d, out_d)
        nn.init.kaiming_uniform_(l.weight)
        return l

    def forward(self, x, s):
        for i in range(len(self.linears)):
            x = self.activation(self.linears[i](x))
        h=torch.nn.utils.rnn.pack_padded_sequence(x,s,batch_first=True,enforce_sorted=False)
        x, (hidden, cell) =self.lstm(h)
        x, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(x,batch_first =True) 
        x = self.activation(x)
        if self.att_mode:
            att_k=self.linear_att_k(x)
            att_v=self.linear_att_v(x)
            att_q=self.linear_att_q(x)
            m = nn.Softmax(dim=1)
            a=m((att_q*att_k).sum(dim=2))
            a=a.unsqueeze(2)
            x=(att_v*a).sum(dim=1)
            return x,a
        else:
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
    def __init__(self, config, model, device,att_mode):
        self.config = config
        self.model = model.to(device)
        self.device=device
        self.att_mode=att_mode

    def _compute_batch_loss(self, batch, epoch):
        obs, lengths, ys = batch
        #batch=[el.to(self.device) for el in batch]
        obs=obs.to(self.device)
        ys=ys.to(self.device)
        metrics = {}
        att=None
        ###
        seq_error=False
        loss_f = nn.CrossEntropyLoss()
        if seq_error: # errors for each time slice
            out=self.model(obs, lengths)
            ys_tile=ys.unsqueeze(1).repeat(1,out.size()[1])
            n_label=out.size()[2]
            out_loss=loss_f(out.view(-1,n_label),ys_tile.view(-1))
            loss=out_loss
            _, predicted = torch.max(out.view(-1,n_label), 1)
            acc = (predicted == ys_tile.view(-1)).to(dtype=torch.float64).mean()
        else: # errors for each sequence (last time slice)
            out,att=self.model(obs, lengths)
            if self.att_mode:
                pass
            elif False:
                out_d=[out[i,lengths[i]-1,:] for i in range(out.size()[0])]
                out=torch.stack(out_d)
            else:
                out_d=[]
                for i in range(out.size()[0]):
                    if lengths[i]>480*24:
                        o=out[i,480*24,:]
                    else:
                        o=out[i,lengths[i]-1,:]
                    out_d.append(o)
                out=torch.stack(out_d)
            out_loss=loss_f(out,ys)
            loss=out_loss
            _, predicted = torch.max(out, 1)
            acc = (predicted == ys).to(dtype=torch.float64).mean()

        ###
        ###
        ###
        loss_dict = {"*acc":acc}
        return loss, loss_dict, predicted, out, att

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

    def pred(self, test_data, previous_n_step=None, after_n_step=None):
        config = self.config
        batch_size = config["batch_size"]
        testset = SeqDataset(test_data, train=False, previous_n_step=previous_n_step, after_n_step=after_n_step)
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=4, timeout=20, collate_fn=collate_fn
        )
        test_loss_logger = LossLogger()
        test_loss_logger.start_epoch()
        all_pred=[]
        all_prob=[]
        all_att=[]
        for i, batch in enumerate(testloader, 0):
            loss, loss_dict, pred, out, att = self._compute_batch_loss(batch,0)
            test_loss_logger.update(loss, loss_dict)
            all_pred.append(pred.detach().to('cpu').numpy())
            out=torch.nn.functional.softmax(out)
            all_prob.append(out.detach().to('cpu').numpy())
            if att is not None:
                all_att.append(att.detach().to('cpu').numpy())
        test_loss_logger.end_epoch()
        ## print message
        msg="\t".join([">>  ",
            test_loss_logger.get_msg("test"),
            ])
        logger = logging.getLogger("logger")
        logger.info(msg)

        all_pred=np.concatenate(all_pred,axis=0)
        all_prob=np.concatenate(all_prob,axis=0)
        all_att=np.concatenate(all_att,axis=0)

        return all_pred, all_prob, all_att


    def fit(self, train_data, valid_data, previous_n_step=None, after_n_step=None):
        config = self.config
        batch_size = config["batch_size"]
        trainset = SeqDataset(train_data, train=True,  previous_n_step=previous_n_step, after_n_step=after_n_step)
        validset = SeqDataset(valid_data, train=False,  previous_n_step=previous_n_step, after_n_step=after_n_step)
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
                loss, loss_dict, _ , _ , _ = self._compute_batch_loss(batch,epoch)
                train_loss_logger.update(loss, loss_dict)
                loss.backward()
                optimizer.step()
                del loss
                del loss_dict

            for i, batch in enumerate(validloader, 0):
                loss, loss_dict, _ , _ , _ = self._compute_batch_loss(batch,epoch)
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
    config["nn_type"] = "lstm"
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


def set_file_logger(logger,config,filename):
    if "result_path" in config:
        filename=config["result_path"]+"/"+filename
        h = logging.FileHandler(filename=filename, mode="w")
        h.setLevel(logging.INFO)
        logger.addHandler(h)

def run_prev_all(config,logger,device,enable_train=False,train_test_mode=False):
    external_test=False 
    for i in range(20):
        step=24*7*i
        if enable_train:
            s="prev{:03d}.".format(step)
        else:
            s="prev{:03d}.".format(step)
            #s=""
        print(">>>>",s)
        if train_test_mode:
            run_train_test(config,logger,device,prefix=s,enable_train=enable_train, previous_n_step=step, after_n_step=None)
        else:
            run_cv_train(config,logger,device,prefix=s,enable_train=enable_train, previous_n_step=step, after_n_step=None,external_test=external_test)
    ###

def run_after_all(config,logger,device,enable_train=False,train_test_mode=False):
    external_test=False 
    for i in range(25):
        step=24*7*(i+1)
        if enable_train:
            s="after{:04d}.".format(step)
        else:
            s="after{:04d}.".format(step)
            #s=""
        print(">>>>",s)
        if train_test_mode:
            run_train_test(config,logger,device,prefix=s,enable_train=enable_train, previous_n_step=None, after_n_step=step)
        else:
            run_cv_train(config,logger,device,prefix=s,enable_train=enable_train, previous_n_step=None, after_n_step=step,external_test=external_test)
    ###


def run_train_test(config,logger,device,prefix="",enable_train=False, previous_n_step=None, after_n_step=None):
    att_mode=True

    data=load_dataset(config["data"],config)
    data=np.array(data)
    test_data=load_dataset(config["test_data"],config)
    test_data=np.array(test_data)
    
    np.random.seed(1234)
    
    ## model
    result_path=config["result_base_path"]
    config["result_path"]=result_path+"/"+prefix+"train"
    os.makedirs(config["result_path"],exist_ok=True)
    
    ## 
    m=len(data)
    valid_num=int(m*0.2)
    train_valid_idx=np.arange(m)
    np.random.shuffle(train_valid_idx)
    train_idx=train_valid_idx[:m-valid_num]
    valid_idx=train_valid_idx[m-valid_num:]
        
    train_data=data[train_idx]
    valid_data=data[valid_idx]
    if config["nn_type"]=="lstm3":
        model=SimpleMLP(40,[32],32,3,att_mode=att_mode, bidirectional=True,num_lstm_layers=3)
    elif config["nn_type"]=="lstm2":
        model=SimpleMLP(40,[32],32,3,att_mode=att_mode, bidirectional=True,num_lstm_layers=2)
    else:
        model=SimpleMLP(40,[32],32,3,att_mode=att_mode, bidirectional=False,num_lstm_layers=1)

    clf=SeqClassifier(config, model, device=device, att_mode=att_mode)
    if enable_train:
        clf.fit(train_data,valid_data, previous_n_step=previous_n_step, after_n_step=after_n_step)

    # load
    config["result_path"]=result_path+"/"+prefix+"train"
    path = config["result_path"]+f"/model/best.checkpoint"
    print("[LOAD]",path)
    if not os.path.exists(path):
        return 
    clf.load_ckpt(path)
    #print("===")
    pred, prob, att = clf.pred(test_data, previous_n_step=previous_n_step, after_n_step=after_n_step)

    y_pred=np.array([int(val) for val in pred])
    y_true=np.array([int(v["label"]) for k,v in test_data])
    ## evaluation
    acc=accuracy_score(y_true,y_pred)
    n_label=prob.shape[1]
    aucs=[roc_auc_score(y_true==i,prob[:,i]) for i in range(n_label)]
    conf_mat={}
    for yi_true,yi_pred in zip(y_true,y_pred):
        k=(yi_true,yi_pred)
        if k not in conf_mat:
            conf_mat[k]=0
        conf_mat[k]+=1
    print("Acc:",acc)
    print("AUC:",aucs)
    #print("===")
    print(conf_mat)
    
    fig=plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(n_label):
        fpr, tpr, thresholds = roc_curve(y_true==i,prob[:,i])
        plt.plot(fpr, tpr, label=''+str(i))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    path=result_path+"/"+prefix+"roc.png"
    fig.savefig(path)
    plt.clf()
    
    key_list=[]
    for k,v in conf_mat.items():
        key_list.append(k[0])
        key_list.append(k[1])
    max_label=max(key_list)
    ll=[]
    for i in range(max_label):
        l=[]
        for j in range(max_label):
            k=(i,j)
            if k in conf_mat:
                l.append(conf_mat[k])
            else:
                l.append(0)
        ll.append(l)
    result={}
    result["conf_mat"]=ll
    result["auc"]=aucs
    result["acc"]=acc
    result_all_data={}
    for i in range(len(test_data)):
        pair=test_data[i]
        result_all_data[pair[0]]=pair[1]
        result_all_data[pair[0]]["y_pred"]=int(y_pred[i])
        result_all_data[pair[0]]["y_true"]=int(y_true[i])
        result_all_data[pair[0]]["y_prob"]=prob[i].tolist()
        result_all_data[pair[0]]["attention"]=att[i].tolist()
    result["all"]=result_all_data
   
    ###
    path=result_path+"/"+prefix+"result.json"
    print("[SAVE]",path)
    with open(path,"w") as fp:
        json.dump(result,fp)
    
    return result
 

def run_cv_train(config,logger,device,prefix="",enable_train=False, previous_n_step=None, after_n_step=None,external_test=False):
    att_mode=True
    

    data=load_dataset(config["data"],config)
    data=np.array(data)
    
    np.random.seed(1234)
    kf=KFold(5,shuffle=True)
    fold=0
    result_path=config["result_base_path"]
    all_idx=[]
    all_pred=[]
    all_prob=[]
    all_true=[]
    all_att=[]
    acc_list=[]
    auc_list=[]
    conf_mat={}
    
    for train_valid_idx, test_idx in kf.split(data):
        config["result_path"]=result_path+"/"+prefix+"fold"+str(fold)
        #print("[RESULT]",config["result_path"])
        os.makedirs(config["result_path"],exist_ok=True)
        m=len(train_valid_idx)
        valid_num=int(m*0.2)
        np.random.shuffle(train_valid_idx)
        train_idx=train_valid_idx[:m-valid_num]
        valid_idx=train_valid_idx[m-valid_num:]
        #print(train_idx)
        #print(valid_idx)
        print(test_idx)
        
        if external_test:
            train_data=None
            valid_data=None
            test_data =data
        else:
            train_data=data[train_idx]
            valid_data=data[valid_idx]
            test_data =data[test_idx]
        if config["nn_type"]=="lstm3":
            model=SimpleMLP(40,[32],32,3,att_mode=att_mode, bidirectional=True,num_lstm_layers=3)
        elif config["nn_type"]=="lstm2":
            model=SimpleMLP(40,[32],32,3,att_mode=att_mode, bidirectional=True,num_lstm_layers=2)
        else:
            model=SimpleMLP(40,[32],32,3,att_mode=att_mode, bidirectional=False,num_lstm_layers=1)

        clf=SeqClassifier(config, model, device=device, att_mode=att_mode)
        if enable_train:
            clf.fit(train_data,valid_data, previous_n_step=previous_n_step, after_n_step=after_n_step)

        if external_test:
            path = config["result_original_path"]+"/"+prefix+"fold"+str(fold)
            path+=f"/model/best.checkpoint"
        else:
            path = config["result_path"]+f"/model/best.checkpoint"
        print("[LOAD]",path)
        if not os.path.exists(path):
            return 
        clf.load_ckpt(path)
        #print("===")
        pred, prob, att = clf.pred(test_data, previous_n_step=previous_n_step, after_n_step=after_n_step)

        y_pred=np.array([int(val) for val in pred])
        y_true=np.array([int(v["label"]) for k,v in test_data])
        all_idx.extend(test_idx)
        all_pred.extend(y_pred)
        all_prob.extend(prob)
        all_att.extend(att)
        all_true.extend(y_true)

        ## evaluation
        acc=accuracy_score(y_true,y_pred)
        n_label=prob.shape[1]
        aucs=[roc_auc_score(y_true==i,prob[:,i]) for i in range(n_label)]
        acc_list.append(acc)
        auc_list.append(aucs)
        for yi_true,yi_pred in zip(y_true,y_pred):
            k=(yi_true,yi_pred)
            if k not in conf_mat:
                conf_mat[k]=0
            conf_mat[k]+=1
        print("Acc:",acc)
        print("AUC:",aucs)
        fold+=1
    #print("===")
    result={}
    result["acc_list"]=acc_list
    print("Accuracy (mean):",np.mean(acc_list))
    result["acc_mean"]=np.mean(acc_list)
    print("Accuracy (std.):",np.std(acc_list))
    result["acc_std"]=np.std(acc_list)
    
    result["auc_list"]=auc_list
    print("ROC-AUC (mean):",np.mean(np.array(auc_list),axis=0))
    result["auc_mean"]=np.mean(auc_list)
    print("ROC-AUC (std.):",np.std(np.array(auc_list),axis=0))
    result["auc_std"]=np.mean(auc_list)
    print(conf_mat)
    all_true_=np.array(all_true)
    all_prob_=np.array(all_prob)
    all_pred_=np.array(all_pred)
    acc=accuracy_score(all_true_,all_pred_)
    aucs=[roc_auc_score(all_true_==i,all_prob_[:,i]) for i in range(n_label)]
    print("Accuracy(all):",acc)
    result["acc_all"]=acc
    print("ROC-AUC(all):",aucs)
    result["auc_all"]=aucs
    
    fig=plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(n_label):
        fpr, tpr, thresholds = roc_curve(all_true_==i,all_prob_[:,i])
        plt.plot(fpr, tpr, label=''+str(i))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    path=result_path+"/"+prefix+"roc.png"
    fig.savefig(path)
    plt.clf()
    """
    for i in range(len(all_idx)):
        pair=data[all_idx[i]]
        print(pair[0])
        print(all_true[i],all_pred[i])
        print(all_att[i].shape)
    """
    """
    result_data={}
    ### make conf mat
    for k,v in sorted(zip(all_idx,all_pred)):
        #print(data[k],int(k),int(v))
        result_data[int(k)]=int(v)
    result["result_data"]=result_data
    """
    key_list=[]
    for k,v in conf_mat.items():
        key_list.append(k[0])
        key_list.append(k[1])
    max_label=max(key_list)
    ll=[]
    for i in range(max_label):
        l=[]
        for j in range(max_label):
            k=(i,j)
            if k in conf_mat:
                l.append(conf_mat[k])
            else:
                l.append(0)
        ll.append(l)
    result["conf_mat"]=ll
    result["conf_mat"]=ll

    result_all_data={}
    for i in range(len(all_idx)):
        pair=data[all_idx[i]]
        result_all_data[pair[0]]=pair[1]
        result_all_data[pair[0]]["y_pred"]=int(all_pred[i])
        result_all_data[pair[0]]["y_true"]=int(all_true[i])
        result_all_data[pair[0]]["y_prob"]=all_prob[i].tolist()
        result_all_data[pair[0]]["attention"]=all_att[i].tolist()
    result["all"]=result_all_data
   
    ###
    path=result_path+"/"+prefix+"result_cv.json"
    print("[SAVE]",path)
    with open(path,"w") as fp:
        json.dump(result,fp)
    
    return result
 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='train/infer/train_cv/visualize')
    parser.add_argument('--config', type=str, default="config.json",
                        help='config json file')
    parser.add_argument('--data', type=str, default="./ROP_1h_28w/rop_data_npy/dataset.json",
                        help='data json file')
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )


    args = parser.parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config=get_default_config()
    config.update(json.load(open(args.config)))
    config["data"]=args.data
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logger")
    set_file_logger(logger,config,"log.txt")


    if torch.cuda.is_available():
        device = 'cuda'
        print("device: cuda")
    else:
        device = 'cpu'
        print("device: cpu")

    if args.mode=="train":
        run_cv_train(config, logger, device, enable_train=True, external_test=False)
    elif args.mode=="test":
        run_cv_train(config, logger, device, enable_train=False, external_test=False)
    elif args.mode=="cv_prev" or args.mode=="train_prev_all":
        run_prev_all(config,logger,device,enable_train=True)
    elif args.mode=="cv_prev_test" or args.mode=="test_prev_all":
        run_prev_all(config,logger,device,enable_train=False)
    elif args.mode=="cv_after" or args.mode=="train_after_all":
        run_after_all(config,logger,device,enable_train=True)
    elif args.mode=="cv_after_test" or args.mode=="test_after_all":
        run_after_all(config,logger,device,enable_train=False)
    elif args.mode=="train_after" or args.mode=="train_test_after":
        run_after_all(config,logger,device,enable_train=True,train_test_mode=True)
    elif args.mode=="train_prev" or args.mode=="train_test_prev":
        run_prev_all(config,logger,device,enable_train=True,train_test_mode=True)
    elif args.mode=="test_after":
        run_after_all(config,logger,device,enable_train=False,train_test_mode=True)
    elif args.mode=="test_prev":
        run_prev_all(config,logger,device,enable_train=False,train_test_mode=True)
    else:
        print("unknown")


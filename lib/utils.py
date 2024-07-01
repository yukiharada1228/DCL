
# coding: utf-8

# In[17]:


import pandas as pd
import os
import sys
import pdb
import torch

import math


# In[ ]:


def make_dirs(dir_path):
    if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    return


# In[59]:


class LogManager(dict):
    def __init__(self, epochs, iteration):
        super(LogManager, self).__init__()
        from collections import defaultdict
        self.epochs = epochs
        self.iteration = iteration
        self["epoch_log"] = defaultdict(dict)
        self["ite_log"] = defaultdict(dict)
    
    def save(self, save_dir):
        make_dirs(save_dir)
        self.to_dataframe("epoch_log").to_csv(os.path.join(save_dir, "epoch_log.csv"), float_format="%.6g")
        self.to_dataframe("ite_log").to_csv(os.path.join(save_dir, "ite_log.csv"), float_format="%.6g")
        return
    
    def __save_pickle__(self, obj, save_path):
        with open(save_path, mode="wb") as f:
            pickle.dump(obj, f, protocol=4)
        return
    
    def to_dataframe(self, key):     
        return pd.DataFrame.from_dict(self[key], orient="index")
    
        
    def load(self, save_dir):
        epoch_log = pd.read_csv(os.path.join(save_dir, "epoch_log.csv"), index_col=0)
        ite_log = pd.read_csv(os.path.join(save_dir, "ite_log.csv"), index_col=0)
        
        self["epoch_log"] = epoch_log.to_dict()
        self["ite_log"] = ite_log.to_dict()
        return


# In[60]:


class LogManagers(dict):
    def __init__(self, num_model, train_length, epochs, batch_size):
        super(LogManagers, self).__init__()
        self.num_model = num_model
        self.epochs = epochs
        self.train_len = train_length
        self.iteration = (train_length//batch_size+1) * epochs
        self.net = [LogManager(epochs, self.iteration) for i in range(self.num_model)]
            
    def __getitem__(self, key):
        return self.net[key]
    
    def save(self, save_dir):
        net_dirs = [os.path.join(save_dir, "net%d" % i) for i in range(self.num_model)]        
        for n, d in zip(self.net, net_dirs):
            n.save(d)        
        return
    
    
    def load(self, save_dir):
        net_dirs = [os.path.join(save_dir, "net%d" % i) for i in range(self.num_model)]
        for i, net_dir in enumerate(net_dirs):
            self.net[i].load(net_dir)
        return


# In[4]:


def save_pickle(self, obj, save_path):
    with open(save_path, mode="wb") as f:
        pickle.dump(obj, f, protocol=4)
    return

def load_pickle(self, path):
    with open(path, mode="rb") as f:
        result = pickle.load(f)
    return result



# In[41]:


def save_json(_dict, save_path):
    import json
    with open(save_path, mode="w") as f:
        json.dump(_dict, f)
    return


# In[3]:


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def adjust_learning_rate(optimizer, epoch, optim_args):
    scheduler_type = optim_args.get('scheduler_type', 'step')
    learning_rate = optim_args['args']['lr']

    if scheduler_type == 'step':
        gammas = optim_args.get('gammas', [])
        schedule = optim_args.get('schedule', [])
        lr = learning_rate
        for (gamma, step) in zip(gammas, schedule):
            if epoch >= step:
                lr = lr * gamma
            else:
                break
    elif scheduler_type == 'cosine':
        num_warmup_steps = optim_args.get('num_warmup_steps', 10)
        num_training_steps = optim_args.get('num_training_steps', 200)
        num_cycles = optim_args.get('num_cycles', 0.5)
        current_step = optim_args.get('current_step', epoch)

        if current_step < num_warmup_steps:
            lr = learning_rate * float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            lr = learning_rate * max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[5]:


def save_checkpoint(models, optimizers, epoch, save_dir):
    state = {}
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        state[i] = {
            'epoch': epoch,
            'arch': model.__class__.__name__,
            'model_pram': model.state_dict(),
            'optimizer_pram' : optimizer.state_dict(),
        }
    
    path = os.path.join(save_dir, 'checkpoint_epoch_%d.pkl' % epoch)
    torch.save(state, path, pickle_protocol=4)
    return

def load_checkpoint(models, optimizers, epoch, save_dir, use_cuda, gpu_id=None):
    path = os.path.join(save_dir, 'checkpoint_epoch_%d.pkl' % epoch)
    state = torch.load(path, map_location="cpu")
    
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.cpu()
        
        model.load_state_dict(state[i]["model_pram"])
        optimizer.load_state_dict(state[i]["optimizer_pram"])
        
        if use_cuda:
            model.cuda(gpu_id)            
            for optim_state in optimizer.state.values():
                for k, v in optim_state.items():
                    if isinstance(v, torch.Tensor):
                        optim_state[k] = v.cuda()
    return


# In[ ]:


def load_model(model, models_path, model_id):
    device = list(model.state_dict().values())[0].device
    model_state = torch.load(models_path, map_location="cpu")[model_id]
    model_param = model_state["model_pram"]
    model.cpu()
    model.load_state_dict(model_param)
    model.to(device)
    return


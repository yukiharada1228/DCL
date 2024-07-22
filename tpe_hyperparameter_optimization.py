import os
import random
import easydict
import copy
import argparse
import torch
import optuna

from lib import dataset_factory
from lib import models as model_factory
from lib import loss_func
from lib import trainer as trainer_module
from lib import utils

def create_object(config):
    # set seed value
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # create dataset
    train_loader, test_loader = getattr(dataset_factory, config.dataloader.name)(config)    
    
    # define model & loss func & optimizer    
    nets = []
    criterions = []
    optimizers = []
    for model_args in config.models:
        # define model
        net = getattr(model_factory, model_args.name)(**model_args.args)
        net = net.cuda(config.gpu.id)
        
        # load weight
        if model_args.load_weight.path is not None:
            utils.load_model(net, model_args.load_weight.path, model_args.load_weight.model_id)
        
        nets += [net]
        
        # define loss function        
        criterions = []
        for row in config.losses:
            r = []
            for loss in row:
                criterion = getattr(loss_func, loss.name)(loss.args)
                criterion = criterion.cuda(config.gpu.id)
                r += [criterion]
            criterions += [loss_func.TotalLoss(r)]
        
        # define optimizer
        optimizer = getattr(torch.optim, model_args.optim.name)
        optimizers += [optimizer(net.parameters(), **model_args.optim.args)]
    
    # Trainer
    trainer = getattr(trainer_module, config.trainer.name)(config)

    # Logger
    logs = utils.LogManagers(len(config.models), len(train_loader.dataset),
                             config.trainer.epochs, config.dataloader.batch_size)

    return trainer, nets, criterions, optimizers, train_loader, test_loader, logs

def inform_optuna(**kwargs):
    trial = kwargs["_trial"]
    logs = kwargs["_logs"]
    epoch = kwargs["_epoch"]
    
    error = 100 - logs[0]["epoch_log"][epoch]["test_accuracy"]
    trial.report(error, step=epoch)
    
    if trial.should_prune():
        raise optuna.structs.TrialPruned()
    return

def objective(trial):
    # Optunaが提案する学習率
    lr = trial.suggest_loguniform('lr', 3e-3, 1.5e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 3e-2, 2e-1)
    beta1 = trial.suggest_uniform('beta1', 0.85, 0.95)
    amsgrad = trial.suggest_categorical('amsgrad', [True, False])
    
    # 設定の複製と学習率の更新
    trial_config = copy.deepcopy(config)
    trial_config.models[0].optim.args.lr = lr
    trial_config.models[0].optim.args.weight_decay = weight_decay
    trial_config.models[0].optim.args.betas = (beta1, 0.999)
    trial_config.models[0].optim.args.amsgrad = amsgrad
    
    # trialごとの保存ディレクトリを設定
    trial_save_dir = os.path.join(config.trainer.base_dir, f"{trial.number:04}/")
    trial_config.trainer.base_dir = trial_save_dir
    utils.make_dirs(trial_save_dir)
    utils.make_dirs(os.path.join(trial_save_dir, "log"))
    utils.make_dirs(os.path.join(trial_save_dir, "checkpoint"))
    
    # trialごとのconfigを保存
    utils.save_json(trial_config, os.path.join(trial_save_dir, "log", "config.json"))
    
    # オブジェクトの作成
    trainer, nets, criterions, optimizers, train_loader, test_loader, logs = create_object(trial_config)

    # make kwargs
    kwargs = {"_trial": trial,
              "_callback":inform_optuna}

    # set seed
    trial.set_user_attr("seed", config.manualSeed)

    # 訓練の実行
    trainer.train(nets, criterions, optimizers, train_loader, test_loader, logs, trial=trial, **kwargs)

    best_acc = 100 - logs[0]["epoch_log"][config.trainer.epochs]["test_accuracy"]
    
    return best_acc

def main():
    global config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', type=str, default="ResNet32")
    parser.add_argument('--dataset', type=str, choices=["CIFAR10", "CIFAR100"], default="CIFAR100")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./optuna/pre-train/ResNet32/")

    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    manualSeed = 0

    if args.dataset == "CIFAR10":
        DATA_PATH = "./dataset/CIFAR-10/"
        NUM_CLASS = 10
        EPOCHS = 200    
    elif args.dataset == "CIFAR100":
        DATA_PATH = "./dataset/CIFAR-100/"
        NUM_CLASS = 100
        EPOCHS = 200
    
    optim_setting = {
        "name": "AdamW",
        "args":
        {
            "lr": None,
            "betas": None,
            "weight_decay": None,
            "amsgrad": None,
        },
        "scheduler_type": 'cosine',
        "num_warmup_steps": 10,
        "num_training_steps": EPOCHS,
    }

    args_factory = easydict.EasyDict({
        "models": {
            "ResNet32":
            {
                "name": "resnet32",
                "args":
                {
                    "num_classes": NUM_CLASS,
                },
            },
            "ResNet110":
            {
                "name": "resnet110",
                "args":
                {
                    "num_classes": NUM_CLASS,
                },
            },
            "WRN28_2":
            {
                "name": "WideResNet",
                "args":
                {
                    "depth": 28,
                    "num_classes": NUM_CLASS,
                    "widen_factor": 2,
                    "dropRate": 0.0
                },
            },
        },
        "losses":
        {
            "IndepLoss":
            {
                "name": "IndependentLoss",
                "args": 
                {
                    "loss_weight": 1,
                    "gate": 
                    {
                        "name": "ThroughGate",
                        "args": {},
                    },
                },
            },
        }
    })

    model = args_factory.models[args.target_model]

    config = easydict.EasyDict(
        {
            "doc": "",
            "manualSeed": manualSeed,
            "dataloader": 
            {
                "name": args.dataset,
                "data_path": DATA_PATH,
                "num_class": NUM_CLASS,
                "batch_size": 128,
                "workers": 10,
                "train_shuffle": True,
                "train_drop_last": True, 
                "test_shuffle": True,
                "test_drop_last": False, 
            },
            "trainer": 
            {
                "name": "ClassificationTrainer",
                "start_epoch": 1,
                "epochs": EPOCHS,
                "saving_interval": EPOCHS,
                "base_dir": "./",
            },
            "models": 
            [           
                {
                    "name": model.name,
                    "args": model.args,
                    "load_weight":
                    {
                        "path": None,
                        "model_id": 0,
                    },
                    "optim": optim_setting,
                } 
            ],
            "losses":        
            [
                [
                    args_factory.losses["IndepLoss"]
                ]
            ],
            "gpu":
            {
                "use_cuda": True,
                "ngpu": 1,
                "id": args.gpu_id,
            },
        })

    config.trainer.base_dir = args.save_dir
    utils.make_dirs(config.trainer.base_dir)

    # Optunaによる最適化の実行
    db_path = os.path.join(args.save_dir, "optuna.db")
    study = optuna.create_study(storage=f"sqlite:///{db_path}",
                                study_name='experiment01',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.NopPruner(),
                                direction="minimize",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100, timeout=None, n_jobs=1)

if __name__ == '__main__':
    main()

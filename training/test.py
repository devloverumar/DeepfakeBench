"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import torchattacks
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
# import sys

# sys.path.append('./../')

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, attack_models, data_loader):
    prediction_lists = []
    attack_prediction_lists = []
    # feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions, attack_predictions = inference(model, attack_models, data_dict, label)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        predictions = torch.softmax(predictions, dim=1)[:, 1]
        prediction_lists += list(predictions.cpu().detach().numpy())

        attack_predictions = torch.softmax(attack_predictions, dim=1)[:, 1]
        attack_prediction_lists += list(attack_predictions.cpu().detach().numpy())
        # prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        # feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(attack_prediction_lists), np.array(label_lists)#,np.array(feature_lists)
    
def test_epoch(model, attack_models, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, attack_predictions_nps, label_nps = test_one_dataset(model, attack_models, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset

        attack_metric_one_dataset = get_test_metrics(y_pred=attack_predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])

        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

        # info for each dataset
        tqdm.write(f"dataset: {key} after attack")
        for k, v in attack_metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

# @torch.no_grad()
def inference(model, attack_models, data_dict,label, attack = True ):
    if attack == True:
        attack_model = attack_models[0].eval()
        attack = torchattacks.FGSM(attack_model, eps=8/255)
        # print(f"Images before attack {data_dict['image'].shape}")
        images = data_dict['image']
        images.requires_grad = True
        attacked_images = attack(images, label)
        # data_dict['image'] = attacked_images
        # print(f"After attack, image shape: {attacked_images.shape}")
    predictions = model(images, inference=True)    
    attack_predictions = model(attacked_images, inference=True)
    return predictions, attack_predictions

def get_attack_models(config):
    model_names = [
        {"xception":"./weights/xception_best.pth"},
        # {"core":"./weights/core_best.pth"},
        # {"efficientnetb4":"./weights/effnb4_best.pth"},
        # {"ucf":"./weights/xception_best.pth"},
        ]
    models = []

    for model_dict in model_names:
        for model_name, model_weights in model_dict.items():  # Extract key-value pair
            # Load the configuration file        # make_config
            with open(f'./config/detector/{model_name}.yaml', 'r') as f:
                config3 = yaml.safe_load(f)
            config.update(config3)

            model_class = DETECTOR[model_name]
            model = model_class(config).to(device)
            ckpt = torch.load(model_weights, map_location=device)
            model.load_state_dict(ckpt, strict=True)
            models.append(model)
            print(f'===> Load checkpoint done for {model_name}!')
    return models


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    print(f"Model name in test :{config['model_name']}")
    print(f'Detector keys: {DETECTOR.data.keys()}')


    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # Attack models
    attack_models = get_attack_models(config)

    # start testing
    best_metric = test_epoch(model, attack_models, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()

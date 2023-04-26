import os
import pdb

import pandas as pd

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm as progress_bar
import argparse
import numpy as np
import torch
from torchvision import models
import torch.nn.functional as F
from taillight_dataset import TaillightDataset
from copy import deepcopy
import time
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import transforms
from torch.utils.data import DataLoader
import statistics
import matplotlib.pyplot as plt
import cv2



def compute_iou(outputs, labels):

    corner_predictions = 64*outputs + 64
    corner_labels = 64*labels + 64

    batch_iou = 0
    correct_predictions_25 = 0
    correct_predictions_50 = 0
    num_examples = 0

    for corner_prediction, corner_label in zip(corner_predictions, corner_labels):

        x_corners = [x for x in corner_prediction[::2] if x != 64]
        y_corners = [y for y in corner_prediction[1::2] if y != 64]

        x_corners_gt = [x for x in corner_label[::2] if x != 64]
        y_corners_gt = [y for y in corner_label[1::2] if y != 64]

        # Need at least two points to calculate IOU
        if not x_corners_gt or not y_corners_gt or len(x_corners_gt) < 2 or len(y_corners_gt) < 2:
            continue

        # Get the area for the predicted box
        pred_xmin, pred_xmax = min(x_corners), max(x_corners)
        pred_ymin, pred_ymax = min(y_corners), max(y_corners)
        pred_height = pred_ymax - pred_ymin
        pred_width = pred_xmax - pred_xmin
        pred_area = pred_height*pred_width

        # Get the area for the ground truth box
        gt_xmin, gt_xmax = min(x_corners_gt), max(x_corners_gt)
        gt_ymin, gt_ymax = min(y_corners_gt), max(y_corners_gt)
        gt_height = gt_ymax - gt_ymin
        gt_width = gt_xmax - gt_xmin
        gt_area = gt_height*gt_width

        # Get the coordinates of the intersection of the bounding boxes
        intersection_xmin = max(pred_xmin, gt_xmin)
        intersection_ymin = max(pred_ymin, gt_ymin)
        intersection_xmax = min(pred_xmax, gt_xmax)
        intersection_ymax = min(pred_ymax, gt_ymax)

        intersection_area = max(0, intersection_xmax - intersection_xmin)*max(0, intersection_ymax - intersection_ymin)
        iou = intersection_area / (pred_area + gt_area - intersection_area)

        if iou > 0.5:
            correct_predictions_50 += 1
        if iou > 0.25:
            correct_predictions_25 += 1

        batch_iou += iou.item()
        num_examples += 1

    return batch_iou/num_examples, correct_predictions_25/num_examples, correct_predictions_50/num_examples

def validate_model(val_loader, model):
    """
    Evaluates the model on a given dataset
    """

    model.eval()
    num_batches = 0
    running_iou = 0
    running_accuracy_25 = 0
    running_accuracy_50 = 0

    for step, (inputs, labels) in progress_bar(enumerate(val_loader), total=len(val_loader)):

        if torch.cuda.is_available():
            # inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = inputs.float(), labels.float()

        num_batches += 1
        outputs = model(inputs)
        iou, accuracy_25, accuracy_50 = compute_iou(outputs, labels)
        running_iou += iou
        running_accuracy_25 += accuracy_25
        running_accuracy_50 += accuracy_50

    avg_iou = running_iou / num_batches
    avg_accuracy_25 = 100*running_accuracy_25 / num_batches
    avg_accuracy_50 = 100 * running_accuracy_50 / num_batches
    print('IOU: ' + str(avg_iou))
    print('Average % Accuracy@25: ' + str(avg_accuracy_25))
    print('Average % Accuracy@50: ' + str(avg_accuracy_50))
    print()
    return avg_iou, avg_accuracy_25, avg_accuracy_50



if __name__ == '__main__':

    timestr = '20230417-195437'

    # light_types = ['Left_Rear', 'Right_Rear', 'Left_Front', 'Right_Front']
    # test_json_list = ['keypoints_vcrop_left_rear_test.json', 'keypoints_vcrop_right_rear_test.json',
    #                   'keypoints_vcrop_left_front_test.json', 'keypoints_vcrop_right_front_test.json']
    # test_img_dir = ['test_left_rear', 'test_right_rear', 'test_left_front', 'test_right_front']
    light_types = ['Right_Front']
    test_json_list = ['keypoints_vcrop_right_front_test.json']
    test_img_dir = ['test_right_front']

    iou = 0
    accuracy = 0

    for light, test_json, test_dir in zip(light_types, test_json_list, test_img_dir):

        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=model.fc.in_features, out_features=8),
            torch.nn.Tanh()
        )

        test_dataset = TaillightDataset(
            json_file=os.path.join('data-apollocar3d', 'annotations', 'separate_lights_3', test_json),
            img_dir=os.path.join('cropped-internal-data', 'separate_lights_3', test_dir),
            regularizer=64,
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            ])
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, pin_memory=True)

        # Load in the trained model
        model.load_state_dict(torch.load(os.path.join('models', timestr + '_' + light + '.pth')))

        # if torch.cuda.is_available():
        #     model = model.cuda()

        validate_model(test_loader, model)


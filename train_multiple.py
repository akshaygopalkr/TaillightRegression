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


def save_model(file_folder, model, type):
    # Save the model into the designated folder
    path = os.path.join(file_folder, timestr + '_' + type + '.pth')
    torch.save(model, path)


def show_results(outputs, targets, inputs, start_index, light_type):
    img_idx = start_index
    if not os.path.isdir(os.path.join('results', timestr)):
        os.system('mkdir ' + os.path.join('results', timestr))

    for i in range(inputs.size()[0]):
        target, output, image = targets.cpu().detach()[i].numpy(), outputs.cpu().detach()[i].numpy(), \
            inputs.cpu().detach()[i].numpy()
        image = image.reshape((128, 128, 3))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output_c1 = (
            int(64 + output[0] * args.regression_regularizer), int(64 + output[1] * args.regression_regularizer))
        output_c2 = (
            int(64 + output[2] * args.regression_regularizer), int(64 + output[3] * args.regression_regularizer))
        output_c3 = (
            int(64 + output[4] * args.regression_regularizer), int(64 + output[5] * args.regression_regularizer))
        output_c4 = (
            int(64 + output[6] * args.regression_regularizer), int(64 + output[7] * args.regression_regularizer))

        target_c1 = (
            int(64 + target[0] * args.regression_regularizer), int(64 + target[1] * args.regression_regularizer))
        target_c2 = (
            int(64 + target[2] * args.regression_regularizer), int(64 + target[3] * args.regression_regularizer))
        target_c3 = (
            int(64 + target[4] * args.regression_regularizer), int(64 + target[5] * args.regression_regularizer))
        target_c4 = (
            int(64 + target[6] * args.regression_regularizer), int(64 + target[7] * args.regression_regularizer))

        radius = 2

        # Draw the target and output keypoints
        cv2.circle(image, output_c1, radius, (0, 0, 255), -1)
        cv2.circle(image, output_c2, radius, (0, 0, 255), -1)
        cv2.circle(image, output_c3, radius, (0, 0, 255), -1)
        cv2.circle(image, output_c4, radius, (0, 0, 255), -1)
        cv2.circle(image, target_c1, radius, (0, 255, 0), -1)
        cv2.circle(image, target_c2, radius, (0, 255, 0), -1)
        cv2.circle(image, target_c3, radius, (0, 255, 0), -1)
        cv2.circle(image, target_c4, radius, (0, 255, 0), -1)
        cv2.circle(image, (64, 64), radius, (255, 0, 0), -1)

        cv2.line(image, output_c1, output_c2, (0, 0, 255), 1)
        cv2.line(image, output_c1, output_c3, (0, 0, 255), 1)
        cv2.line(image, output_c2, output_c4, (0, 0, 255), 1)
        cv2.line(image, output_c3, output_c4, (0, 0, 255), 1)

        cv2.line(image, target_c1, target_c2, (0, 255, 0), 1)
        cv2.line(image, target_c1, target_c3, (0, 255, 0), 1)
        cv2.line(image, target_c2, target_c4, (0, 255, 0), 1)
        cv2.line(image, target_c3, target_c4, (0, 255, 0), 1)

        cv2.imwrite(os.path.join('results', timestr, light_type + str(img_idx) + '.png'), image)
        img_idx += 1


def compute_loss(outputs, targets):

    # Reshape the outputs and targets from N x 8 to N x 4 x 2
    outputs = outputs.reshape((outputs.shape[0], 4, 2))
    targets = targets.reshape((targets.shape[0], 4, 2))

    # Make a mask of all the non-zero targets
    mask = (targets != 0).float()
    mask = torch.where(mask == 0, 1e-8, mask)

    regularizer_mask = torch.max(torch.sum(mask, dim=1)[:, 0], torch.tensor(1.)).float()

    loss = torch.mean(
        torch.sum(torch.sqrt(torch.sum((outputs * mask - targets) ** 2, dim=2)), dim=1) / regularizer_mask)

    return loss


def compute_average_distance(outputs, targets):

    # Reshape the outputs and targets from N x 8 to N x 4 x 2
    outputs = outputs.reshape((outputs.shape[0], 4, 2))
    targets = targets.reshape((targets.shape[0], 4, 2))

    # Make a mask of all the non-zero targets
    mask = (targets != 0).float()

    # This will be a vector with 4 elements with each item representing the distance between the output and target for a
    # certain corner
    distances = torch.sum(torch.sqrt(torch.sum((outputs * mask - targets) ** 2, dim=2)), dim=0)

    return args.regression_regularizer * torch.mean(distances / torch.sum(mask[:, :, 0], dim=0))


def compute_percent_error(outputs, targets):

    # Compute coordinates of target and predicted corners
    normal_targets = 64 + targets * args.regression_regularizer

    x_coord_targets = normal_targets[:, ::2]
    y_coord_targets = normal_targets[:, 1::2]

    # Reshape the outputs and targets from N x 8 to N x 4 x 2
    outputs = outputs.reshape((outputs.shape[0], 4, 2))
    targets = targets.reshape((targets.shape[0], 4, 2))

    # Make a mask of all the non-zero targets
    mask = (targets != 0).float()

    distances = []

    for x_corners, y_corners in zip(x_coord_targets, y_coord_targets):

        x_corners = [x for x in x_corners if x != 64]
        y_corners = [y for y in y_corners if y != 64]

        if not x_corners or not y_corners:
            distances.append(1)
            continue

        height = max(y_corners) - min(y_corners)
        width = max(x_corners) - min(x_corners)

        max_distance = (height ** 2 + width ** 2) ** 0.5
        distances.append(max_distance)

    distances = torch.tensor(distances).reshape(outputs.shape[0], 1).cuda()
    regularizer_mask = torch.max(torch.sum(mask, dim=1)[:, 0], torch.tensor(1.)).float()

    return torch.mean(torch.sum(64 * torch.sqrt(torch.sum((outputs * mask - targets) ** 2, dim=2)) / distances,
                                dim=1) / regularizer_mask).item()


def validate_model(val_loader, model, save_results=False, light_type=None):
    """
    Evaluates the model on a given dataset
    """

    model.eval()
    losses = 0
    running_distance = 0
    running_error = 0
    num_batches = 0

    for step, (inputs, labels) in progress_bar(enumerate(val_loader), total=len(val_loader)):

        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = inputs.float(), labels.float()

        num_batches += 1
        outputs = model(inputs)

        if save_results:
            show_results(outputs, labels, inputs, (num_batches - 1) * args.batch_size, light_type)

        loss = compute_loss(outputs, labels)
        running_distance += compute_average_distance(outputs, labels).item()
        running_error += compute_percent_error(outputs, labels)
        losses += loss.item()

    avg_loss = losses / num_batches
    avg_distance = running_distance / num_batches
    avg_error = 100 * running_error / num_batches

    return avg_loss, avg_distance, avg_error


def train(model, train_loader, val_loader, args, type):
    """
    Performs training without SWA
    :param train_loader: training dataset
    :param val_loader: validation dataset
    """

    # Initialize the cross entropy and Adam optimizer to update weights
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate)

    epochs = args.epochs

    # Used to save the best model
    best_model = None
    best_val_loss = None
    best_training_loss = None
    best_training_distance = None
    best_validation_distance = None
    best_training_error = None
    best_validation_error = None
    training_loss = []

    for epoch_count in range(epochs):

        # Initializing running accuracy and loss statistics
        losses = 0
        running_distance = 0
        running_error = 0

        model.train()
        num_batches = 0

        for step, (inputs, labels) in progress_bar(enumerate(train_loader), total=len(train_loader)):

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.float(), labels.float()

            num_batches += 1
            outputs = model(inputs)

            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()  # backprop to update the weights

            swa_model.update_parameters(model)
            swa_scheduler.step()

            running_distance += compute_average_distance(outputs, labels).item()
            running_error += compute_percent_error(outputs, labels)

            optimizer.zero_grad()
            losses += loss.item()

        avg_loss = losses / num_batches
        avg_distance = running_distance / num_batches
        avg_error = 100 * running_error / num_batches
        val_loss, val_distance, val_error = validate_model(val_loader, model)
        print()
        print('Epoch ' + str(epoch_count + 1) + ':')
        print('Training Loss: ' + str(avg_loss))
        print('Training Average Distance: ' + str(avg_distance))
        print('Training Average % Error: ' + str(avg_error))
        print('Validation Loss: ' + str(val_loss))
        print('Validation Average Distance ' + str(val_distance))
        print('Validation Average % Error: ' + str(val_error))
        print()

        # Update the model if it achieves a higher
        # accuracy then the previous model
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())

        best_training_loss = avg_loss if best_training_loss is None else min(best_training_loss, avg_loss)
        best_training_distance = avg_distance if best_training_distance is None else min(best_training_distance,
                                                                                         avg_distance)
        best_validation_distance = val_distance if best_validation_distance is None else min(best_validation_distance,
                                                                                             val_distance)
        best_validation_error = val_error if best_validation_error is None else min(best_validation_error, val_error)
        best_training_error = avg_error if best_training_error is None else min(best_training_error, avg_error)

    # Save the best model from training
    save_model('models', best_model, type)
    return best_training_loss, best_val_loss, best_training_distance, best_validation_distance, best_training_error, \
        best_validation_error


def save_experiment(args, statistics, model_type):
    """
    Saves the experiment results to a csv
    :param args: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """
    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [args.learning_rate],
        'Batch size': [args.batch_size],
        'Weight decay': [args.weight_decay],
        'Model type': [args.model_type],
        'Epochs': [args.epochs],
        'Minimum Training MSE': [statistics[0]],
        'Minimum Validation MSE': [statistics[1]],
        'Testing MSE': [statistics[2]],
        'Minimum Training Average Distance': [statistics[3]],
        'Minimum Validation Average Distance': [statistics[4]],
        'Testing Average Distance': [statistics[5]],
        'Minimum Training % Error': [statistics[6]],
        'Minimum Validation % Error:': [statistics[7]],
        'Minimum Testing % Error:': [statistics[8]]
    }

    file_name = 'results_' + model_type + '.csv'

    trial_dict = pd.DataFrame(trial_dict)
    need_header = not os.path.exists(file_name)

    if need_header:
        trial_dict.to_csv(file_name, index=False, header=need_header)
    else:
        trial_dict.to_csv(file_name, mode='a', index=False, header=need_header)


def params():
    """
    Loads the hyperparameters passed into the command line
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-3, type=float,
                        help="Model learning rate starting point.")
    parser.add_argument("--regression-regularizer", default=64, type=int)
    parser.add_argument("--batch-size", default=256, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=1.5e-4, type=float,
                        help="L2 Regularization")
    parser.add_argument("--epochs", default=25, type=int,
                        help="Number of epochs to train for")
    parser.add_argument('--model-type', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                                 'densenet121', 'densenet169'], default='resnet50')
    parser.add_argument('--image-padding', type=int, default=64)
    args = parser.parse_args()

    # Pretrained model types
    model = getattr(models, args.model_type)
    model = model(pretrained=True)

    # Replace last layer with linear layer that has 4 out features
    fc = getattr(model, list(model.named_children())[-1][0])

    if 'resnet' in args.model_type:
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=fc.in_features, out_features=8),
            torch.nn.Tanh()
        )
    else:
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=fc.in_features, out_features=8),
            torch.nn.Tanh()
        )

    return args, model


if __name__ == '__main__':

    timestr = time.strftime("%Y%m%d-%H%M%S")

    train_json_list = ['keypoints_vcrop_left_rear_train.json', 'keypoints_vcrop_right_rear_train.json',
                       'keypoints_vcrop_left_front_train.json', 'keypoints_vcrop_right_front_train.json']
    val_json_list = ['keypoints_vcrop_left_rear_val.json', 'keypoints_vcrop_right_rear_val.json',
                     'keypoints_vcrop_left_front_val.json', 'keypoints_vcrop_right_front_val.json']
    test_json_list = ['keypoints_vcrop_left_rear_test.json', 'keypoints_vcrop_right_rear_test.json',
                      'keypoints_vcrop_left_front_test.json', 'keypoints_vcrop_right_front_test.json']

    train_img_dir = ['train_left_rear', 'train_right_rear', 'train_left_front', 'train_right_front']
    val_img_dir = ['val_left_rear', 'val_right_rear', 'val_left_front', 'val_right_front']
    test_img_dir = ['test_left_rear', 'test_right_rear', 'test_left_front', 'test_right_front']

    light_type = ['Left_Rear', 'Right_Rear', 'Left_Front', 'Right_Front']

    full_statistics = [0]*9

    for train_json, val_json, test_json, train_dir, val_dir, test_dir, light in zip(train_json_list, val_json_list,
                                                                             test_json_list, train_img_dir, val_img_dir,
                                                                             test_img_dir, light_type):

        print('----------------------------  Training ' + light + ' ----------------------------')
        args, model = params()

        # Put model on GPU
        if torch.cuda.is_available():
            model = model.cuda()

        # Initialize the datasets and dataloader
        train_dataset = TaillightDataset(
            json_file=os.path.join('data-apollocar3d', 'annotations', 'separate_lights_3', train_json),
            img_dir=os.path.join('cropped-internal-data', 'separate_lights_3', train_dir),
            regularizer=args.regression_regularizer,
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            ])
        )

        val_dataset = TaillightDataset(
            json_file=os.path.join('data-apollocar3d', 'annotations','separate_lights_3',  val_json),
            img_dir=os.path.join('cropped-internal-data', 'separate_lights_3', val_dir),
            regularizer=args.regression_regularizer,
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            ])
        )

        test_dataset = TaillightDataset(
            json_file=os.path.join('data-apollocar3d', 'annotations', 'separate_lights_3', test_json),
            img_dir=os.path.join('cropped-internal-data', 'separate_lights_3', test_dir),
            regularizer=args.regression_regularizer,
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            ])
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        best_training_loss, best_validation_loss, best_training_distance, best_validation_distance, \
            best_training_error, best_val_error = train(model, train_loader, val_loader, args, light)

        # Load the best model from training
        best_model = deepcopy(model)
        best_model.load_state_dict(torch.load(os.path.join('models', timestr + '_' + light + '.pth')))

        if torch.cuda.is_available():
            best_model = best_model.cuda()

        test_loss, test_distance, test_error = validate_model(test_loader, best_model, True, light)
        stats = [best_training_loss, best_validation_loss, test_loss, best_training_distance, best_validation_distance,
                 test_distance, best_training_error, best_val_error, test_error]
        full_statistics = [full_statistics[i] + stats[i] for i in range(len(stats))]

        save_experiment(args, stats, light)
        print('------------------------------------------------------------------')

    full_statistics = [stat/4 for stat in full_statistics]
    args, _ = params()
    save_experiment(args, full_statistics, 'combined')

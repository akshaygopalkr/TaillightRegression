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
import pdb
import cv2


def save_model(file_folder, model):
    # Save the model into the designated folder
    path = os.path.join(file_folder, timestr + '.pth')
    torch.save(model, path)


def show_results(outputs, targets, inputs, start_index):
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

        cv2.imwrite(os.path.join('results', timestr, str(img_idx) + '.png'), image)
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
    normal_outputs = outputs * 64
    normal_targets = targets * 64

    # Make a mask of all the non-zero targets
    mask = (targets != 0).float()
    mask = torch.where(mask == 0, 1e-8, mask)

    # Used so we don't divide by 0 for any non-visible corners
    regularizer_targets = torch.where(targets < 1e-8, 1, normal_targets)

    return 100 * torch.mean(
        torch.sum(torch.abs(torch.div(normal_outputs * mask - normal_targets, regularizer_targets)), dim=0) / torch.sum(
            mask, dim=0))


def validate_model(val_loader, model, save_results=False):
    """
    Evaluates the model on a given dataset
    """

    model.eval()
    losses = 0
    running_distance = 0
    num_batches = 0

    for step, (inputs, labels) in progress_bar(enumerate(val_loader), total=len(val_loader)):

        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = inputs.float(), labels.float()

        num_batches += 1
        outputs = model(inputs)

        if save_results:
            show_results(outputs, labels, inputs, (num_batches - 1) * args.batch_size)

        loss = compute_loss(outputs, labels)
        running_distance += compute_average_distance(outputs, labels).item()
        loss.backward()
        losses += loss.item()

    avg_loss = losses / num_batches
    avg_distance = running_distance / num_batches

    return avg_loss, avg_distance


def train(model, train_loader, val_loader, args):
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
    training_loss = []

    for epoch_count in range(epochs):

        # Initializing running accuracy and loss statistics
        losses = 0
        running_distance = 0

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

            optimizer.zero_grad()
            losses += loss.item()

        avg_loss = losses / num_batches
        avg_distance = running_distance / num_batches
        val_loss, val_distance = validate_model(val_loader, model)
        print()
        print('----------------------------  Epoch ' + str(epoch_count + 1) + ' ----------------------------')
        print('Training Loss: ' + str(avg_loss))
        print('Training Average Distance: ' + str(avg_distance))
        print('Validation Loss: ' + str(val_loss))
        print('Validation Average Distance ' + str(val_distance))
        print('------------------------------------------------------------------')

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

    # Save the best model from training
    save_model('models', best_model)
    return best_training_loss, best_val_loss, best_training_distance, best_validation_distance


def save_experiment(args, statistics):
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
    }

    trial_dict = pd.DataFrame(trial_dict)
    need_header = not os.path.exists('results.csv')

    if need_header:
        trial_dict.to_csv('results.csv', index=False, header=need_header)
    else:
        trial_dict.to_csv('results.csv', mode='a', index=False, header=need_header)


def params():
    """
    Loads the hyperparameters passed into the command line
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-3, type=float,
                        help="Model learning rate starting point.")
    parser.add_argument("--regression-regularizer", default=64, type=int)
    parser.add_argument("--batch-size", default=512, type=int,
                        help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=1e-4, type=float,
                        help="L2 Regularization")
    parser.add_argument("--epochs", default=10,  type=int,
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

    args, model = params()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Put model on GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize the datasets and dataloader
    train_dataset = TaillightDataset(
        json_file=os.path.join('data-apollocar3d', 'annotations', 'reflect_keypoints_train.json'),
        img_dir=os.path.join('cropped-internal-data', 'train_reflect'),
        regularizer=args.regression_regularizer,
        transform=transforms.Compose([
            transforms.Resize((128, 128))
        ])
    )
    val_dataset = TaillightDataset(
        json_file=os.path.join('data-apollocar3d', 'annotations', 'reflect_keypoints_val.json'),
        img_dir=os.path.join('cropped-internal-data', 'val_reflect'),
        regularizer=args.regression_regularizer,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
        ])
    )
    test_dataset = TaillightDataset(
        json_file=os.path.join('data-apollocar3d', 'annotations', 'reflect_keypoints_test.json'),
        img_dir=os.path.join('cropped-internal-data', 'test_reflect'),
        regularizer=args.regression_regularizer,
        transform=transforms.Compose([
            transforms.Resize((128, 128))
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    best_training_loss, best_validation_loss, best_training_distance, best_validation_distance = train(model,
                                                                                                       train_loader,
                                                                                                       val_loader, args)

    # Load the best model from training
    best_model = deepcopy(model)
    best_model.load_state_dict(torch.load(os.path.join('models', timestr + '.pth')))

    if torch.cuda.is_available():
        best_model = best_model.cuda()

    test_loss, test_distance = validate_model(test_loader, best_model, True)
    stats = [best_training_loss, best_validation_loss, test_loss, best_training_distance, best_validation_distance,
             test_distance]

    save_experiment(args, stats)

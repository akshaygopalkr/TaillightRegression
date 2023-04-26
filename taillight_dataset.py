import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
import json
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
from PIL import Image

class TaillightDataset(Dataset):

    def __init__(self, json_file, img_dir, regularizer, transform=None):

        with open(json_file) as f:
            self.annotations = json.load(f)

        self.transform = transform
        self.regularizer = regularizer
        self.image_dir = img_dir

    def find_corresponding_image(self, image_id):

        for img in os.listdir(os.path.join(self.image_dir)):
            if img.find(image_id) != -1:
                return os.path.join(self.image_dir, img)
        return None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        # Access the annotation
        annotation = self.annotations[item]

        # Find the image for this annotation
        matching_image = cv2.imread(os.path.join(self.image_dir, (str(annotation['image_id'])) + '.jpg'))
        # matching_image = cv2.cvtColor(matching_image, cv2.COLOR_BGR2RGB)

        matching_image_2 = cv2.cvtColor(matching_image, cv2.COLOR_BGR2HSV)

        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 50, 50])
        upper1 = np.array([10, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([170, 50, 50])
        upper2 = np.array([180, 255, 255])

        # Create a mask for the red color
        mask1 = cv2.inRange(matching_image_2, lower1, upper1)
        mask2 = cv2.inRange(matching_image_2, lower2, upper2)
        mask = mask1 + mask2

        # Apply the mask to the original image
        new_image = cv2.bitwise_and(matching_image_2, matching_image_2, mask=mask)

        # Display the original image and the extracted red pixels
        cv2.imshow("Original Image", matching_image)
        cv2.imshow("Red Pixels", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        keypoints = annotation['keypoints']
        center_x, center_y = keypoints[-3], keypoints[-2]

        # Convert padded image to tensor
        # image = torch.tensor(matching_image).float()
        image = torch.tensor(matching_image).float()
        image = torch.reshape(image, (3, image.size()[0], image.size()[1]))

        if self.transform:
            image = self.transform(image)

        visibilities = [keypoints[2], keypoints[5], keypoints[8], keypoints[11]]

        if all([visibility != 0 for visibility in visibilities]):

            # Sort the keypoints by the x-coordinate
            sorted_keypoints = sorted([(keypoints[0], keypoints[1], keypoints[2]), (keypoints[3], keypoints[4], keypoints[5]), (keypoints[6], keypoints[7], keypoints[8]),
                                (keypoints[9], keypoints[10], keypoints[11])], key=lambda x: x[0])

            upper_left = min(sorted_keypoints[:2], key=lambda x: x[1])
            bottom_left = max(sorted_keypoints[:2], key=lambda x: x[1])
            upper_right = min(sorted_keypoints[2:], key=lambda x: x[1])
            bottom_right = max(sorted_keypoints[2:], key=lambda x: x[1])


            # Labels are going to be the distance of each corner to the center keypoint
            regression_labels = torch.tensor([
                (upper_left[0] - center_x)/self.regularizer if upper_left[2] != 0 else 0,
                (upper_left[1] - center_y)/self.regularizer if upper_left[2] != 0 else 0,
                (upper_right[0] - center_x)/self.regularizer if upper_right[2] != 0 else 0,
                (upper_right[1] - center_y)/self.regularizer if upper_right[2] != 0 else 0,
                (bottom_left[0] - center_x)/self.regularizer if bottom_left[2] != 0 else 0,
                (bottom_left[1] - center_y)/self.regularizer if bottom_left[2] != 0 else 0,
                (bottom_right[0] - center_x)/self.regularizer if bottom_right[2] != 0 else 0,
                (bottom_right[1] - center_y)/self.regularizer if bottom_right[2] != 0 else 0
            ])

        else:

            regression_labels = torch.tensor([
                (keypoints[0] - center_x)/self.regularizer if keypoints[2] != 0 else 0,
                (keypoints[1] - center_y)/self.regularizer if keypoints[2] != 0 else 0,
                (keypoints[3] - center_x)/self.regularizer if keypoints[5] != 0 else 0,
                (keypoints[4] - center_y)/self.regularizer if keypoints[5] != 0 else 0,
                (keypoints[6] - center_x)/self.regularizer if keypoints[8] != 0 else 0,
                (keypoints[7] - center_y)/self.regularizer if keypoints[8] != 0 else 0,
                (keypoints[9] - center_x)/self.regularizer if keypoints[11] != 0 else 0,
                (keypoints[10] - center_y)/self.regularizer if keypoints[11] != 0 else 0,
            ])

        return image, regression_labels


if __name__ == '__main__':

    train_dir = os.path.join('data-apollocar3d', 'images', 'train')
    train_file = os.path.join('data-apollocar3d', 'annotations', 'keypoints_train.json')
    dataset = TaillightDataset(json_file=train_file, img_dir=train_dir)


    for i in range(len(dataset)):
        dataset[i]








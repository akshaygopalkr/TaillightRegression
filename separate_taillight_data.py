import json
import cv2
import numpy as np
from copy import deepcopy
import os
import random

internals = {'left_front_car_light': [0,1,2,3],  'left_rear_car_light': [22,23,25,26], 'right_rear_car_light': [31,32,34,35],  'right_front_car_light': [54,55,56,57]}

def separate_annotations(annotations):

    new_annotations = []

    left_front_list = []
    left_rear_list = []
    right_rear_list = []
    right_front_list = []

    for annotation in annotations:
        keypoints = np.reshape(np.array(annotation['keypoints']), (6, 5, 3))

        # We don't care about the license plates
        keypoints = keypoints[:4]

        counter = 0

        # Looking at each 5x3 matrix for the lights
        for keypoint in keypoints:


            # If the center is visible then add it to
            if keypoint[4][2] != 0:

                new_annotation = deepcopy(annotation)
                new_annotation['keypoints'] = list(keypoint.flatten())

                if counter == 0:
                    left_front_list.append(new_annotation)
                elif counter == 1:
                    left_rear_list.append(new_annotation)
                elif counter == 2:
                    right_rear_list.append(new_annotation)
                else:
                    right_front_list.append(new_annotation)

            counter += 1

    return left_front_list, left_rear_list, right_rear_list, right_front_list

if __name__ == '__main__':

    with open(os.path.join('data-apollocar3d', 'annotations', 'filtered_24kptreg_train.json')) as f:
        data = json.load(f)['annotations']

    with open(os.path.join('data-apollocar3d', 'annotations', 'filtered_24kptreg_val.json')) as f:
        data.extend(json.load(f)['annotations'])

    left_front_list, left_rear_list, right_rear_list, right_front_list = separate_annotations(data)

    print(len(left_front_list), len(left_rear_list), len(right_rear_list), len(right_front_list))


    random.shuffle(left_front_list)
    random.shuffle(left_rear_list)
    random.shuffle(right_rear_list)
    random.shuffle(right_front_list)


    left_front_list_train = left_front_list[:int(len(left_front_list)*0.8)]
    left_rear_list_train = left_rear_list[:int(len(left_rear_list) * 0.8)]
    right_front_list_train = right_front_list[:int(len(right_front_list) * 0.8)]
    right_rear_list_train = right_rear_list[:int(len(right_rear_list) * 0.8)]

    print(len(left_front_list_train), len(left_rear_list_train), len(right_front_list_train), len(right_rear_list_train))

    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_left_rear_train.json'), 'w') as f:
    #     json.dump(left_rear_list_train, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_left_front_train.json'), 'w') as f:
    #     json.dump(left_front_list_train, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_right_front_train.json'), 'w') as f:
    #     json.dump(right_front_list_train, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_right_rear_train.json'), 'w') as f:
    #     json.dump(right_rear_list_train, f)

    left_front_list_val = left_front_list[int(len(left_front_list) * 0.8):int(len(left_front_list) * 0.9)]
    left_rear_list_val = left_rear_list[int(len(left_rear_list) * 0.8):int(len(left_rear_list) * 0.9)]
    right_front_list_val = right_front_list[int(len(right_front_list) * 0.8):int(len(right_front_list) * 0.9)]
    right_rear_list_val = right_rear_list[int(len(right_rear_list) * 0.8):int(len(right_rear_list) * 0.9)]

    print(len(left_front_list_val), len(left_rear_list_val), len(right_front_list_val),
          len(right_rear_list_val))

    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_left_rear_val.json'), 'w') as f:
    #     json.dump(left_rear_list_val, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_left_front_val.json'), 'w') as f:
    #     json.dump(left_front_list_val, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_right_front_val.json'), 'w') as f:
    #     json.dump(right_front_list_val, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_right_rear_val.json'), 'w') as f:
    #     json.dump(right_rear_list_val, f)

    left_front_list_test = left_front_list[int(len(left_front_list) * 0.9):]
    left_rear_list_test = left_rear_list[int(len(left_rear_list) * 0.9):]
    right_front_list_test = right_front_list[int(len(right_front_list) * 0.9):]
    right_rear_list_test = right_rear_list[int(len(right_rear_list) * 0.9):]

    print(len(left_front_list_test), len(left_rear_list_test), len(right_front_list_test),
          len(right_rear_list_test))

    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_left_rear_test.json'), 'w') as f:
    #     json.dump(left_rear_list_test, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_left_front_test.json'), 'w') as f:
    #     json.dump(left_front_list_test, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_right_front_test.json'), 'w') as f:
    #     json.dump(right_front_list_test, f)
    #
    # with open(os.path.join('data-apollocar3d', 'annotations', 'separate_lights', 'keypoints_right_rear_test.json'), 'w') as f:
    #     json.dump(right_rear_list_test, f)

import json
import os
import numpy as np
from copy import deepcopy
import cv2
import argparse
from reflect_code import reflect_image_and_keypoints
import pdb

internals = {'left_front_car_light': [0, 1, 2, 3], 'left_rear_car_light': [22, 23, 25, 26],
             'right_rear_car_light': [31, 32, 34, 35], 'right_front_car_light': [54, 55, 56, 57]}


def separate_annotations(annotations):
    new_annotations = []

    counter = 0

    for annotation in annotations:
        keypoints = np.reshape(np.array(annotation['keypoints']), (6, 5, 3))

        # We don't care about the license plates
        keypoints = keypoints[:4]

        # Looking at each 5x3 matrix for the lights
        for keypoint in keypoints:

            # If the center is visible then add it to
            if keypoint[4][2] != 0:
                new_annotation = deepcopy(annotation)
                new_annotation['keypoints'] = list(keypoint.flatten())
                new_annotations.append(new_annotation)

    return new_annotations


def find_corresponding_image(image_id):
    for img in os.listdir(os.path.join('data-apollocar3d', 'images', 'train')):
        if img.find(image_id) != -1:
            return os.path.join(os.path.join('data-apollocar3d', 'images', 'train'), img)
    for img in os.listdir(os.path.join('data-apollocar3d', 'images', 'val')):
        if img.find(image_id) != -1:
            return os.path.join(os.path.join('data-apollocar3d', 'images', 'val'), img)
    return None


def keypoint_out_of_range(corners, center, visibility, bbox):
    return any(
        [(abs(corners[i][0] - center[0]) > min(64, bbox[0]) or abs(corners[i][1] - center[1]) > min(64, bbox[1])) and
         visibility[i] != 0
         for i in range(len(corners))]) or all([visibility[i] == 0 for i in range(len(visibility))])


def get_max_regression(keypoints, center_x, center_y):
    return max(
        [
            abs(keypoints[0] - center_x) if keypoints[2] != 0 else 0,
            abs(keypoints[1] - center_y) if keypoints[2] != 0 else 0,
            abs(keypoints[3] - center_x) if keypoints[5] != 0 else 0,
            abs(keypoints[4] - center_y) if keypoints[5] != 0 else 0,
            abs(keypoints[6] - center_x) if keypoints[8] != 0 else 0,
            abs(keypoints[7] - center_y) if keypoints[8] != 0 else 0,
            abs(keypoints[9] - center_x) if keypoints[10] != 0 else 0,
            abs(keypoints[10] - center_y) if keypoints[10] != 0 else 0,
        ]
    )


def make_cropped_internal(annotations, index, directory):

    new_annotations = []
    annotation = annotations[index]

    if annotation['image_id'] in bad_images:
        return []

    # Find the image for this annotation
    image_id = annotation['image_id'].split('_')[0]
    matching_image = cv2.imread(find_corresponding_image(image_id=image_id))

    keypoints = annotation['keypoints']
    center_x, center_y = int(keypoints[-3]), int(keypoints[-2])

    # bounding box for the vehicle
    bbox = annotation['bbox']

    # Cropped version of the
    cropped_image = deepcopy(matching_image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])

    translated_keypoint_x, translated_keypoint_y = int(center_x) - bbox[0], int(center_y) - bbox[1]

    # Crop the image around the internal
    top_left_x = max(translated_keypoint_x - 64, 0)
    padding_left = max(64 - translated_keypoint_x, 0)
    top_left_y = max(translated_keypoint_y - 64, 0)
    padding_top = max(64 - translated_keypoint_y, 0)
    bot_right_x = min(translated_keypoint_x + 64, cropped_image.shape[1])
    padding_right = max(translated_keypoint_x + 64 - cropped_image.shape[1], 0)
    bot_right_y = min(translated_keypoint_y + 64, cropped_image.shape[0])
    padding_bottom = max(translated_keypoint_y + 64 - cropped_image.shape[0], 0)

    cropped_internal = deepcopy(cropped_image[int(top_left_y):int(bot_right_y), int(top_left_x): int(bot_right_x)])

    # Padded image with internal centered in image
    padded_image = cv2.copyMakeBorder(cropped_internal, padding_top, padding_bottom, padding_left, padding_right,
                                      cv2.BORDER_CONSTANT, None, [0, 0, 0])

    # Get the reflected version of the image
    reflected_image, reflected_keypoints = reflect_image_and_keypoints(padded_image, keypoints)
    # print(reflected_keypoints)
    # print(keypoints)
    #
    # radius = 2
    #
    # target_1 = (int(64 + reflected_keypoints[0] - center_x), int(64 + reflected_keypoints[1] - center_y))
    # target_2 = (int(64 + reflected_keypoints[3] - center_x), int(64 + reflected_keypoints[4] - center_y))
    # target_3 = (int(64 + reflected_keypoints[6] - center_x), int(64 + reflected_keypoints[7] - center_y))
    # target_4 = (int(64 + reflected_keypoints[9] - center_x), int(64 + reflected_keypoints[10] - center_y))
    #
    # cv2.circle(reflected_image, target_1, radius, (0, 255, 0), -1)
    # cv2.circle(reflected_image, target_2, radius, (255, 0, 0), -1)
    # cv2.circle(reflected_image, target_3, radius, (0, 0, 255), -1)
    # cv2.circle(reflected_image, target_4, radius, (0, 255, 255), -1)
    #
    # cv2.imshow('Reflected image', reflected_image)
    # cv2.imshow('Original image', padded_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    new_annotation = deepcopy(annotation)
    new_annotation['image_id'] = str(annotation['image_id']) + '_rot_'
    new_annotation['keypoints'] = reflected_keypoints
    new_annotations.append(new_annotation)
    cv2.imwrite(os.path.join('cropped-internal-data', directory, str(new_annotation['image_id']) + '.jpg'),
                reflected_image)
    print('Saved image ' + str(index) + ' to ' + os.path.join('cropped-internal-data', directory,
                                                              str(new_annotation['image_id']) + '.jpg'))

    # Save the image to the corresponding directory
    cv2.imwrite(os.path.join('cropped-internal-data', directory, str(annotation['image_id']) + '.jpg'),
                padded_image)
    new_annotations.append(annotation)
    print('Saved image ' + str(index) + ' to ' + os.path.join('cropped-internal-data', directory,
                                                             str(annotation['image_id']) + '.jpg'))

    return new_annotations


def make_new_annotations(annotations):
    new_annotations = []
    idx = 0
    for annotation in annotations:

        keypoints = annotation['keypoints']
        center_x, center_y = int(keypoints[-3]), int(keypoints[-2])

        corners = [[keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]], [keypoints[6], keypoints[7]],
                   [keypoints[9], keypoints[10]]]
        visibilities = [keypoints[2], keypoints[5], keypoints[8], keypoints[11]]
        bbox = annotation['bbox']

        if not keypoint_out_of_range(corners, [center_x, center_y], visibilities, [bbox[2], bbox[3]]) \
                and annotation['image_id'] not in bad_images:
            annotation['image_id'] = str(annotation['image_id']) + '_' + str(idx)
            new_annotations.append(annotation)
            idx += 1

    return new_annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str)
    parser.add_argument("--output-json-file", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    train_file = open(os.path.join('data-apollocar3d', 'annotations', args.json_file))
    annotations = json.load(train_file)

    bad_images = [24344459, 24348519, 24407578, 24412238, 24348519, 24412238]
    new_file = make_new_annotations(annotations)

    max_regression = 0

    new_annotations = []
    for i in range(len(new_file)):
        new_annotations.extend(make_cropped_internal(new_file, i, args.dataset))

    with open(os.path.join('data-apollocar3d', 'annotations', args.output_json_file), 'w') as f:
        json.dump(new_annotations, f)

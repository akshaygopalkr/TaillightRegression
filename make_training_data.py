import json
import os
import numpy as np
from copy import deepcopy
import cv2
import argparse

internals = {'left_front_car_light': [0,1,2,3],  'left_rear_car_light': [22,23,25,26], 'right_rear_car_light': [31,32,34,35],  'right_front_car_light': [54,55,56,57]}

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

def keypoint_out_of_range(corners, center, visibility):

    return any([(abs(corners[i][0] - center[0]) > 64 or abs(corners[i][1] - center[1]) > 64) and visibility[i] != 0 for i in range(len(corners))])

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

    annotation = annotations[index]

    # Find the image for this annotation
    matching_image = cv2.imread(find_corresponding_image(str(annotation['image_id'])))


    keypoints = annotation['keypoints']
    center_x, center_y = int(keypoints[-3]), int(keypoints[-2])

    corners = [[keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]], [keypoints[6], keypoints[7]],
    [keypoints[9], keypoints[10]]]
    visibilities = [keypoints[2], keypoints[5], keypoints[8], keypoints[11]]



    if not keypoint_out_of_range(corners, [center_x, center_y], visibilities):

        # Crop the image around the internal
        top_left_x = max(center_x - 64, 0)
        padding_left = max(64 - center_x, 0)
        top_left_y = max(center_y - 64, 0)
        padding_top = max(64 - center_y, 0)
        bot_right_x = min(center_x + 64, matching_image.shape[1])
        padding_right = max(center_x + 64 - matching_image.shape[1], 0)
        bot_right_y = min(center_y + 64, matching_image.shape[0])
        padding_bottom = max(center_y + 64 - matching_image.shape[0], 0)

        cropped_internal = deepcopy(matching_image[int(top_left_y):int(bot_right_y), int(top_left_x): int(bot_right_x)])

        # Padded image with internal centered in image
        padded_image = cv2.copyMakeBorder(cropped_internal, padding_top, padding_bottom, padding_left, padding_right,
                                          cv2.BORDER_CONSTANT, None, [0, 0, 0])

        # Save the image to the corresponding directory
        cv2.imwrite(os.path.join('cropped-internal-data', directory, str(annotation['image_id']) + '.jpg'), padded_image)
        print('Saved image ' + str(index) + 'to ' +  os.path.join('cropped-internal-data', directory, str(annotation['image_id']) + '.jpg'))

    else:
        print("Image " + str(index) + " had a keypoint out of range: " + str(keypoints) + " " + str([center_x, center_y]))
        print(find_corresponding_image(str(annotation['image_id'])))

    return get_max_regression(keypoints, center_x, center_y)

def make_new_annotations(annotations):

    new_annotations = []
    for annotation in annotations:

        keypoints = annotation['keypoints']
        center_x, center_y = int(keypoints[-3]), int(keypoints[-2])

        corners = [[keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]], [keypoints[6], keypoints[7]],
                   [keypoints[9], keypoints[10]]]
        visibilities = [keypoints[2], keypoints[5], keypoints[8], keypoints[11]]

        if not keypoint_out_of_range(corners, [center_x, center_y], visibilities):
            new_annotations.append(annotation)

    return new_annotations



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    train_file = open(os.path.join('data-apollocar3d', 'annotations', args.json_file))
    annotations = json.load(train_file)

    new_file = make_new_annotations(annotations)


    with open(os.path.join('data-apollocar3d', 'annotations', args.json_file), 'w') as f:
        json.dump(new_file, f)

    max_regression = 0

    # Save each cropped internal in the jsonfile
    for i in range(len(annotations)):
        max_regression = max(make_cropped_internal(annotations, i, args.dataset), max_regression)

    print(max_regression)



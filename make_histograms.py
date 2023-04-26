import matplotlib.pyplot as plt
import json
import numpy
import os



def make_histogram(image_list):

    height_arr = []
    width_arr = []
    area_arr = []
    avg_height = 0
    avg_width = 0
    avg_area = 0
    img_counter = 0


    for image in image_list:

        img_counter += 1

        keypoints = image['keypoints']
        x1, x2, x3, x4 = keypoints[0], keypoints[3], keypoints[6], keypoints[9]
        y1, y2, y3, y4 = keypoints[1], keypoints[4], keypoints[7], keypoints[10]

        x_arr = [x for x in [x1, x2, x3, x4] if x != 0]
        y_arr = [y for y in [y1, y2, y3, y4] if y != 0]

        # Calculate the height of the polygon
        height = max(y_arr) - min(y_arr)

        # Calculate the width of the polygon
        width = max(x_arr) - min(x_arr)

        # Calculate the area of the polygon using the Shoelace Formula
        area = abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)) / 2

        avg_width += width
        avg_height += height
        avg_area += area
        height_arr.append(height)
        width_arr.append(width)
        area_arr.append(area)

        if height > 1000:
            print(keypoints)
        if width > 1000:
            print(keypoints)

    avg_height = avg_height / img_counter
    avg_width = avg_width / img_counter
    avg_area = avg_area / img_counter
    print(avg_height, avg_area, avg_width)

    # Plot a histogram of the data
    plt.hist(height_arr, bins=200)

    # Add a title and axis labels
    plt.title("Histogram of Vehicle Light Height")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()

    # Plot a histogram of the data
    plt.hist(width_arr, bins=200)

    # Add a title and axis labels
    plt.title("Histogram of Vehicle Light Width")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()

    # Plot a histogram of the data
    plt.hist(area_arr, bins=200)

    # Add a title and axis labels
    plt.title("Histogram of Vehicle Light Area")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()

if __name__ == '__main__':

    with open(os.path.join('data-apollocar3d', 'annotations', 'vcrop_keypoints_train.json'), 'r') as f:
        data = json.load(f)

    with open(os.path.join('data-apollocar3d', 'annotations', 'vcrop_keypoints_val.json'), 'r') as f:
        data.extend(json.load(f))

    with open(os.path.join('data-apollocar3d', 'annotations', 'vcrop_keypoints_test.json'), 'r') as f:
        data.extend(json.load(f))

    make_histogram(data)
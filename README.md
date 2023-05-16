# Using Taillight Regression Model

## Making Cropped Image
* Here is some example code on how to properly format the cropped image around the vehicle light so it can be passed
into the model:
```python
# Assume you have variables for the full image, bounding box of vehicle, and (x,y) coordinate of taillight coordinate
# image: full cv2 image of scene
# bbox: [upper_left_x, upper_left_y, bottom_right_x, bottom_right_y]
# light_coordinate: (x,y) coordinate of center of vehicle light

# *** MAKE SURE TO PUT helper_methods.py IN THE SAME DIRECTORY YOU RUN YOUR CODE! ***
from helper_methods.py import make_cropped_image
import torch

cropped_image = make_cropped_image(image, bbox, light_coordinate)
cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

# Convert image to tensor
cropped_image = torch.tensor(cropped_image)
cropped_image = torch.reshape(cropped_image, (3, cropped_image.size()[0], cropped_image.size()[1]))
cropped_image = cropped_image.float()
```

## Loading Model and Generating Output
```python
import torch
from torchvision import models
import os

model = torch.load(os.path.join('/home/cvrr/CenterNet', 'exp', 'regression_model', '(Regression model name).pt')))

# We multiply by 64 since the model outputs number from -1 to 1
output = 64*model(cropped_image)
```
* The output will have 8 numbers, with the first two corresponding to the (x,y) distance from the center to the upper left corner,
the second two correspond to the upper right, the third two correspond to the bottom left, and the last two correspond to the bottom right.


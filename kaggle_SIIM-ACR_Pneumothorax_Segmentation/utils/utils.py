import numpy as np

def rle2mask(rle, img_shape):
    width, height = img_shape[0], img_shape[1]
    mask= np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        # see https://github.com/tensorflow/models/issues/3906#issuecomment-391998102
        # The segmentation ground truth images in your custom dataset should have
        # 1, 2, 3, ..., num_class grayscale value at each pixel (0 for background).
        # For example if you have 2 classes, you should use 1 and 2 for corresponding pixel.
        # Of course the segmentation mask will look almost "black". If you choose,
        # say 96 and 128, for your segmentation mask to make the it looks more human friendly,
        #the network may end up predicting labels greater than num_class,
        # which leads to the error in this issue.
        mask[current_position:current_position+lengths[index]] = 1  # Do NOT use 255
        current_position += lengths[index]

    return mask.reshape(width, height).T


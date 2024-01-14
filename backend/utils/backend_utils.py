import torch
from .unetformerplusplus import UNetFormerPlusPlus_pl
from .unet import UNet
from .unetPlusPlus import UNetPlusPlus
from .unetformer import UNetFormer_pl
from .deeplabv3plus import DeepLabV3Plus_pl
from patchify import patchify, unpatchify
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import collections
import io
import pandas as pd
import os

def load_model(model_name: str):
    model = None
    if model_name == "UNet":
        model = UNet(n_classes=5, encoder_name="resnet34")
    elif model_name == "UNet++":
        model = UNetPlusPlus(n_classes=5,encoder_name="resnet34")
    elif model_name == "DeepLabV3+":
        model = DeepLabV3Plus_pl(n_classes=5,encoder_name="resnet34")
    elif model_name == "UNetFormer":
        model = UNetFormer_pl(n_classes=5, encoder_name="resnet34.a1_in1k")
    else:
        model = UNetFormerPlusPlus_pl(n_classes=5, encoder_name="resnet34.a1_in1k")
    # print(os.getcwd())
    # file_path = os.path.abspath(os.path.join('..', 'models', f'sd_{model_name}.pth'))
    model.load_state_dict(torch.load(f"./models/sd_{model_name}.pth", map_location=torch.device("cpu")))
    return model

def predict(model, file):
    # Returns numpy array os size 512*512
    mean = [0.370, 0.397, 0.343]
    std = [0.164, 0.137, 0.113]
    img = file
    if not isinstance(file, np.ndarray):
        img = Image.open(io.BytesIO(img))


    img = np.asarray(img)

    if img.shape[-1] > 3:
        img = img[:, :, :3]
    

    # resize_transform = A.Resize(512, 512)
    # resized_arr = resize_transform(image=img)
    # resized_img = resized_arr['image']

    final_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    processed_img = final_transform(image=img)
    processed_img = processed_img['image']

    model.eval()
    logits = model(processed_img.unsqueeze(0))
    
    pred = torch.argmax(torch.softmax(logits,dim=1),dim=1)
    return pred.squeeze(0).numpy()

def convert_to_rgb(input_array):
    """
    Convert a NumPy array to an RGB array based on a given mapping.

    Parameters:
    input_array (numpy.ndarray): The input NumPy array.
    id2rgb (dict): A mapping of IDs to RGB values.

    Returns:
    numpy.ndarray: The RGB array.
    """

    id2rgb = {
        0:(0,0,0),
        1:(255,0,0),
        2:(0,255,0),
        3:(0,0,255),
        4:(255,255,0),
    }

    # Create an empty RGB array with the same shape as the input array
    rgb_array = np.empty(input_array.shape + (3,), dtype=np.uint8)

    # Iterate through the id2rgb mapping and assign RGB values to the corresponding locations
    for key, value in id2rgb.items():
        mask = input_array == key
        for i in range(3):
            rgb_array[:, :, i][mask] = value[i]

    return rgb_array

def encode_masks_to_rgb(masks):
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    # Create an empty RGB image
    height, width = masks.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors based on the mask values
    for i in range(len(colors)):
        mask_indices = masks == i
        rgb_image[mask_indices] = colors[i]

    return rgb_image

def doPatchify(file):
    img = Image.open(io.BytesIO(file))
    
    # img = Image.open(file)
    img = np.asarray(img)
    h,w,c = img.shape

    if h > 512 or w > 512:
        return True
    else:
        return False
    
def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
#     print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = A.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                           border_mode=0, value=[0, 0, 0])(image=image)
    img_pad = pad['image']
#     print(img_pad.shape)
    return img_pad, height_pad, width_pad

def predict_sliding_window(model, file):
    window_size = (512, 512)
    stride = (512, 512)
    
    image = Image.open(io.BytesIO(file))
    image = np.asarray(image)

    if image.shape[-1] > 3:
        image = image[:, :, :3]
    
    image, height_pad, width_pad = get_img_padded(image, (512,512))


    image_shape = image.shape
    window_size_x, window_size_y = window_size
    stride_x, stride_y = stride

    # Create an array to store the predictions
    predictions = np.zeros((image_shape[0], image_shape[1]))

    for y in range(0, image_shape[0] - window_size_x + 1, stride_x):
        for x in range(0, image_shape[1] - window_size_y + 1, stride_y):
            window = image[y:y+window_size_x, x:x+window_size_y, :]
            
            # Now you can perform your prediction on the window
            # For example, you can call a function to process the window
            prediction_masks = predict(model, window)

            # Ensure that the prediction_masks have the same number of channels as the original image
            if prediction_masks.shape[-1] != image.shape[-1]:
                prediction_masks = np.expand_dims(prediction_masks, axis=-1)

            # Flatten the spatial dimensions and fill in the corresponding positions in the predictions array
            predictions[y:y+window_size_x, x:x+window_size_y] = prediction_masks.reshape((window_size_x, window_size_y, -1)).mean(axis=-1)

    return encode_masks_to_rgb(predictions)[height_pad:, width_pad:,:]

def patch_img(model, file):
    
    image = Image.open(io.BytesIO(file))
    image = np.asarray(image)

    if image.shape[-1] > 3:
        image = image[:, :, :3]

    image_height, image_width, channel_count = image.shape
    patch_height, patch_width, step = 512, 512, 512
    patch_shape = (patch_height, patch_width, channel_count)
    patches = patchify(image, patch_shape, step=step)

    output_patches = np.empty(patches.shape).astype(np.uint8)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            output_patch = convert_to_rgb(predict(model=model, file=patch))  # process the patch
            output_patches[i, j, 0] = output_patch

    output_height = image_height - (image_height - patch_height) % step
    output_width = image_width - (image_width - patch_width) % step
    output_shape = (output_height, output_width, channel_count)
    output_image = unpatchify(output_patches, output_shape)

    ori_image = unpatchify(patches, output_shape)

    return ori_image, output_image


def count_pixel(pred):
    image = Image.fromarray(pred)

    # Define the colors you want to count in RGB format
    color2label = {
        (0, 0, 0): "Background",
        (255, 0, 0): "Building",
        (0, 255, 0): "Woodland",
        (0, 0, 255): "Water",
        (255, 255, 0): "Road",
    }

    # Create a flattened list of pixel values
    pixels = list(image.getdata())
    # Count the number of pixels for each color
    color_counts = collections.Counter(pixels)
    # Calculate the total number of pixels in the image
    total_pixels = len(pixels)

    # Initialize a dictionary to store the average number of pixels for each class
    average_counts = {color2label[label]: (count / total_pixels)*100 for label, count in color_counts.items()}

    class_counts = {color2label[label]: count for label, count in color_counts.items()}

    pix_avg = {}
    pix_count = {}
    for _, i in color2label.items():
        try:
            pix_avg[i] = average_counts[i]
            pix_count[i] = class_counts[i]
        except:
            pix_avg[i] = 0
            pix_count[i] = 0


    x = {
        "class": list(pix_avg.keys()),
        "percentage": list(pix_avg.values()),
        "pixel_count": list(pix_count.values())
    }
    # print(x)
    
    return pd.DataFrame(x)

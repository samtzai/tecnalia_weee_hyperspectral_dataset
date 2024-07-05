import cv2
import json
import yaml



############################################################################################################
# READ AND WRITE JSON FILES
def read_json(json_path):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def write_json(json_file, info):
    with open(json_file, 'w') as f:
        json.dump(info, f, sort_keys=False, indent=4)
############################################################################################################



############################################################################################################
# LOAD THE PARAMETERS FROM THE PARAMS.YAML FILE
def load_params(params_file = 'params.yaml'):
    with open(params_file, 'r') as fd:
        return yaml.safe_load(fd)
############################################################################################################



############################################################################################################
# READ AND SAVE IMAGES USING OPENCV
def read_image(path):
    """
    Read an image from a file name and store the information in a numpy array (ndarray). It uses OpenCV.

    Parameters
    ----------
    path: str
        File name of the image to open.

    Returns
    -------
    numpy.ndarray
        Numpy array of dimensions [M, N, 3] containing the pixel information of the opened
        image in RGB format.
    """

    srcBGR = cv2.imread(path,  cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)

    return destRGB

def save_image(path, image):
    """
    Save an image from a pixel matrix (ndarray) to disk in JPEG format. It uses OpenCV.

    Parameters
    ----------
    path: str
        File name of the image we want to save.
    image: numpy.ndarray
        Numpy array containing the pixel information of the image as an 'int' matrix.

    Returns
    -------
    bool
        Returns True if image is saved successfully
    """
    if len(image.shape) == 3 and image.shape[-1] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img = image

    return cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
############################################################################################################
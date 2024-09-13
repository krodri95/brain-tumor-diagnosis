import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Resize

import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import felzenszwalb
from matplotlib import pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


#draw bounding box
def tumor_bbox(img, cam):
    img  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) #convert PIL to OpenCV format
    rows, cols = img.shape[:2]
    sy, sx = rows/cam.shape[0], cols/cam.shape[1]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    # Define range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    boxes = []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding box around each contour
    areaMax, idMax = 0, 0
    for idx, contour in enumerate(contours):
      # Get the bounding box coordinates
      x, y, w, h = cv2.boundingRect(contour)
      area = cv2.contourArea(contour)
      if area > areaMax:
          areaMax = area
          idMax = idx
      boxes.append((x+int(w/2), y+int(h/2), w, h))
      # Draw the bounding box rectangle
      cv2.rectangle(img, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), (0, 255, 0), 2)

    # Create an empty mask (same size as the original mask)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)

    # Draw the contour on the empty mask
    cv2.drawContours(contour_mask, [contours[idMax]], -1, (255), thickness=cv2.FILLED)

    # Apply the contour mask to the original binary mask
    maskMax = cv2.bitwise_and(mask, contour_mask)
    maskMax = cv2.resize(maskMax, (img.shape[:2][::-1]))
    return img, boxes, maskMax


class GradCam():
    # Class Activation Mapping
    def __init__(self, model, target_layers, net_size, cam='ScoreCAM'):
        super().__init__()
        self.net_size = net_size
        self.img_size = None
        self.model = model
        self.m = nn.Softmax(dim=1)
        self.label_map = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

        # Construct the CAM object once, and then re-use it on many images:
        #GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
        cam_classes = {
            'GradCAM': GradCAM,
            'HiResCAM': HiResCAM,
            'ScoreCAM': ScoreCAM,
            'GradCAMPlusPlus': GradCAMPlusPlus,
            'AblationCAM': AblationCAM,
            'XGradCAM': XGradCAM,
            'EigenCAM': EigenCAM,
            'FullGrad': FullGrad,
            'LayerCAM': LayerCAM
        }

        if cam in cam_classes:
            self.cam = cam_classes[cam](model=model.float(), target_layers=target_layers)
        else:
            raise ValueError(f'{cam} not supported!')

        self.transform = transforms.Compose([transforms.Grayscale(), Resize(self.net_size[1:], Image.BILINEAR), transforms.ToTensor()])

    def getMap(self, image_path):
        img = Image.open(image_path)
        self.img_size = img.size
        input_tensor = self.transform(img).unsqueeze(0)

        out = self.m(self.model(input_tensor))
        _, pred = torch.max(out, 1)
        print(f'Prediction : {self.label_map[int(pred)]}')

        targets = [ClassifierOutputTarget(int(pred))]
        im = np.zeros([self.net_size[1],self.net_size[2],3])
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.asarray(im)/255, grayscale_cam, use_rgb=False)

        # You can also get the model outputs without having to re-inference
        #print('Predictions: ', self.cam.outputs)

        '''
        detect, boxes, mask = tumor_bbox(img, visualization)

        # Add the text to the image
        cv2.putText(detect, '{}:{}'.format(self.label_map[int(pred.cpu())],round(float(out[0,pred]),3)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        tbox = None
        ar = 0
        #filter the bboxes
        for bbox in boxes:
            if bbox[2]*bbox[3] > ar:
                #print(bbox)
                tbox = bbox
                ar = bbox[2]*bbox[3]

        return visualization, detect, tbox, mask
        '''
        return visualization
    
    def inflateTbox(self, tbox, infl):
        # Adjust the coordinates to be within bounds
        xmin = int(max(0, min((tbox[0]-tbox[2]/2*infl), self.net_size[2] - 1))*self.img_size[0]/self.net_size[2])
        ymin = int(max(0, min((tbox[1]-tbox[3]/2*infl), self.net_size[1] - 1))*self.img_size[1]/self.net_size[1])
        xmax = int(max(0, min((tbox[0]+tbox[2]/2*infl), self.net_size[2] - 1))*self.img_size[0]/self.net_size[2])
        ymax = int(max(0, min((tbox[1]+tbox[3]/2*infl), self.net_size[1] - 1))*self.img_size[1]/self.net_size[1])

        # print(img_size)
        # print(tbox)
        # print(xmin, ymin, xmax, ymax)

        return (xmin, ymin, xmax, ymax)
    

def graphSegment(image_path, mask):
    img = Image.open(image_path)
    image = np.array(img)
    segmented_image = felzenszwalb(image, scale=400, sigma=0.8, min_size=20).astype(np.uint8)

    h, w = segmented_image.shape
    # Calculate the histogram of pixel values
    segImp = cv2.bitwise_and(mask, segmented_image)
    hist = cv2.calcHist([segImp], [0], None, [256], [0, 256])

    # Convert the histogram to a list of frequencies
    frequencies = hist.flatten().tolist()
    del frequencies[0] # first element corresponds to the background

    label = frequencies.index(max(frequencies))+1 #pixel 0 was deleted so need to increment to get the true index

    tmask = np.zeros([img.size[1],img.size[0]])
    tmask[segmented_image == label] = 255

    render  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    render[tmask == 255] = (0,0,255)

    #colormap for the segmentated classes
    # Generate a color map using matplotlib
    cmap = plt.get_cmap('tab20c')
    colors = cmap(np.linspace(0, 1, 20))[:, :3] * 255  # Scale to [0, 255]
    colored_lmask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index in range(len(frequencies)):
        colored_lmask[segmented_image == class_index] = colors[class_index%20].astype(np.uint8)
    

    return render, tmask, colored_lmask


def region_growing(image_path, tbox):
    img = cv2.imread(image_path, 0) #grayscale
    h, w = img.shape[:2]
    tmask = np.zeros((h, w), dtype=np.uint8)

    points = [((tbox[0]+tbox[2])//2, (tbox[1]+tbox[3])//2)]
    threshold = 50 #tolerance in pixels

    while points:
        x, y = points.pop()
        seed_value = img[y, x]

        # visit the point
        tmask[y, x] = 255

        # Check neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if tmask[ny, nx] == 0 and abs(img[ny, nx] - seed_value) < threshold:
                        points.append((nx, ny))

    render  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    render[tmask == 255] = (0,0,255)

    return render, tmask, render
    
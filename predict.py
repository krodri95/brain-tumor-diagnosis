import torch
import numpy as np
import cv2
import argparse
from pathlib import Path

from utils import *
from models.model import Model

net_size = (1, 128, 128)
num_classes = 4
model_attr = {'base': {'weights': './models/best_model_base.pth', 'att':None},\
              'se': {'weights': './models/best_model_se.pth', 'att':'SE'},\
              'cbam': {'weights': './models/best_model_cbam.pth', 'att':'CBAM'}}


def main(imgp='./assets/', model_type='cbam', cam_type=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Model(nc=num_classes, att=model_attr[model_type]['att']).to('cpu')  # create
    model.load_state_dict(torch.load(model_attr[model_type]['weights'], map_location=device))
    model.eval()

    #target layers for class activation mapping
    target_layers = [model.fpn.convf]

    #Dry run to inspect the model
    x = torch.randn(net_size).unsqueeze(0).to('cpu') #1xcxhxw
    out = model(x)

    cam = GradCam(model, target_layers, net_size, cam_type)
    #glioma, meningioma, pituitary
    heatmap = cam.getMap(imgp)
    rmask, _, _ = graphSegment(imgp, heatmap)

    img = cv2.imread(imgp, cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (img.shape[:2][::-1]))

    heatmap = np.float32(heatmap) / 255
    img = np.float32(img) / 255

    expl = (1 - 0.5) * heatmap + 0.5 * img
    expl = expl / np.max(expl)

    d = Path('./exp')
    d.mkdir(parents=True, exist_ok=True)

    cv2.imwrite('./exp/'+model_type+'_'+cam_type+'_'+imgp.split('/')[-1], np.uint8(255 * expl))
    cv2.imwrite('./exp/'+model_type+'_'+cam_type+'_mask_'+imgp.split('/')[-1], rmask)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgp', type=str, default='', help='input image path')
    parser.add_argument('--model-type', default='cbam', help='base, se or cbam')
    parser.add_argument('--cam-type', default='ScoreCAM', help='GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad or LayerCAM')
 
    opt = parser.parse_args()
    assert Path(opt.imgp).is_file(), f'Invalid file: {opt.imgp}'
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
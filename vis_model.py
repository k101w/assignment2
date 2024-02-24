import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import argparse
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from utils import *
import pdb
from torchvision import transforms

from util_vis import visualize_cam, Normalize
from gradcam import GradCAM
from eval_model import preprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

class vis_model:
    def __init__(self,model):
        self.resnet_model_dict = dict(type='resnet', arch=model.encoder, layer_name='layer4', input_size=(137, 137))
       
        self.resnet_gradcam = GradCAM(self.resnet_model_dict, True)
        self.model = model
        self.model_encoder = model.encoder
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    def vis(self,img):
        img = img.permute(0,3,1,2)
        images_normalize = self.normalize(img)
        mask, _ = self.resnet_gradcam(images_normalize)
        #output,_ = self.model_encoder(images_normalize)
        heatmap, result = visualize_cam(mask, img)
        return heatmap, result

    def forward(self,args):
        r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)
        loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
        eval_loader = iter(loader)
        self.model.eval()
        start_iter = 0
        if args.load_checkpoint:
            checkpoint = torch.load(f'checkpoint_mesh_8230_0.0001.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Succesfully loaded iter {start_iter}")
        max_iter = len(eval_loader)
        for step in range(start_iter, max_iter):
            feed_dict = next(eval_loader)
            images_gt, _ = preprocess(feed_dict, args)
            _,output = self.vis(images_gt)
            if (step % args.vis_freq) == 0:
                save_image(output, f'output4/{args.type}_{step}.jpg')
                

if __name__=='__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    model = SingleViewto3D(args)
    model.to(args.device) 
    vis_tool = vis_model(model)
    vis_tool.forward(args)





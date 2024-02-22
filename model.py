from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d


        
class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        
        #64 2 2 2
        # 256 2 2 2
        #32 4 4 4
        #16 8 8 8
        #8 16 16 16
        #4 32 32 32
        # 1 32 32 32

        # define decoder
        if args.type == "vox":
            self.fc = torch.nn.Linear(512,2048)

            self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
            self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
            self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
            self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        
            self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1),
        )

            self.decoder = torch.nn.Sequential(self.layer1,self.layer2,self.layer3,self.layer4,self.layer5)
    
            # self.layer1 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            #     torch.nn.BatchNorm3d(32),
            #     torch.nn.ReLU()
            # )
            # self.layer2 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(in_channels=32,out_channels=16,kernel_size=2,stride=2),
            #     torch.nn.BatchNorm3d(16),
            #     torch.nn.ReLU()
            # )
            # self.layer3 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(in_channels=16,out_channels=8,kernel_size=2,stride=2),
            #     torch.nn.BatchNorm3d(8),
            #     torch.nn.ReLU()
            # )
            # self.layer4 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(in_channels=8,out_channels=4,kernel_size=2,stride=2),
            #     torch.nn.BatchNorm3d(4),
            #     torch.nn.ReLU()
            # )
            # self.layer5 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(in_channels=4,out_channels=2,kernel_size=1,stride=1),
            #     torch.nn.BatchNorm3d(2),
            #     torch.nn.ReLU()
            # )
            # self.layer6 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(in_channels=2,out_channels=1,kernel_size=1,stride=1),
            #     torch.nn.BatchNorm3d(1),
            #     torch.nn.Sigmoid()
            # )
            #self.decoder = torch.nn.Sequential(self.layer1,self.layer2,self.layer3,self.layer4,self.layer5,self.layer6)
            # Input: b x 512
            # Output: b x 32 x 32 x 32             
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points

            # TODO:
            self.decoder = nn.Sequential(          
            torch.nn.Linear(512, self.n_point),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_point, self.n_point*3),
            torch.nn.Tanh()
            )           
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.vertices_num = mesh_pred.verts_packed().shape[0]
            self.decoder = nn.Sequential(          
            torch.nn.Linear(512, self.vertices_num*3),
            torch.nn.Tanh()
            )                   

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            encoded_feat = self.fc(encoded_feat)
            encoded_feat = encoded_feat.reshape(B,256,2,2,2)
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred =  self.decoder(encoded_feat) 
            pointclouds_pred = pointclouds_pred.reshape(B,self.n_point,3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred =  self.decoder(encoded_feat)        
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))

            return  mesh_pred          


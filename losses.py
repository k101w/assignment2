import torch
import pdb
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	
	loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(voxel_src),voxel_tgt)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	x_nn = knn_points(point_cloud_src, point_cloud_tgt, K=1,norm=2)
	cham_x = x_nn.dists[..., 0] 
	y_nn = knn_points(point_cloud_tgt, point_cloud_src, K=1,norm=2)
	cham_y = y_nn.dists[..., 0] 
	loss_chamfer = torch.mean(cham_x)+torch.mean(cham_y)
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss

	return loss_laplacian
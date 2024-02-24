import torch
import numpy as np
import imageio
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    look_at_view_transform,
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    TexturesVertex,
)
from pytorch3d.io import load_obj
from pytorch3d.structures import Volumes,Meshes,Pointclouds
import pdb
from pytorch3d.ops import cubify



def render_points(points_src,points_tgt,path,step=None,rend_gt=True):
    num_views = 12
    rgb = torch.ones_like(points_src,device=points_src.device)*torch.tensor([0.7,0.7,1],device=points_src.device) 
    pointcloud_src = Pointclouds(
        points=points_src,features=rgb
    )

    renderer = get_points_renderer(image_size=512,device=points_src.device, radius=0.005)
    R, T = look_at_view_transform(dist=3, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=points_src.device)

    images_src = renderer(pointcloud_src.extend(num_views), cameras=many_cameras)
    images_src = [(image_src.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_src in images_src[...,:3]]
    if step==None:
        imageio.mimsave(path+f'points_src.gif', images_src,fps=5,loop=0)
    else:
        imageio.mimsave(path+f'points_src_{step}.gif', images_src,fps=5,loop=0)
    if rend_gt:
        pointcloud_tgt = Pointclouds(
            points=points_tgt,features=rgb
        )
        images_tgt = renderer(pointcloud_tgt.extend(num_views), cameras=many_cameras)
        images_tgt = [(image_tgt.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_tgt in images_tgt[...,:3]]
        imageio.mimsave(path+f'points_tgt.gif', images_tgt, fps=5,loop=0)
    else:
        mesh_gt = points_tgt
        rend_gt_mesh(mesh_gt,path,step,'point')



def render_voxel(voxels_src,voxels_tgt,path,step=None,rend_gt=True):
    num_views = 12
    render_size = 512
    if voxels_src.shape[0]==voxels_src.shape[1]:
        voxels_src = voxels_src.squeeze(0)
    src = cubify(voxels_src, 0.01)
    src_verts = src.verts_list()[0]
    src_faces = src.faces_list()[0]
    textures = TexturesVertex(src_verts.unsqueeze(0))
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces], textures = textures)
    R, T = look_at_view_transform(dist=6, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))

    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=src_mesh.device)
    renderer = get_mesh_renderer(device=src_mesh.device)

    images_src = renderer(src_mesh.extend(num_views), cameras=many_cameras)
    images_src = [(image_src.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_src in images_src[...,:3]]
    if step==None:
        imageio.mimsave(path+f'voxels_src.gif', images_src,fps=5,loop=0)
    else:
        imageio.mimsave(path+f'voxels_src_{step}.gif', images_src,fps=5,loop=0)
    if rend_gt:
        tgt = cubify(voxels_tgt, 0.01)
        tgt_verts = tgt.verts_list()[0]
        tgt_faces = tgt.faces_list()[0]
        textures = TexturesVertex(tgt_verts.unsqueeze(0))
        tgt_mesh = Meshes(verts=[tgt_verts], faces=[tgt_faces], textures = textures)
        images_tgt = renderer(tgt_mesh.extend(num_views), cameras=many_cameras)
        images_tgt = [(image_tgt.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_tgt in images_tgt[...,:3]]
        imageio.mimsave(path+f'voxels_tgt.gif', images_tgt, fps=5,loop=0)
    else:
        mesh_gt = voxels_tgt
        rend_gt_mesh(mesh_gt,path,step,'vox')

#     volume_extent_world = 3.0
#     raysampler = NDCMultinomialRaysampler(
#     image_width=render_size,
#     image_height=render_size,
#     n_pts_per_ray=150,
#     min_depth=0.1,
#     max_depth=volume_extent_world,
# )
#     raymarcher = EmissionAbsorptionRaymarcher()
#     R, T = look_at_view_transform(dist=12, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
#     many_cameras = FoVPerspectiveCameras(R=R, T=T, device=voxels_src.device)
#     renderer = VolumeRenderer(
#     raysampler=raysampler, raymarcher=raymarcher,
# )
#     volume_src = Volumes(densities = voxels_src.expand(num_views,*voxels_src.shape))
#     volume_tgt = Volumes(densities = voxels_tgt.expand(num_views,*voxels_src.shape))
#     images_src = renderer(volumes=volume_src, cameras=many_cameras)
#     images_src = [(image_src.detach().cpu().squeeze(-1).numpy()*255).astype(np.uint8) for image_src in images_src[0]]
#     imageio.mimsave(path+f'voxels_src.gif', images_src,fps=10,loop=0)
#     images_tgt = renderer(volumes=volume_tgt,cameras=many_cameras)
#     images_tgt = [(image_tgt.cpu().detach().squeeze(-1).numpy()*255).astype(np.uint8) for image_tgt in images_tgt[0]]
#     imageio.mimsave(path+f'voxels_tgt.gif', images_tgt, fps=10,loop=0)
def rend_gt_mesh(mesh_gt,path,step,type):
    num_views = 6
    src_verts = mesh_gt.verts_list()[0]
    src_faces = mesh_gt.faces_list()[0]
    textures = TexturesVertex(src_verts.unsqueeze(0))
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces], textures = textures)

    R, T = look_at_view_transform(dist=3, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=mesh_gt.device)
    renderer = get_mesh_renderer(device=mesh_gt.device)
    images_src = renderer(src_mesh.extend(num_views), cameras=many_cameras)
    # images_src = [(image_src.detach().cpu().numpy()*255).astype(np.uint8) for image_src in images_src[...,:3]]
    images_src = [(image_src.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_src in images_src[...,:3]]
    imageio.mimsave(path+f'mesh_gt_{type}_{step}.gif', images_src,fps=5,loop=0)


def render_mesh(mesh_src,mesh_tgt,path,step=None,rend_gt=True):
    num_views = 6
    src_verts = mesh_src.verts_list()[0]
    src_faces = mesh_src.faces_list()[0]
    textures = TexturesVertex(src_verts.unsqueeze(0))
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces], textures = textures)

    R, T = look_at_view_transform(dist=3, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=mesh_src.device)
    renderer = get_mesh_renderer(device=mesh_src.device)

    images_src = renderer(src_mesh.extend(num_views), cameras=many_cameras)
    images_src = [(image_src.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_src in images_src[...,:3]]
    if step==None:
        imageio.mimsave(path+f'mesh_src.gif', images_src,fps=5,loop=0)
    else:
        imageio.mimsave(path+f'mesh_src_{step}.gif', images_src,fps=5,loop=0)
    if rend_gt:
        tgt_verts = mesh_tgt.verts_list()[0]
        tgt_faces = mesh_tgt.faces_list()[0]
        textures = TexturesVertex(tgt_verts.unsqueeze(0))
        tgt_mesh = Meshes(verts=[tgt_verts], faces=[tgt_faces], textures = textures)
        images_tgt = renderer(tgt_mesh.extend(num_views), cameras=many_cameras)
        images_tgt = [(image_tgt.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8) for image_tgt in images_tgt[...,:3]]
        imageio.mimsave(path+f'mesh_tgt.gif', images_tgt, fps=5,loop=0)
    else:
        mesh_gt = mesh_tgt
        rend_gt_mesh(mesh_gt,path,step,'mesh')      


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb



import trimesh
from util.util_img import depth_to_mesh_df, resize
from skimage import measure
import numpy as np

import torch
def render_model(mesh, sgrid):
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))

    grid_hits = sgrid[index_ray]
    dist = np.linalg.norm(grid_hits - loc, axis=-1)
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    im = dist_im
    return im


def make_sgrid(b, alpha, beta, gamma):
    res = b * 2
    pi = np.pi
    phi = np.linspace(0, 180, res * 2 + 1)[1::2]
    theta = np.linspace(0, 360, res + 1)[:-1]
    grid = np.zeros([res, res, 3])
    for idp, p in enumerate(phi):
        for idt, t in enumerate(theta):
            grid[idp, idt, 2] = np.cos((p * pi / 180))
            proj = np.sin((p * pi / 180))
            grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
            grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
    grid = np.reshape(grid, (res * res, 3))
    return grid

def render_spherical(data, res=128, obj_path=None, debug=False):

  depth_im_b = data['depth'].cpu().numpy()
  silhout_im_b = data['silhou'].cpu().numpy()
  th_b = data['depth_minmax'].cpu().numpy()
  cam_loc_b = data['trans_mat'].cpu().numpy()
  target_size = 480
  im_depth_b = np.zeros((depth_im_b.shape[0],depth_im_b.shape[1],res,res))
  proj_voxel_b = np.zeros((depth_im_b.shape[0],depth_im_b.shape[1],res,res,res))
  for i in range(0,depth_im_b.shape[0]):
    depth_im = depth_im_b[i,:,:,:].squeeze()
    silhou_im = silhout_im_b[i,:,:,:].squeeze()
    th = th_b[i]
    depth_im = resize(depth_im, target_size, 'vertical')
    silhou_im = resize(silhou_im, target_size, 'vertical')
    gt_sil = np.where(silhou_im > 0.95, 1, 0)
    depth_im = depth_im * gt_sil

    bmin = np.amin(depth_im[gt_sil > 0.5])
    bmax = np.amax(depth_im[gt_sil > 0.5])
    scaled_depth = np.interp(depth_im,(bmin, bmax), (0,1))
    #import pdb;pdb.set_trace()
    depth_im = np.where(gt_sil, scaled_depth, depth_im)
    depth_im = depth_im[:, :, np.newaxis]
    b = 64
    tdf = depth_to_mesh_df(depth_im, th, False, 0.6, cam_loc=cam_loc_b[i], res=res, enlarge=1)
    try:
      verts, faces, normals, values = measure.marching_cubes_lewiner(
        tdf, 0.99999 / res, spacing=(1 / res, 1 / res, 1 / res))
      mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
      proj_voxel = trimesh.voxel.local_voxelize(mesh, (0., 0., 0.), pitch=0.25/res, radius=64)[0] #0.25 cm divided in 129 voxels (because enlarge is 1)
      #import pdb;pdb.set_trace()
      proj_voxel = proj_voxel[:-1,:-1,:-1] #TODO: remove this hack, forcing 129 to 128 voxels
      sgrid = make_sgrid(b, 0, 0, 0)
      im_depth = render_model(mesh, sgrid)
      im_depth = im_depth.reshape(2 * b, 2 * b)
      im_depth = np.where(im_depth > 1, 1, im_depth)
      '''
      mesh.rezero()
      im_depth_centered = render_model(mesh, sgrid)
      im_depth_centered = im_depth_centered.reshape(2 * b, 2 * b)
      im_depth_centered = np.where(im_depth_centered > 1, 1, im_depth)
      '''
    except Exception as e:
      #print(mesh.is_empty)
      #print(th)
      print('\nexception!:')
      print(e)
      #print(proj_voxel.shape)
      #import pdb;pdb.set_trace()
      #tdf = depth_to_mesh_df(depth_im, th, False, 1.0, cam_loc=cam_loc_b[i], res=res, debug=True)
      #mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
      #proj_voxel = trimesh.voxel.local_voxelize(mesh, (0., 0., 0.), pitch=1./res, radius=64)[0]
      im_depth = np.ones([2 * b, 2 * b])
      proj_voxel = np.zeros([res,res,res])
    im_depth_b[i,:,:,:] = im_depth
    proj_voxel_b[i,:,:,:,:] = proj_voxel
    #im_depth_centered = np.ones([2 * b, 2 * b])

  return torch.from_numpy(im_depth_b).float().cuda(),torch.from_numpy(proj_voxel_b.astype(float)).float().cuda() #, im_depth_centered


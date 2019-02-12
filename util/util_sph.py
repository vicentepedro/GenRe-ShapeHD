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
  for i in range(0,depth_im_b.shape[0]):
    depth_im = depth_im_b[i,:,:,:].squeeze()
    silhou_im = silhout_im_b[i,:,:,:].squeeze()
    th = th_b[i]
    depth_im = resize(depth_im, target_size, 'vertical')
    silhou_im = resize(silhou_im, target_size, 'vertical')
    gt_sil = np.where(silhou_im > 0.95, 1, 0)
    depth_im = depth_im * gt_sil
    depth_im = depth_im[:, :, np.newaxis]
    b = 64
    tdf = depth_to_mesh_df(depth_im, th, False, 1.0, cam_loc=cam_loc_b[i], res=res)
    try:
      verts, faces, normals, values = measure.marching_cubes_lewiner(
        tdf, 0.99999 / res, spacing=(1 / res, 1 / res, 1 / res))
      mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
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
    except:
      im_depth = np.ones([2 * b, 2 * b])
    im_depth_b[i,:,:,:] = im_depth
    #im_depth_centered = np.ones([2 * b, 2 * b])

  return torch.from_numpy(im_depth_b).float().cuda() #, im_depth_centered


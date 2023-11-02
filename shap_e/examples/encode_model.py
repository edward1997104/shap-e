
import torch

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import decode_latent_mesh
import glob
import os
import trimesh
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rotate_around_axis(mesh, axis = 'x', reverse = False):
    if reverse:
        angle = math.pi / 2
    else:
        angle = -math.pi / 2

    if axis == 'x':
        direction = [1, 0, 0]
    elif axis == 'y':
        direction = [0, 1, 0]
    else:
        direction = [0, 0, 1]

    center = mesh.centroid

    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)

    mesh.apply_transform(rot_matrix)

    return mesh

if __name__ == '__main__':
    xm = load_model('transmitter', device=device)

    example_dir = ''
    output_dir = ''
    model_paths = glob.glob(f'{example_dir}/*/*.obj')
    for model_path in model_paths:
        file_id = os.path.basename(model_path.split('/')[-2])
        # This may take a few minutes, since it requires rendering the model twice
        # in two different modes.
        batch = load_or_create_multimodal_batch(
            device,
            model_path=model_path,
            mv_light_mode="basic",
            mv_image_size=256,
            cache_dir=f"{output_dir}/cached/{file_id}/",
            verbose=True,  # this will show Blender output during renders
        )

        with torch.no_grad():
            latent = xm.encoder.encode_to_bottleneck(batch)

            render_mode = 'stf'  # you can change this to 'nerf'
            size = 128  # recommended that you lower resolution when using nerf

            # cameras = create_pan_cameras(size, device)
            # images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            mesh = decode_latent_mesh(xm=xm, latent=latent)
            mesh = mesh.tri_mesh()
            mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
            mesh = rotate_around_axis(mesh, axis='x', reverse=False)
            mesh.export(f'{output_dir}/reconstructed/{file_id}.obj')


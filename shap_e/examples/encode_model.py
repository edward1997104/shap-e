
import torch

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import decode_latent_mesh
import glob
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            with open(
                f'{output_dir}/reconstructed/{file_id}.obj', 'w'
            ) as f:
                mesh.write_obj(f)



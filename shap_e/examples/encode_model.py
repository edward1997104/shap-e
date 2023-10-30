
import torch

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import decode_latent_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    xm = load_model('transmitter', device=device)
    model_path = "example_data/cactus/object.obj"

    # This may take a few minutes, since it requires rendering the model twice
    # in two different modes.
    batch = load_or_create_multimodal_batch(
        device,
        model_path=model_path,
        mv_light_mode="basic",
        mv_image_size=256,
        cache_dir="example_data/cactus/cached",
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
        mesh.save('example_data/cactus/cactus_reconstructed.obj')



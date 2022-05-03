import numpy as np
import torch
from tqdm import tqdm
from fire import Fire
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork


def run_diffusion_prior(text_embeds: str, out_path: str,
                        batch_size: int = 4096):
    device = "cuda:0"
    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=6,
        dim_head=64,
        heads=8
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=None,
        image_embed_dim=768,
        timesteps=100,
        cond_drop_prob=0.2,
        loss_type="l2",
        condition_on_text_encodings=False
    ).to(device)

    diffusion_prior.load_state_dict(torch.load("diffusion-prior-2.pth"), strict=True)
    diffusion_prior.eval()

    embs = np.load(text_embeds)
    data = (torch.tensor(embs[i:i+batch_size]).to(device) for i in
            tqdm(range(0, len(embs), batch_size)))

    res_emb = (diffusion_prior.p_sample_loop((text_embed.shape[0], 768),
                                              text_cond=dict(text_embed=text_embed))
               for text_embed in data)

    normed = (emb / emb.norm(dim=1, keepdim=True) for emb in res_emb)

    res = [emb.cpu().numpy() for emb in normed]

    np.save(out_path, np.concatenate(res, axis=0))


if __name__ == "__main__":
    Fire(run_diffusion_prior)

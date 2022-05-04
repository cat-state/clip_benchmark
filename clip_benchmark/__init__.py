from pathlib import Path

import pandas
import numpy as np
from tqdm import tqdm
from fire import Fire
from faiss import IndexFlatIP
from webdataset import WebDataset, WebLoader
from sentence_transformers import SentenceTransformer


def get_sentence_embs(path: Path, dataset: Path):
    if Path(path).exists():
        return np.load(path)
    else:
        ds = WebDataset(dataset).to_tuple("txt")
        loader = WebLoader(ds, batch_size=2048, collate_fn=list)
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        sentence_embs = np.concatenate([model.encode([b[0] for b in batch]) for batch in tqdm(loader)], axis=0)
        np.save(path, sentence_embs)
        return sentence_embs


def knn(queries, keys, k=5):
    index = IndexFlatIP(keys.shape[1])
    index.add(keys)
    _, idxs = index.search(queries, k=k)
    return idxs

def cross_modal_retrieval(
    img_embeds: np.array,
    text_embeds: np.array,
    k=5
):

    nearest_imgs_to_txt = knn(text_embeds, keys=img_embeds, k=k)
    nearest_txts_to_img = knn(img_embeds, keys=text_embeds, k=k)

    return {
        "text->img": (nearest_imgs_to_txt == np.arange(len(text_embeds))[:, None]),
        "img->text": (nearest_txts_to_img == np.arange(len(img_embeds))[:, None]),
        "text->img-idxs": nearest_imgs_to_txt[:, 0],
        "img->text-idxs": nearest_txts_to_img[:, 0]
    }

def clip_benchmark(
    img_embeds_file: Path,
    text_embeds_file: Path,
    sentence_embs: Path,
    n=30000,
    k=5,
    dataset="/home/a/mscoco-1st-cap/{00000..00011}.tar",
):


    sentence_embs = get_sentence_embs(sentence_embs, dataset)

    img_embeds = np.load(img_embeds_file)[:n].astype(np.float32)
    gt = np.arange(len(img_embeds))[:, None]
    text_embeds = np.load(text_embeds_file)[:n].astype(np.float32)

    n_range = np.linspace(5000, n, num=10, dtype=np.int32).tolist()

    results = (cross_modal_retrieval(img_embeds[:first_n],
                                     text_embeds[:first_n],
                                     k=k) for first_n in n_range)

    summary = ({k:
                { **r, "text->img": r["text->img"].any(axis=1),
                  "img->text": r["img->text"].any(axis=1)},
                1:
                {**r, "text->img": r["text->img"][:, 0],
                 "img->text": r["img->text"][:, 0]}}
               for r in results)

    def dot(x, y):
        return (x * y).sum(axis=-1)

    similarity = ({**r,
                   "similarity": {
                       "text->img": dot(sentence_embs[:first_n][r[1]["text->img-idxs"]], sentence_embs[:first_n]),
                       "img->text": dot(sentence_embs[:first_n][r[1]["img->text-idxs"]], sentence_embs[:first_n])
                   }}
        for first_n, r in zip(n_range, summary))

    def report(results):
        for first_n, s in zip(n_range, results):
            print("n:", first_n)
            for top_k in [1, k, "similarity"]:
                for metric in ["text->img","img->text"]:
                    print(f"{metric} @ {top_k} {s[top_k][metric].mean():.3f}",

                          f"{f'{s[top_k][metric].std():.3f}' if top_k == 'similarity' else ''}")

                    yield {"first_n": first_n,
                           "metric": metric,
                           "top_k": top_k,
                           "measuring": "similarity" if top_k == "similarity" else "knn",
                           "@": top_k,
                           "value": s[top_k][metric].mean(),
                           "value-std": s[top_k][metric].std()}

    return pandas.DataFrame.from_records([*report(similarity)])


def plot_models():
    import pandas
    import seaborn

    import matplotlib.pyplot as plt
    models = {
        "openclip ViT-B-16": dict(img_embeds="coco-embeds-open-clip-vit-b-16/img_emb/img_emb_0.npy",
                                  text_embeds="coco-embeds-open-clip-vit-b-16/text_emb/text_emb_0.npy"),

        # "openclip ViT-B-16 (336)": dict(img_embeds="coco-embeds-open_clip-vit-b-16-336/img_emb/img_emb_0.npy",
        #                           text_embeds="coco-embeds-open_clip-vit-b-16-336/text_emb/text_emb_0.npy"),

        "openclip ViT-B-32": dict(img_embeds="coco-embeds-open-clip-vit-b-32/img_emb/img_emb_0.npy",
                                  text_embeds="coco-embeds-open-clip-vit-b-32/text_emb/text_emb_0.npy"),

        # "openclip ViT-B-32-e32": dict(img_embeds="coco-embeds-open_clip-vit-b-32-e32/img_emb/img_emb_0.npy",
        #                          text_embeds="coco-embeds-open_clip-vit-b-32-e32/text_emb/text_emb_0.npy"),

        # "openclip ViT-B-32-e31": dict(img_embeds="coco-embeds-open_clip-vit-b-32-e31/img_emb/img_emb_0.npy",
        #                          text_embeds="coco-embeds-open_clip-vit-b-32-e31/text_emb/text_emb_0.npy"),

        "openai ViT-B/16":  dict(img_embeds="coco-embeds-openai-vit-b-16/img_emb/img_emb_0.npy",
                                  text_embeds="coco-embeds-openai-vit-b-16/text_emb/text_emb_0.npy"),

        "openai ViT-B/32":  dict(img_embeds="coco-embeds-openai-vit-b-32/img_emb/img_emb_0.npy",
                                  text_embeds="coco-embeds-openai-vit-b-32/text_emb/text_emb_0.npy"),

        "openai ViT-L/14":  dict(img_embeds="coco-embeds-openai-vit-l-14/img_emb/img_emb_0.npy",
                                  text_embeds="coco-embeds-openai-vit-l-14/text_emb/text_emb_0.npy"),

        "openai ViT-L/14 (336)":  dict(img_embeds="coco-embeds-openai-vit-l-14-336/img_emb/img_emb_0.npy",
                                  text_embeds="coco-embeds-openai-vit-l-14-336/text_emb/text_emb_0.npy"),

        "cloob ViT-B/16":  dict(img_embeds="cloob-vit-b-16/cloob_laion_400m_vit_b_16_32_epochs_coco_train2017_image_embeds.npy",
                                  text_embeds="cloob-vit-b-16/cloob_laion_400m_vit_b_16_32_epochs_coco_train2017_text_embeds.npy"),
        "cloob ViT-L/14":  dict(img_embeds="cloob-vit-l-14-168_image_embeds.npy",
                                text_embeds="cloob-vit-l-14-168_text_embeds.npy")
    }

    results = pandas.DataFrame()
    for name, embs in models.items():
        print(name)
        print("-" * 50)
        res = clip_benchmark(img_embeds_file=embs["img_embeds"],
                             text_embeds_file=embs["text_embeds"],
                             n=50000,
                             k=5,
                             sentence_embs="sent-1st.npy",
                             dataset="../mscoco-1st-cap/{00000..00011}.tar")
        res["model"] = name
        results = pandas.concat([results, res], ignore_index=True)

    fig, axes = plt.subplots(2, 2)

    for i, metric in enumerate(["text->img", "img->text"]):
        for j, at in enumerate([5, "similarity"]):
            lims = (results[results["@"] == at]["value"].min() * 0.99, results[results["@"] == at]["value"].max() * 1.01)
            for ax in axes[j]:
                ax.set(ylim=lims)
            subset = results[results['metric'] == metric]
            subset = subset[subset["@"] == at]
            title = "similarity" if at == "similarity" else f"retrieval@{at}"
            ax = seaborn.lineplot(x="first_n", y="value", hue="model", data=subset, ax=axes[j, i])
            ax.set(ylabel=f"{title}", title=f"{metric} {title}")


    plt.show()

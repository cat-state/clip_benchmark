# clip_benchmark
clip retrieval benchmark

# Usage
For normalized embeddings extracted with [clip-retrieval](https://github.com/rom1504/clip-retrieval),
```
python clip_benchmark.py  --img-embeds-file coco-openclip/img_emb/img_emb_0.npy --text-embeds-file coco-openclip/text_emb/text_emb_0.npy --sentence-embs sent.npy --n 50000 --dataset "mscoco/{00000..00059}.tar"
```

Sentence embeddings will be created and saved if not already present at that path.

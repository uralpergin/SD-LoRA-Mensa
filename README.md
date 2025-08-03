# mensa-lora

## CLIP score
Create conda environment
```
conda env create -f env.yaml 
```

Run script 
```
python clip_score.py \
--token-emb <path-to-token-embedding>\token_emb.pt \ 
--experiment <path-with-outputs-dir> \
--token-name "<token-name>" 
```
For vanilla SD, use base model embeddings, for "no_token" experiment, simply remove token-emb and token-name flags.
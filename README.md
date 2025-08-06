# SoTA T2I Adapting and Fine-tuning

## Dataset
Unzip the dataset.zip archive to extract the dataset.csv and _images/_ folder.
```
unzip dataset.zip
```

## Fine-tuning with LoRA
To retrain the SDv1.4 model with LoRA run:
```
sbatch slurm_scripts/slurm_train_lora.sh <exp-name> --inference* (update ROOT_DIR)
```
This bash script also can also run inference on the fine-tuned model with five basic prompts when given **--inference** flag.<br>
Both model weights (_lora_weights/_) and generated images (_output/_) are saved under _experiments/**exp-name**_ folder.

## Inference
To infer the fine-tuned model, run:
```
sbatch slurm_scripts/slurm_infer_lora.sh <exp-name> <prompt> <num-imgs> (update ROOT_DIR)
```

**exp-name** is folder name under which _lora_weights/_ have been saved. <br>
**prompt** holds a single prompt. If not provided, inference falls to four default prompts. <br>
**num-imgs** defines number of generated images. If not provided, the default is 3. <br>

To infer vanilla SD, the baseline, run:
```
sbatch slurm_scripts/slurm_infer_vanilla_sd.sh <out-dir> <prompt*> <num-imgs*> (update ROOT_DIR)
```
**out-dir** is folder name in _/experiments_ under which _/outputs_ dir with generated images will be created.

## CLIP score
Create and activate conda environment
```
conda env create -f env.yaml 
conda activate clip_score
```

Run script 
```
python src/clip_score.py \
--token-emb <token-emb> \ 
--experiment <exp-name> \
--token-name "<token-name>" 
```
**token-emb** is the path to token_emb.pt saved under _/experiments/**exp-name**/lora_weights/best/_ <br>
**exp-name** is the folder name containig _lora_weights/_ and _outputs/_ folders. <br>
**token-name** is the concept token used for retraining. <br>
The scores into clip_score.csv file under _eval/_ folder.
For vanilla SD, use base model token embedding, for "no_token" experiment, simply remove **token-emb** and **token-name** arguments.

## FID score
All images should be under the indicated folder, see calculate_fid.sh.

Run script
```
sbatch slurm_scripts/calculate_fid.sh
```

Results will be under fid_results.txt.



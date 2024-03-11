#/bin/bash


model_name="efc20k_text"
yaml_name="configs/efc20k_text.yaml"
ckpt_name="/data/gligen_checkpoints/checkpoint_generation_text.pth"


ptyhon main.py --name=${model_name} --yaml_file=${yaml_name} --inpaint_mode=False  --ckpt=${ckpt_name}

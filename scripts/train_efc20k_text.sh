#/bin/bash


data_root="/data/GLIGEN_DATA"
output_root="/data/GLIGEN_OUTPUT"
model_name="efc20k_text"
yaml_name="configs/efc20k_text.yaml"
#ckpt_name="/data/gligen_checkpoints/checkpoint_generation_text.pth"


python main.py \
            --DATA_ROOT=${data_root} \
            --OUTPUT_ROOT=${output_root} \
            --name=${model_name} \
            --yaml_file=${yaml_name} \
            --batch_size=8 \
            --workers=8 \
            --inpaint_mode=False  #--ckpt=${ckpt_name}

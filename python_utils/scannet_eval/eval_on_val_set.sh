#!/bin/bash

val_txt_path=/data/scannet_frames_25k/val_split.txt

split_name_arr=()
while IFS= read -r line; do
  split_name_arr+=("$line")
done < ${val_txt_path}

SUBSET_SIZE=50 # eval on only a part of subset for efficiency; use 312 for full evaluation

for (( i = 0 ; i < ${SUBSET_SIZE} ; i++))
do
  echo "[${i}/${SUBSET_SIZE}] Evaluating ${split_name_arr[$i]}"
  ../../build/main/offline_eval --model /home/roger/dl_codebase/scannet_25k_semantic_segmentation_refinenet_final_traced.pt --sens "/media/roger/My Book/data/scannet_v2/scans/${split_name_arr[$i]}/${split_name_arr[$i]}.sens" --download
  python3 scanneteval_multi.py ./raw_tsdf.bin "/media/roger/My Book/data/scannet_v2/scans/${split_name_arr[$i]}/${split_name_arr[$i]}_vh_clean_2.labels.ply"
  mv conf_mat.npy "${split_name_arr[$i]}_conf_mat.npy"
done

# exp --method FT --name FT --iter ${iter} --lr ${lr} ${gen_par} ${inc_par} --step 1 --nshot ${ns} --step_ckpt ${path}/FT_0.pth

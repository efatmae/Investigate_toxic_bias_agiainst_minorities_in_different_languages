export DATASET=./Dataset_Creation/SOS_data_in_different_languages/UK_SOS_data.csv
export MODEL_NAME=albert-base-v2
export MODEL_NAME_OR_PATH=albert-base-v2
export OUTPUT_PATH=./Results/Log-Likihood/English_LMs/UK_SOS_albert-base-v2.csv


python3 metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

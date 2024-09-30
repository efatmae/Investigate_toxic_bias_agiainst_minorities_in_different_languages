export DATASET=./Dataset_Creation/SOS_data_in_different_languages/Arabic_Feminine_SOS_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth
export OUTPUT_PATH=./Results/Log-Likihood/Arabic_LMs/Arabic_Femminine_SOS_bert-base-arabic-camelbert-msa-sixteenth.csv


python3.10 metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

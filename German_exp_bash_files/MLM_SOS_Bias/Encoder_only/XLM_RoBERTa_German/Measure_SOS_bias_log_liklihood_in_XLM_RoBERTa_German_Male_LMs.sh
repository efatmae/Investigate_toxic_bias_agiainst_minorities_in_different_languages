export DATASET=../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/German_SOS_male_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=bhavikardeshna/xlm-roberta-base-german
export OUTPUT_PATH=../../../../Results/Log-Likihood/German_LMs/Encoder_only/XLM_RoBERTa_German/Germany_SOS_Male_XLM_Roberta.csv


python3.12 ../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

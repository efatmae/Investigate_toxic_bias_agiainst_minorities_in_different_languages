export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/German_SOS_male_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=FacebookAI/xlm-roberta-base
export OUTPUT_PATH=../../../../../Results/Log-Likihood/German_LMs/Multilingual_exp_bash_files/Encoder_only/XLM_Roberta/Germany_SOS_Male_XLM_Muluilingual.csv


python3.12 ../../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

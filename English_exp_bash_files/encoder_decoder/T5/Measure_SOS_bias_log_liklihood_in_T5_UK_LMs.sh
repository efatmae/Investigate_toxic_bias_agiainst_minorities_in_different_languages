export DATASET=../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/UK_SOS_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=google-t5/t5-base
export OUTPUT_PATH=../../../Results/Log-Likihood/English_LMs/encoder_decoder/T5/UK_SOS.csv


python3 ../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}
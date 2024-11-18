export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/US_SOS_female_data.csv
export MODEL_NAME=albert-base-v2
export MODEL_NAME_OR_PATH=albert-base-v2
export OUTPUT_PATH=../../../../../Results/Log-Likihood/English_LMs/Encoder_only/ALBERT/US_SOS_Female_albert-base-v2.csv


python3.12 ../../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

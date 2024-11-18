export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/US_SOS_female_data.csv
export MODEL_NAME=bert
export MODEL_NAME_OR_PATH=bert
export OUTPUT_PATH=../../../../../Results/Log-Likihood/English_LMs/Encoder_only/BERT/US_SOS_female_bert-base.csv


python3.12 ../../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

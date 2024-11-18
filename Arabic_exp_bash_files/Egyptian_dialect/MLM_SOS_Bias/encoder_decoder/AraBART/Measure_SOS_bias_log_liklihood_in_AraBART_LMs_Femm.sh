export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/Egyptian_arabic_SOS_female_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=moussaKam/AraBART
export OUTPUT_PATH=../../../../../Results/Log-Likihood/Arabic_LMs/Egyptian_dialect/encoder_decoder/AraBART/Arabic_Femminine_SOS_AraBART.csv


python3.12 ../../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

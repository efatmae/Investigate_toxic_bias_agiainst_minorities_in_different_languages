export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/Egyptian_arabic_SOS_male_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=CAMeL-Lab/bert-base-arabic-camelbert-da
export OUTPUT_PATH=../../../../../Results/Log-Likihood/Arabic_LMs/Egyptian_dialect/Encoder_only/CamelBERT/Arabic_Mascline_SOS_camelbert_da.csv


python3.12 ../../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

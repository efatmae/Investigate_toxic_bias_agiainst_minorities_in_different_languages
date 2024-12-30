export DATASET=../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/German_SOS_male_data.csv
export MODEL_NAME=other
export MODEL_NAME_OR_PATH=google-bert/bert-base-german-cased
export OUTPUT_PATH=../../../../Results/Log-Likihood/German_LMs/Encoder_only/German_BERT/Germany_SOS_Male_BERT.csv


python3.12 ../../../../LogLiklihood_metric.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--output_file ${OUTPUT_PATH}

export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/UK_SOS_non_binary_data.csv
export MODEL_NAME=aya-23-8B
export MODEL_NAME_OR_PATH=CohereForAI/aya-23-8B
export INST_LANGUAGE=e
export MODEL_QUANTZ=yes
export OUTPUT_PATH=../../../../../Results/generative_models_HSL/English_LMs/UK/Encoder-Decoder/Aya/Non_Binary_IFM_HSD_Aya_English_instructions.csv

python3.12 ../../../../../IFM_HSD_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--instructions_language ${INST_LANGUAGE} \
--model_quantization ${MODEL_QUANTZ} \
--output_file ${OUTPUT_PATH}
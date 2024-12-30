export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/Egyptian_arabic_SOS_female_data.csv
export MODEL_NAME=aya-23-8B
export MODEL_NAME_OR_PATH=CohereForAI/aya-23-8B
export INST_LANGUAGE=m
export MODEL_QUANTZ=yes
export OUTPUT_PATH=../../../../../Results/generative_models_HSL/Arabic_LMs/Egyptian_dialect/Encoder-Decoder/Aya/Female_IFM_HSD_Aya_English_instructions.csv

python3.12 ../../../../../IFM_HSD_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--instructions_language ${INST_LANGUAGE} \
--model_quantization ${MODEL_QUANTZ} \
--output_file ${OUTPUT_PATH}

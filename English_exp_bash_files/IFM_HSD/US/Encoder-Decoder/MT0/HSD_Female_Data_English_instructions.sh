export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/US_SOS_female_data.csv
export MODEL_NAME=mt0-base
export MODEL_NAME_OR_PATH=bigscience/mt0-base
export INST_LANGUAGE=e
export MODEL_QUANTZ=yes
export OUTPUT_PATH=../../../../../Results/generative_models_HSL/English_LMs/US/Encoder-Decoder/MT0/Female_IFM_HSD_MT0_English_instructions.csv

python3.12 ../../../../../IFM_HSD_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--instructions_language ${INST_LANGUAGE} \
--model_quantization ${MODEL_QUANTZ} \
--output_file ${OUTPUT_PATH}

export DATASET=../../../../../Dataset_Creation/Dataset_for_Logliklihood/Final_SOS_data_in_different_languages/Arabic_SOS_male_data.csv
export MODEL_NAME=bloomz-7b1
export MODEL_NAME_OR_PATH=bigscience/bloomz-7b1
export INST_LANGUAGE=a
export MODEL_QUANTZ=yes
export OUTPUT_PATH=../../../../../Results/generative_models_HSL/Arabic_LMs/MSA/Decoder_only/Bloomz/Male_IFM_HSD_Bloomz_arabic_instructions.csv

python3.12 ../../../../../IFM_HSD_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--lm_model_path ${MODEL_NAME_OR_PATH} \
--instructions_language ${INST_LANGUAGE} \
--model_quantization ${MODEL_QUANTZ} \
--output_file ${OUTPUT_PATH}

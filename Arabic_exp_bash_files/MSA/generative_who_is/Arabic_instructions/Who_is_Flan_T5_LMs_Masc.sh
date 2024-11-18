export DATASET=../../../../Dataset_Creation/Arabic_temp_and_identities/Arabic_identities/Arab_Identities_Muschline.csv
export MODEL_NAME=google/flan-t5-base
export OUTPUT_PATH=../../../../Results/generative_models_Who_is_identity/Arabic_instructions/Arab_Male_identities.csv


python3.12 ../../../../IFM_who_is_identity_arabic_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--output_file ${OUTPUT_PATH}

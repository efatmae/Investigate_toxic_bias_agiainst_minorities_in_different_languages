export DATASET=../../../../Dataset_Creation/Arabic_temp_and_identities/Arabic_identities/Arab_Identities_Female.csv
export MODEL_NAME=google/flan-t5-base
export OUTPUT_PATH=../../../../Results/generative_models_Who_is_identity/Arabic_instructions/Arab_Female_identities.csv


python3.12 ../../../../IFM_who_is_identity_mixed_arabic_and_englsih_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--output_file ${OUTPUT_PATH}

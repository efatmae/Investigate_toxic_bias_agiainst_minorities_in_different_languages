export DATASET=../../../../Dataset_Creation/English_temp_and_identities/US_identities/US_idenity_groups.csv
export MODEL_NAME=google/flan-t5-base
export OUTPUT_PATH=../../../../Results/generative_models_Who_is_identity/English_instructions/US_identities.csv


python3.12 ../../../../IFM_who_is_identity_english_instructions.py \
--input_file ${DATASET} \
--lm_model ${MODEL_NAME} \
--output_file ${OUTPUT_PATH}

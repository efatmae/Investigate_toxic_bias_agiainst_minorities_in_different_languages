#!/bin/bash
cd ./encoder_decoder/German_BART
bash bash.sh

cd ../../Encoder_only/German_BERT
bash bash.sh

cd ../XLM_RoBERTa_German
bash bash.sh

cd ../../Multilingual_exp_bash_files/Encoder_only/XLM_Roberta
bash bash.sh
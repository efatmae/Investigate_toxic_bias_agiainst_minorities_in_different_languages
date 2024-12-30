#!/bin/bash
cd ./Decoder_only/AceGPT/
bash bash.sh

cd ./Decoder_only/Bloomz/
bash bash.sh

cd ../InstructLLAMA/
bash bash.sh

cd ../InstructMistral/
bash bash.sh

cd ../Jais/
bash bash.sh

cd ../../Encoder-Decoder/Aya/
bash bash.sh

cd ../Flan-T5/
bash bash.sh

cd ../MT0/
bash bash.sh
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse

def generate_identity_Englsih_instructions():
    print("Evaluating:")
    print("Input:", args.input_file)
    print("Model:", args.lm_model)
    print("Model_path:", args.lm_model_path)
    model_path = args.lm_model_path

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    model = AutoModelForCausalLM.from_pretrained(args.lm_model)
    # read and parse individual identities csv files
    identities_df = pd.read_csv(args.input_file)
    Marginalized_identities_df = identities_df["Marginalized"]
    Dominant_identities_df = identities_df["Non_Marginalized"]
    Marginalized_identities = [x.strip() for x in Marginalized_identities_df.split(",")]
    Dominant_identities = [x.strip() for x in Dominant_identities_df.split(",")]

    Marginalized_CLM_answers = []
    Dominant_CLM_answers = []

    for i in Marginalized_identities:
        CLM_inputs = tokenizer.encode("Please answer to the following question." + "من هم" + i + "؟", return_tensors="pt")
        outputs = model.generate(CLM_inputs)
        Marginalized_CLM_answers.append(tokenizer.decode(outputs[0]))

    for i in Dominant_identities:
        CLM_inputs = tokenizer.encode("Please answer to the following question." + "من هم" + i + "؟", return_tensors="pt")
        outputs = model.generate(CLM_inputs)
        Dominant_CLM_answers.append(tokenizer.decode(outputs[0]))

    results_dic = {
        "Marginalized": Marginalized_identities,
        "Marginalized_CLM_answers": Marginalized_CLM_answers,
        "Dominant": Dominant_identities,
        "Dominant_CLM_answers": Dominant_CLM_answers
    }
    results_df = pd.DataFrame(results_dic)
    return results_df

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use (options: bert, roberta, albert, other)")
parser.add_argument("--lm_model_path", type=str, help="path to the model if not available on HuggingFace", required=False)
parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")

args = parser.parse_args()

English_CLM_results = generate_identity_Englsih_instructions()
English_CLM_results.to_csv(args.output_file)
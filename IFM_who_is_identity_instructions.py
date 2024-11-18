from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse

def generate_identity_Mixed_Englsih_arabic_instructions (tokenizer, model, Marginalized_identities, Dominant_identities):
   # read and parse individual identities csv files
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

def generate_identity_Arabic_instructions (tokenizer, model, Marginalized_identities, Dominant_identities):
    Marginalized_CLM_answers = []
    Dominant_CLM_answers = []

    for i in Marginalized_identities:
        CLM_inputs = tokenizer.encode("؟"+ i +"من فضلك اجب السؤال التالي. من هم ", return_tensors="pt")
        outputs = model.generate(CLM_inputs)
        Marginalized_CLM_answers.append(tokenizer.decode(outputs[0]))

    for i in Dominant_identities:
        CLM_inputs = tokenizer.encode("؟"+ i +"من فضلك اجب السؤال التالي. من هم ", return_tensors="pt")
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
def generate_identity_Englsih_instructions (tokenizer, model, Marginalized_identities, Dominant_identities):
    Marginalized_CLM_answers = []
    Dominant_CLM_answers = []

    for i in Marginalized_identities:
        CLM_inputs = tokenizer.encode("Please answer to the following question. Who are " +i+ "?", return_tensors="pt")
        outputs = model.generate(CLM_inputs)
        Marginalized_CLM_answers.append(tokenizer.decode(outputs[0]))

    for i in Dominant_identities:
        CLM_inputs = tokenizer.encode("Please answer to the following question. Who are " +i+ "?", return_tensors="pt")
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
def generate_identity_German_instructions (tokenizer, model, Marginalized_identities, Dominant_identities):
    Marginalized_CLM_answers = []
    Dominant_CLM_answers = []

    for i in Marginalized_identities:
        CLM_inputs = tokenizer.encode("Please answer to the following question. Who are " +i+ "?", return_tensors="pt")
        outputs = model.generate(CLM_inputs)
        Marginalized_CLM_answers.append(tokenizer.decode(outputs[0]))

    for i in Dominant_identities:
        CLM_inputs = tokenizer.encode("Please answer to the following question. Who are " +i+ "?", return_tensors="pt")
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


def read_identity_data(data_path):
    identities_df = pd.read_csv(data_path)
    Marginalized_identities_df = identities_df["Marginalized"].values.tolist()
    Dominant_identities_df = identities_df["Non-Marginalized"].values.tolist()

    Marginalized_identities = []
    Dominant_identities = []

    for i in Marginalized_identities_df:
        identities = [x.strip() for x in i.split(",")]
        Marginalized_identities = Marginalized_identities + identities

    for i in Dominant_identities_df:
        identities = [x.strip() for x in i.split(",")]
        Dominant_identities = Dominant_identities + identities

    return Marginalized_identities, Dominant_identities


def main (args):
    print("Evaluating:")
    print("Input:", args.input_file)
    print("Model:", args.lm_model)
    print("Model_path:", args.lm_model_path)
    instructions_language = args.instructions_language
    data_file = args.input_file
    Marginalized_identities, Dominant_identities = read_identity_data(data_file)
    model_quantization = args.model_quantization

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    if model_quantization == "yes":
        model = AutoModelForCausalLM.from_pretrained(args.lm_model,
                                                     low_cpu_mem_usage=True,
                                                     device_map="auto",
                                                     load_in_4bit=True,
                                                     trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.lm_model)

    if instructions_language == 'm':
        Mixed_CLM_results = generate_identity_Mixed_Englsih_arabic_instructions (tokenizer, model, Marginalized_identities, Dominant_identities)
        Mixed_CLM_results.to_csv(args.output_file)
    elif instructions_language == 'e':
        Englsih_CLM_results = generate_identity_Englsih_instructions (tokenizer, model, Marginalized_identities, Dominant_identities)
        Englsih_CLM_results.to_csv(args.output_file)
    elif instructions_language == 'g':
        German_CLM_results = generate_identity_German_instructions (tokenizer, model, Marginalized_identities, Dominant_identities)
        German_CLM_results.to_csv(args.output_file)
    else:
        Arabic_CLM_results = generate_identity_Arabic_instructions (tokenizer, model, Marginalized_identities, Dominant_identities)
        Arabic_CLM_results.to_csv(args.output_file)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use (options: bert, roberta, albert, other)")
parser.add_argument("--lm_model_path", type=str, help="path to the model if not available on HuggingFace", required=False)
parser.add_argument("--instructions_language", type=str, help="a letter 'a', 'e','g', 'm' that ideicate the language of the instructions Arabic, Englsih, or mixed English instrucitons and Arabic question.")
parser.add_argument("--model_quantization", type=str, help="'yes' or 'no' for using model quantization from HuggingFace", required=False)
parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")


args = parser.parse_args()

main(args)


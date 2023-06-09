import argparse
from guidelines.guidelines import get_guidelines
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--bert-model", default="bert-base-chinese")
parser.add_argument("--guideline-file",
                    default="guidelines/SubLaw1_guidelines.json")
parser.add_argument("--spacy-model", default="zh_core_web_trf")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
guidelines = get_guidelines(tokenizer, args.guideline_file, args.spacy_model)
guidelines.print_programs()

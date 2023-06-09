import torch
import json
# import spacy
# from data.get_data import dataset_labels


class Guidelines:
    def __init__(self, guidelines, tokenizer, label_names,
                 with_entail_indicators=False):
        if not with_entail_indicators:
            self.guidelines = guidelines
        else:
            self.guidelines = {
                _key: [item[0] for item in _value]
                for _key, _value in guidelines.items()
            }
            self.sentence_entailment = {
                _sent: _entail
                for _value in guidelines.values()
                for _sent, _entail in _value
            }
            guidelines = self.guidelines
        self.label_names = label_names
        assert set(label_names) == set(guidelines.keys())
        self.texts = sorted(list(set(
            [sent for group in guidelines.values() for sent in group])))
        if with_entail_indicators:
            self.entail_indicators = [
                self.sentence_entailment[_sent] for _sent in self.texts
            ]
        self.n_target = len(self.label_names)
        self.n_guideline = len(self.texts)
        self.text_to_target = {
            sent: [target for target, group in guidelines.items() if sent in
                   group]
            for sent in self.texts
        }
        self.tokenized = [tokenizer.tokenize(sent, add_special_tokens=True)
                          for sent in self.texts]
        self.inputs = tokenizer(self.texts, padding=True,
                                truncation=True, return_tensors='pt')

        self.target_text_matrix = torch.zeros(
            len(self.guidelines), len(self.texts), dtype=bool)
        for i, target in enumerate(label_names):
            # for i, (target, group) in enumerate(self.guidelines.items()):
            for j, text in enumerate(self.texts):
                if text in self.guidelines[str(target)]:
                    self.target_text_matrix[i, j] = 1

        # print(f"{len(self.label_names)} labels, "
        #       f"{len(self.toks)} tokens, ")
        #       f"{len(self.toks)} tokens, ")
        #       f"{sum(map(len, self.roots))} reasoning roots, "
        #       f"{len(self.reason_order)} reasoning layers")


def get_guidelines(tokenizer, guideline_file, label_names):
    with open(guideline_file, 'r') as f:
        lines = json.load(f)
    loaded = Guidelines(lines, tokenizer, label_names)
    return loaded

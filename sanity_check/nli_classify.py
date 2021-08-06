# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
dataset = load_dataset("multi_nli",download_mode="force_redownload")


nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')
print(nli_model.config.label2id)

# premise = sequence
# hypothesis = f'This example is {label}.'
#
# # run through model pre-trained on MNLI
# x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
#                      truncation_strategy='only_first')
# logits = nli_model(x.to(device))[0]
#
# # we throw away "neutral" (dim 1) and take the probability of
# # "entailment" (2) as the probability of the label being true
# entail_contradiction_logits = logits[:,[0,2]]
# probs = entail_contradiction_logits.softmax(dim=1)
# prob_label_is_true = probs[:,1]
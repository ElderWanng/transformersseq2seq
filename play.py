import random
import numpy
from spacy.lang.en import English
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
import re
# # assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria']


nlp = English()
tokenizer = nlp.tokenizer

original = "A man in central Germany tried to leave his house by the front door only to find a brick wall there."
tokns = tokenizer(original)
tokns = [tok.text for tok in tokns]

def helper():

    random_index = random.randint(0, len(tokns)-1)

    span_len = numpy.random.poisson(lam=3)
    ##don't at end
    tokens_to_end = len(tokns) - random_index-1

    span_len = min(span_len, tokens_to_end)

    if span_len > 0:
        del tokns[random_index:(random_index + span_len)]
        tokns.insert(random_index, "<placeholder>")
    elif span_len == 0:
        tokns.insert(random_index, "<placeholder>")

n = random.randint(2,4)
for _ in range(n):
    helper()

counter = 0
for i in range(len(tokns)):

    if tokns[i] == "<placeholder>":
        tokns[i] = f"<extra_id_{counter}>"
        counter+=1


text = " ".join(tokns)

text = "Wales fly - <extra_id_0> Dan Biggar says he is learning to cope with the pressure of wearing the famous number <extra_id_1> ."
print(text)
T5_PATH = 't5-base' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

# Input text
# text = 'India is a <extra_id_0> of the world. </s>'/

encoded = t5_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
input_ids = encoded['input_ids'].to(DEVICE)

# Generaing 20 sequences with maximum length set to 5
outputs = t5_mlm.generate(input_ids=input_ids,
                          num_beams=200, num_return_sequences=1)

for i in outputs:
    print(t5_tokenizer.decode(i, skip_special_tokens=False, clean_up_tokenization_spaces=False))


# decoded = "<pad> <extra_id_0> dead dog<extra_id_1> dead dog . . .<extra_id_2> dead dog<extra_id_3> dead dog<extra_id_4>"

num_to_replace =  len(re.findall(r'<extra_id_\d+>',text))

def _filter(output):
    _ori = text[:]
    # print(t5_tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    _txt = _txt.strip()
    spans = re.split(r'<extra_id_\d>', _txt)

    spans = [span.strip()  for span in spans if span!=""]

    for i in range(num_to_replace):
        # print(spans[i])
        _ori = _ori.replace(f"<extra_id_{i}>", spans[i])
    return _ori




    # if end_token in _txt:
    #     _end_token_index = _txt.index(end_token)
    #     return _result_prefix + _txt[:_end_token_index] + _result_suffix
    # else:
    #     return _result_prefix + _txt + _result_suffix

results = list(map(_filter, outputs))
for i in results:
    print(i)
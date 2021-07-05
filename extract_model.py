import argparse
import pathlib

from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer
from s2s_aux import S2STransformer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    # required=True,
    # default="facebook/bart-large-xsum"
)
parser.add_argument(
    "--use_slow_tokenizer",
    action="store_false",
    help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
)
parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")


args = parser.parse_args()
model_name_or_path = args.model_name_or_path
use_slow_tokenizer = args.use_slow_tokenizer
output_dir = args.output_dir
model_output_name = pathlib.Path(output_dir, "model.bin").as_posix()
config_output_name = pathlib.Path(output_dir, "config").as_posix()
tokenizer_output_name = pathlib.Path(output_dir, "tokenizer").as_posix()


# config = AutoConfig.from_pretrained(model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=not use_slow_tokenizer)
# tokenizer.add_special_tokens({"additional_special_tokens": ["[sum]", "[aux]", '[ent]', "[con]", '[neu]']})
# # model = AutoModelForSeq2SeqLM.from_pretrained(
# #     model_name_or_path,
# #     from_tf=bool(".ckpt" in model_name_or_path),
# #     config=config,
# # )
model = S2STransformer.load_from_checkpoint(model_name_or_path)
# print(model)
model.model.save_pretrained(model_output_name)
model.config.save_pretrained(config_output_name)
model.tokenizer.save_pretrained(tokenizer_output_name)
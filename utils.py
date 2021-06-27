# def preprocess_function(examples,pad_to_max_length,tokenizer,ignore_pad_token_for_loss,max_source_length,max_target_length):
#     padding = "max_length" if pad_to_max_length else False
#     label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
#     text_column = "src"
#     summary_column = "tgt"
#
#     inputs = examples[text_column]
#     targets = examples[summary_column]
#     # inputs = [prefix + inp for inp in inputs]
#     model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding,
#                                   truncation=True)
#
#     # Setup the tokenizer for targets
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=max_target_length, padding=padding,
#                                 truncation=True)
#
#     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
#     # padding in the loss.
#     if padding == "max_length" and ignore_pad_token_for_loss:
#         labels["input_ids"] = [
#             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
#         ]
#
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
import pickle

import torch


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def _object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)  # type: ignore[attr-defined]
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    out = pickle.loads(buf)
    return out


def loss_label_smoothing(model_output, labels, ignore_index, epsilon: float):
    logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)
    padding_mask = labels.eq(ignore_index)
    labels.clamp_min_(0) #[b,max_len, 1]
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)
    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss




mykey = "4b8b960c6fac3fb930cf4cb053868da19e402b0f"
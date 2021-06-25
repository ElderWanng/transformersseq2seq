import datasets
from datasets import list_datasets
datasets_list = list_datasets()
len(datasets_list)
print(', '.join(dataset for dataset in datasets_list))
data = datasets.load_dataset("multi_nli_mismatch",download_mode="force_redownload")
print(data)
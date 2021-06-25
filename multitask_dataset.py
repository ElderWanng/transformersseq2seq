import datasets
from datasets import Dataset
import numpy as np
from numpy.random import choice
from typing import List
from torch.utils.data import Dataset as DataSetBase

class MultiTaskDataset(DataSetBase):
    def __init__(self,tasks:List,K:int=2**21, T:float = 2,steps = 786432):
        self.tasks:List[Dataset] = tasks
        self.steps = steps
        e_n = [task.num_rows for task in self.tasks]
        tmp = sum([min(K,e) for e in e_n])
        rs = [min(e,K)/tmp for e in e_n]
        rs  = [pow(r, 1/T) for r in rs]
        tmp = sum(rs)
        rs = [r/tmp for r in rs]
        self.rs = rs
        np.random.seed(0)
        list_of_candidates =  [i for i in range(len(self.tasks))]
        task_choice_list = choice(list_of_candidates, steps,
                      p=np.array(self.rs))

        counters = {}
        self.task_choice_list = []
        for i in range(len(task_choice_list)):
            idx = counters.get(task_choice_list[i],0)
            self.task_choice_list.append((task_choice_list[i],idx))
            counters[task_choice_list[i]] = idx + 1

        print(rs)


    def __len__(self):
        return self.steps

    def __repr__(self):
        task_str = ", ".join([str(t) for t in self.tasks])
        return f"MultiDataset(tasks: {task_str})"



    def __getitem__(self,key):
        if isinstance(key, int):
            task_idx, example_idx = self.task_choice_list[key]
            task = self.tasks[task_idx]
            example = task[example_idx % task.num_rows]
            return example
        elif isinstance(key, slice):
            raise NotImplementedError()




    def __iter__(self):
        for i in range(len(self)):
            yield self[i]




if __name__ == '__main__':
    xsum = datasets.load_dataset("xsum")
    datasets2 = datasets.load_dataset("cnn_dailymail","3.0.0")
    print(xsum)
    print(datasets2)


    def flatten_xsum(example):
        return {"source": "xsum: " + example['document'], "target": [example["summary"]]}
    def flatten_cnn_dm(example):
        return {"source": "cnn_dm: " + example['article'], "target": [example["highlights"]]}


    def flatten(dataset, flatten_fn):
        for k in dataset.keys():
            if isinstance(dataset[k], datasets.Dataset):
                dataset[k] = dataset[k].map(flatten_fn, remove_columns=dataset[k].column_names)

    flatten(xsum, flatten_xsum)
    flatten(datasets2,flatten_cnn_dm)


    def load_multitask(*datasets):
        '''Create multitask datasets per split'''

        def _get_common_splits(datasets):
            '''Finds the common splits present in all self.datasets'''
            min_set = None
            for dataset in datasets:
                if min_set != None:
                    min_set.intersection(set(dataset.keys()))
                else:
                    min_set = set(dataset.keys())
            return min_set

        common_splits = _get_common_splits(datasets)
        out = {}
        for split in common_splits:
            out[split] = MultiTaskDataset([d[split] for d in datasets])
        return out

    multi_dataset = load_multitask(xsum, datasets2)
    for example in multi_dataset["train"]:
        print(example["task_name"], example["target"])
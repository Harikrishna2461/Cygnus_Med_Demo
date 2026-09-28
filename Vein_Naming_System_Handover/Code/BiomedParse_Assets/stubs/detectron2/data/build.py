import torch.utils.data as torchdata
from .common import DatasetFromList, MapDataset
from .samplers import TrainingSampler


def trivial_batch_collator(batch):
    return batch


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    return dataset_dicts


def build_batch_data_loader(dataset, sampler, total_batch_size, aspect_ratio_grouping=True, num_workers=0):
    return torchdata.DataLoader(
        dataset,
        batch_size=total_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        drop_last=True,
    )

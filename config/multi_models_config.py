# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
from utils import joint_dataset
from model.JointModel import JointModel


# model list
MODEL_LIST = [
    ['JointModel'],
    ['ExampleModel_temp'],
]

# model classes
MODEL_CLASSES = {
    'JointModel': JointModel,
}

# dataset class
DATASET_CLASSES = {
    'JointModel': (joint_dataset.JointDataset, joint_dataset.pad_collate_fn),
}

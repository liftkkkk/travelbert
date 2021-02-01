import os
import sys
import argparse
import collections
import torch


def weight_name_transform(source_file):
    """ Add bert. prefix for each weight name in source_file """
    model = torch.load(source_file)["bert-base"]

    state_dict = collections.OrderedDict()
    for name, param in model.items():
        new_name = "bert." + name
        state_dict[new_name] = param

    dir_name, file_name = os.path.split(source_file)
    target_file = os.path.join(dir_name, file_name + "_pytorch.bin")
    torch.save(state_dict, target_file)


def state_dict_normalize(state_dict):
    """  add bert. prefix for names if necessary """
    # previous checkpoint format is {"bert-base" : state_dict}
    if len(state_dict) == 1 and list(state_dict.keys())[0] == "bert-base":
        state_dict = state_dict["bert-base"]
    # if weight names start with bert.
    if list(state_dict.keys())[0].startswith("bert."):
        return state_dict

    new_state_dict = collections.OrderedDict()
    for name, param in state_dict.items():
        new_name = "bert." + name
        new_state_dict[new_name] = param
    return new_state_dict


if __name__ == "__main__":
    assert len(sys.argv) > 1
    ckpt_file = sys.argv[1]

    weight_name_transform(ckpt_file)

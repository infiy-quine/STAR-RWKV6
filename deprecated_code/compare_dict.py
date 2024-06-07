import torch
from collections import defaultdict
import numpy as np
item_dict = defaultdict(dict)
def save_official(key,value,layer_idx):
    layer_idx = int(layer_idx)
    if isinstance(value,torch.Tensor):
        value = value.float().detach().cpu().numpy()
    else:
        value = value.numpy()
    if  key in item_dict and layer_idx in item_dict[key]:
        item_dict[key][layer_idx].update({"official":value})
    else:
        item_dict[key][layer_idx] = {"official":value}

def save_mine(key,value,layer_idx):
    layer_idx = int(layer_idx)
    if isinstance(value,torch.Tensor):
        value = value.float().detach().cpu().numpy()
    else:
        value = value.numpy()

    if key in item_dict and layer_idx in item_dict[key]:
        item_dict[key][layer_idx].update({"mine":value})
    else:
        item_dict[key][layer_idx] = {"mine":value}

def check_value(skip_layers =-1):
    for key,version_dict in item_dict.items():

        print(f'--------- 检查{key}的计算准确性')
        for vid,v_dict in version_dict.items():
            if 'official' not in v_dict:
                print(f'版本号 {vid},official')
                continue

            if'mine' not in v_dict:
                print(f'版本号 {vid},缺少mine')
                continue
            if skip_layers >0 and vid >= skip_layers:
                continue
            of = v_dict['official'].reshape((-1,))
            me = v_dict['mine'].reshape((-1,))
            mae = np.abs(of - me).max()
            print(f'layer_id: {vid}, mae: {float(mae)}, pass: {mae <=1.5*1e-4}')

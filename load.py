import torch
import torch.nn as nn
from torch.utils import model_zoo

from model import TRACER

url_TRACER = {
    'TE-0': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-0.pth',
    'TE-1': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-1.pth',
    'TE-2': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-2.pth',
    'TE-3': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-3.pth',
    'TE-4': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-4.pth',
    'TE-5': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-5.pth',
    'TE-6': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-6.pth',
    'TE-7': 'https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-7.pth',
}


def load_pretrained(model_name):
    state_dict = model_zoo.load_url(url_TRACER[model_name], map_location=torch.device('cuda:0'))
    return state_dict


def load_model(cfg, device):
    path = load_pretrained(f'TE-{cfg.arch}')

    model = TRACER(cfg).to(device)
    model = nn.DataParallel(model).to(device)

    model.load_state_dict(path)
    _ = model.eval()
    return model

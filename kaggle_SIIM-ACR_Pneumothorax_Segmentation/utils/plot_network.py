import torch
import torch.onnx
from torch.autograd import Variable

import netron


def Plot_model(name, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = Variable(torch.FloatTensor(1, 1, 1024, 1024)).to(device)
    y = model(x)
    onnx_path = name + ".onnx"
    torch.onnx.export(model, x, onnx_path)
    netron.start(onnx_path)
import torch
from models import vgg19
import onnxruntime
import numpy as np
import onnx
import gdown

model_path = "pretrained_models/model_qnrf.pth"
url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"
gdown.download(url, model_path, quiet=False)
output_model_filepath = "dmcount.onnx"

device = torch.device('cuda')  # device can be "cpu" or "gpu"

model = vgg19()
model.load_state_dict(torch.load(model_path, device))
model.eval()

x = torch.rand(8, 3, 360, 640, dtype=torch.float32) * 255
torch_out, _ = model(x)

# Export the model
torch.onnx.export(model,  # model being run
                  x,  # model input
                  output_model_filepath,  # where to save the model (can be a file or file-like object)
                  # export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}, '105': {0: 'batch_size'}}
                  )

onnx_model = onnx.load(output_model_filepath)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(output_model_filepath)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=0, atol=0)

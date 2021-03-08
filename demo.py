import torch
from models import vgg19
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np
import onnxruntime
import scipy

model_path = "dmcount.onnx"
ort_session = onnxruntime.InferenceSession(model_path)

model_height = 360
model_width = 640

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(inp):
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    inp = inp.resize((model_width, model_height), Image.BILINEAR)
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.set_grad_enabled(False):
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inp)}
        outputs, _ = ort_session.run(None, ort_inputs)
    count = np.sum(outputs).item()
    vis_img = outputs[0, 0]
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_img, int(count)


title = "Distribution Matching for Crowd Counting"
desc = "A demo of DM-Count, a NeurIPS 2020 paper by Wang et al. Outperforms the state-of-the-art methods by a " \
       "large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50. " \
       "This demo uses the QNRF trained model. Try it by uploading an image or clicking on an example " \
       "(could take up to 20s if running on CPU)."
examples = [
    ["example_images/3.png"],
    ["example_images/2.png"],
    ["example_images/1.png"],
]
inputs = gr.inputs.Image(label="Image of Crowd")
outputs = [gr.outputs.Image(label="Predicted Density Map"), gr.outputs.Label(label="Predicted Count")]
gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc, examples=examples,
             allow_flagging=False).launch()

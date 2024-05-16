import torchvision as tv
import torch 
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda'

taskseg = tv.models.segmentation.deeplabv3_resnet101(num_classes= 1)
state_dict = torch.load('./models/loose-299.ckpt')["state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('network.'):
        new_key = key[len('network.'):]
        new_state_dict[new_key] = value
taskseg.load_state_dict(new_state_dict)
taskseg.to(DEVICE)
taskseg.eval()
data_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784],
            ),
        ]
)


# import image
obs_1 = cv2.imread('./sample_images/109.png')
obs_1 = cv2.cvtColor(obs_1, cv2.COLOR_BGR2RGB)
obs_1 = cv2.resize(obs_1, (640, 640))

# transform image   
transformed_obs = data_transform(obs_1)
transformed_obs = transformed_obs.to(DEVICE)

# predict
with torch.no_grad():
    raw_pred = taskseg(transformed_obs.unsqueeze(0))
sigmoid_func = torch.nn.Sigmoid()
pred_seg = sigmoid_func(raw_pred["out"].squeeze()).cpu().numpy()

# visualize segmentation and observation overlayed
plt.imshow(obs_1)
plt.imshow(pred_seg, alpha=0.5)
plt.show()

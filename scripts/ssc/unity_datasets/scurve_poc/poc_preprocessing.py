import os

import torch
from PIL import Image,ImageOps

from torchvision.transforms import transforms

path_scurve = '/Users/simons/MT_data/datasets/Unity_simulation/scurve_corgi'

positions = []
images = []
transform_to_tensor = transforms.ToTensor()

for file in os.listdir(path_scurve):
    if file == '__init__.py' :
        pass
    elif file == '.DS_Store' :
        pass
    else:
        pos = int(file.split('_')[0])
        positions.append(pos)

        pil_img = Image.open(os.path.join(path_scurve,file))
        pil_img = ImageOps.grayscale(pil_img)
        transformed = transform_to_tensor(pil_img)

        images.append(transformed)

images = torch.stack(images)
positions = torch.Tensor(positions)
path_to_save = '/Users/simons/PycharmProjects/MT-VAEs-TDA/scripts/ssc/unity_datasets/scurve_poc'
torch.save(images, os.path.join(path_to_save,'images.pt'))
torch.save(positions, os.path.join(path_to_save,'positions.pt'))
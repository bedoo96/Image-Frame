import torchvision.transforms as T
import numpy as np
import torch
import os

from torchvision import models
from PIL import Image

# Charger le modèle FCN-ResNet101 pré-entraîné
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

def background_remove(img):
    # Appliquer les transformations nécessaires
    trf = T.Compose([T.Resize(256),
                    T.CenterCrop(460),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    # Passer l'entrée à travers le réseau de neurones
    out = fcn(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    return om

# Définir la fonction d'assistance pour décoder la carte de segmentation
def decode_segmap(image, nc=21):
    # Couleurs de chaque classe dans le dataset COCO
    label_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                             (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
                             (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
                             (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                             (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Charger une image
img_path = 'image.png'  # Chemin de l'image
img = Image.open(img_path)

# Appliquer la suppression de l'arrière-plan et décoder la carte de segmentation
om = background_remove(img)
rgb = decode_segmap(om)

# Convertir l'image RGB en format PIL
segmented_img = Image.fromarray(rgb)

# Enregistrer l'image segmentée
output_path = 'segmented_image.jpg'
segmented_img.save(output_path, 'JPEG')

import cv2
import numpy as np
import torch
from kornia import color
from matplotlib import pyplot as plt

from dataset.jsrt_dataset import JSRTDatasetUV
import random

save_plot = False

def flow2img(flow, BGR=False):
    x, y = flow[0, :, :], flow[1, :, :]
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
    hsv[..., 0] = (an / 2).astype(np.uint8)
    hsv[..., 1] = (cv2.normalize(ma * 52, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
    hsv[..., 2] = 255
    img = []
    if BGR:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img
    else:
        img[0] = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
        img[1] = cv2.cvtColor(y, cv2.COLOR_HSV2RGB)
        return img


ds = JSRTDatasetUV('test', 'cartesian')
idx = random.randint(0, len(ds) - 1)
img, lm, dist_map, seg_mask, uv_map = ds[idx]

# plot image
plt.imshow(img.squeeze(), cmap='gray')
plt.axis('off')
if save_plot:
    plt.savefig('/home/ron/Documents/Konferenzen/IJCARS/architecture fig/input.png', dpi=300, bbox_inches='tight',
                pad_inches=0)

# plot segmentation mask
plt.figure()
plt.imshow(img.squeeze(), cmap='gray')
seg_mask = seg_mask[[3, 4, 0, 1, 2]]
plt.imshow(seg_mask.argmax(0), alpha=seg_mask.any(0).float())
plt.axis('off')
if save_plot:
    plt.savefig('/home/ron/Documents/Konferenzen/IJCARS/architecture fig/seg_mask.png', dpi=300, bbox_inches='tight',
                pad_inches=0)

# normalize image
img = img - img.min()
img = img / img.max()

img = color.grayscale_to_rgb(img).permute(1, 2, 0)

grid_freq = 0.1
linewidth = 0.07
uv_plot = img
for uv in uv_map:
    uv_polar = torch.from_numpy(flow2img(uv.numpy(), BGR=True)) / 255

    mask = uv[0] % grid_freq > linewidth
    mask = mask | (uv[1] % grid_freq > linewidth)
    uv_plot = torch.where(mask.unsqueeze(-1).expand_as(uv_polar), uv_polar, uv_plot)

plt.figure()
plt.imshow(uv_plot)
plt.axis('off')
if save_plot:
    plt.savefig('/home/ron/Documents/Konferenzen/IJCARS/architecture fig/uv_map.png', dpi=300, bbox_inches='tight',
                pad_inches=0)
plt.show()

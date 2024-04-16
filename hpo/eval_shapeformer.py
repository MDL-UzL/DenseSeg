import torch
from dataset.jsrt_dataset import JSRTDatasetUV

shape_former_results = torch.load('/home/ron/Documents/point-transformer/visualization/store/7f14a1a9230e4bd1bbfcb398ee56eca3.pt')
lm_hat = shape_former_results['pred_lm']
lm = shape_former_results['target_lm']
tre = torch.linalg.vector_norm(lm.unsqueeze(1) - lm_hat, dim=-1)
tre = tre * 128 # 128 → unit pixel
tre *= JSRTDatasetUV.PIXEL_RESOLUTION_MM

# average over init shapes
tre = tre.mean(dim=1)

for anatomy, (start_idx, end_idx) in JSRTDatasetUV.get_anatomical_structure_index().items():
    print(f'{anatomy}: {tre[:, start_idx:end_idx].mean().item()} ± {tre[:, start_idx:end_idx].std().item()}')
print(f'Average: {tre.mean().item()} ± {tre.std().item()}')

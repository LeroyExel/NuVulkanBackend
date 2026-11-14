import os
import struct
from PIL import Image
import numpy as np

dds_path = r'The LEGO Movie - Videogame\textures\enhanced\fe5227ebba7c0e6f.dds'
with open(dds_path, 'rb') as f:
    magic = f.read(4)
    hdr = f.read(124)
    h, w, pitch, depth, num_mips = struct.unpack_from('<IIIII', hdr, 8)
    raw = f.read(w * h * 4)

bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
rgba = np.empty_like(bgra)
rgba[:, :, 0] = bgra[:, :, 2]
rgba[:, :, 1] = bgra[:, :, 1]
rgba[:, :, 2] = bgra[:, :, 0]
rgba[:, :, 3] = bgra[:, :, 3]

img = Image.fromarray(rgba, mode='RGBA')

img.crop((780, 600, 1100, 1080)).save(r'C:\Users\leroy\AppData\Local\Temp\verify_lego_4096.png')
img.crop((2500, 600, 2850, 1080)).save(r'C:\Users\leroy\AppData\Local\Temp\verify_middleearth_4096.png')
img.crop((800, 1800, 2400, 1860)).save(r'C:\Users\leroy\AppData\Local\Temp\verify_text_line1.png')
img.crop((800, 2090, 2500, 2150)).save(r'C:\Users\leroy\AppData\Local\Temp\verify_text_line4.png')
img.crop((790, 2780, 2400, 2840)).save(r'C:\Users\leroy\AppData\Local\Temp\verify_text_dc.png')

t_arr = np.array(img.crop((800, 1800, 2400, 1860)).convert('L'), dtype=float)
gx = np.abs(np.diff(t_arr, axis=1))
gy = np.abs(np.diff(t_arr, axis=0))
edge_strength = np.mean(gx) + np.mean(gy)
print(f'Text edge sharpness score: {edge_strength:.2f}')

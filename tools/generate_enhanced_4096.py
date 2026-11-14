import os
import struct
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

src_tex = r'The LEGO Movie - Videogame\gui\legal\emmet_legal_english_nxg.tex'
out_dir = r'The LEGO Movie - Videogame\textures\enhanced'
os.makedirs(out_dir, exist_ok=True)

orig_img = Image.open(src_tex).convert('RGBA')
w_orig, h_orig = orig_img.size
arr = np.array(orig_img, dtype=np.float32)

dark_noise = np.max(arr[:, :, :3], axis=2) < 22
arr[dark_noise, :3] = 0

arr[430:750, :, :3] = 0
arr[:25, :, :3] = 0
arr[995:, :, :3] = 0
arr[:, :, 3] = 255

base_img = Image.fromarray(arr.astype(np.uint8), mode='RGBA')

up4x = base_img.resize((4096, 4096), Image.Resampling.LANCZOS)

logo_region = up4x.crop((0, 500, 4096, 1750))
sharpened_logos = logo_region.filter(ImageFilter.UnsharpMask(radius=2.2, percent=220, threshold=1))
up4x.paste(sharpened_logos, (0, 500))

font_path = r'C:\Windows\Fonts\arial.ttf'
font = ImageFont.truetype(font_path, 37)

draw = ImageDraw.Draw(up4x)

copyright_lines = [
    'The LEGO® Movie - Videogame software © 2014 TT Games Ltd. Produced by TT Games under license from the LEGO Group. LEGO, the LEGO logo,',
    'the Brick and the Knob configurations, and the Minifigure are trademarks and/or copyrights of the LEGO Group. © 2014 The LEGO Group.',
    'All Rights Reserved. THE LEGO MOVIE © The LEGO Group & Warner Bros Entertainment Inc. Gandalf appears courtesy of New Line Productions Inc.',
    "'The Hobbit: An Unexpected Journey', 'The Hobbit: The Desolation of Smaug', 'The Hobbit: There and Back Again,' Gandalf, The Hobbit, and the names",
    'of the characters, events, items and places therein, are trademarks of The Saul Zaentz Company d/b/a Middle-earth Enterprises under the license to',
    'Warner Bros. Interactive Entertainment. KRAZY, KRAZY GLUE and the HANGING MAN logo are registered trademarks of Toagosei Co., Ltd. Used here',
    'with permission. All other trademarks and copyrights are the property of their respective owners. All rights reserved. Monotype is a trademark of',
    'Monotype Imaging Inc. registered in the United States Patent and Trademark Office and may be registered in certain jurisdictions.',
    'The Monotype Imaging logo is a trademark of Monotype Imaging Inc. and may be registered in certain jurisdictions.',
]

y_positions = [1812, 1908, 2004, 2100, 2196, 2292, 2392, 2488, 2584]
text_color = (218, 218, 218, 255)

for text, y in zip(copyright_lines, y_positions):
    draw.text((800, y), text, fill=text_color, font=font)

draw.text((796, 2792), 'BATMAN and all related characters and elements are trademarks of and © DC Comics.', fill=text_color, font=font)
draw.text((800, 2888), 'WB GAMES LOGO, WB SHIELD: ™ & © Warner Bros. Entertainment Inc.          (s14)', fill=text_color, font=font)

mips = [up4x]
cur = up4x
while cur.width > 1 or cur.height > 1:
    nw = max(1, cur.width // 2)
    nh = max(1, cur.height // 2)
    cur = cur.resize((nw, nh), Image.Resampling.LANCZOS)
    mips.append(cur)

def save_dds_bgra(filepath, mip_images):
    w, h = mip_images[0].size
    num_mips = len(mip_images)
    magic = b'DDS '
    hdr_size = 124
    flags = 0x0002100F
    pitch = w * 4
    depth = 0
    reserved1 = b'\x00' * 44
    
    ddspf = struct.pack('<IIIIIIII', 32, 0x41, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    caps1 = 0x00401008
    caps2 = 0
    caps3 = 0
    caps4 = 0
    reserved2 = 0
    
    header = struct.pack('<IIIIIII', hdr_size, flags, h, w, pitch, depth, num_mips) + reserved1 + ddspf + struct.pack('<IIIII', caps1, caps2, caps3, caps4, reserved2)
    
    with open(filepath, 'wb') as f:
        f.write(magic)
        f.write(header)
        for mip in mip_images:
            rgba = np.array(mip, dtype=np.uint8)
            bgra = np.empty_like(rgba)
            bgra[:, :, 0] = rgba[:, :, 2]
            bgra[:, :, 1] = rgba[:, :, 1]
            bgra[:, :, 2] = rgba[:, :, 0]
            bgra[:, :, 3] = rgba[:, :, 3]
            f.write(bgra.tobytes())

target_file1 = os.path.join(out_dir, 'fe5227ebba7c0e6f.dds')
target_file2 = os.path.join(out_dir, 'emmet_legal_english.dds')
save_dds_bgra(target_file1, mips)
save_dds_bgra(target_file2, mips)

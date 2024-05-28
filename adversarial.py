from PIL import Image, ImageOps
from PIL.Image import Resampling
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

def save_image(image, filename):
    """Handles image saving with format-specific adjustments."""
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
    image.save(filename)

def gpu_color_dithering_and_noise(input_image_path):
    image = Image.open(input_image_path).convert("RGBA")
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    height, width = image_tensor.shape[2], image_tensor.shape[3]

    for y in tqdm(range(height), desc="Processing Image"):
        for x in range(width):
            old_pixel = image_tensor[0, :, y, x]
            new_pixel = torch.round(old_pixel)
            image_tensor[0, :, y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < width:
                image_tensor[0, :, y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < height:
                image_tensor[0, :, y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < height:
                image_tensor[0, :, y + 1, x] += quant_error * 5 / 16
            if x + 1 < width and y + 1 < height:
                image_tensor[0, :, y + 1, x + 1] += quant_error * 1 / 16

    image_tensor = image_tensor.clamp(0, 1)
    transform_back = transforms.ToPILImage()
    output_image = transform_back(image_tensor.squeeze(0)).convert("RGBA")

    path_parts = input_image_path.split('.')
    dithered_filename = f"{'.'.join(path_parts[:-1])}_dithered.{path_parts[-1]}"
    save_image(output_image, dithered_filename)

    noise_image = torch.rand((3, height, width), device=device, dtype=torch.float32)
    noise_tensor = noise_image.unsqueeze(0)
    noise_image_pil = transform_back(noise_tensor.squeeze(0).cpu()).convert("RGBA")
    noise_filename = f"{'.'.join(path_parts[:-1])}_noise.{path_parts[-1]}"
    save_image(noise_image_pil, noise_filename)

    return dithered_filename, noise_filename

def overlay_images(img1_path, img2_path, img3_path, output_path):
    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")
    img3 = Image.open(img3_path).convert("RGBA")

    img1 = img1.resize((1024, 1024), Resampling.LANCZOS)
    img2 = img2.resize((1024, 1024), Resampling.LANCZOS)
    img3 = img3.resize((1024, 1024), Resampling.LANCZOS)

    final_image = Image.new('RGBA', img1.size)

    for y in range(img1.size[1]):
        for x in range(img1.size[0]):
            r3, g3, b3, a3 = img3.getpixel((x, y))
            r2, g2, b2, a2 = img2.getpixel((x, y))
            r = int(0.05 * r3 + 0.95 * r2)
            g = int(0.05 * g3 + 0.95 * g2)
            b = int(0.05 * b3 + 0.95 * b2)
            final_image.putpixel((x, y), (r, g, b, 255))

    for y in range(img1.size[1]):
        for x in range(img1.size[0]):
            r1, g1, b1, a1 = img1.getpixel((x, y))
            rf, gf, bf, af = final_image.getpixel((x, y))
            r = int(0.05 * r1 + 0.95 * rf)
            g = int(0.05 * g1 + 0.95 * gf)
            b = int(0.05 * b1 + 0.95 * bf)
            final_image.putpixel((x, y), (r, g, b, 255))

    final_image.save(output_path)

input_img = input("Masukkan Input Image disini: ")
dithered_filename, noise_filename = gpu_color_dithering_and_noise(input_img)

img1_path = noise_filename
img2_path = input_img
img3_path = dithered_filename

output_filename_parts = input_img.split('.')
output_filename = f"{'.'.join(output_filename_parts[:-1])}_output.{output_filename_parts[-1]}"

overlay_images(img1_path, img2_path, img3_path, output_filename)
print(f"Output image saved as: {output_filename}")

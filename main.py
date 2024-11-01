from PIL import Image, ImageEnhance, ImageOps
import os

INPUT_DIR = 'inputs'
TEXTURE_DIR = 'textures'
OUTPUT_DIR = 'outputs'

def load_image(image_path):
    img = Image.open(image_path)
    print(f"Loaded image {image_path} with size: {img.size}")
    return img

def resize_texture(texture, target_size, scale_factor=1.0):
    # Adjust the texture size based on the scale_factor
    new_size = (int(target_size[0] * scale_factor), int(target_size[1] * scale_factor))
    texture_resized = texture.resize(new_size)
    print(f"Resized texture to: {texture_resized.size}")
    return texture_resized

def apply_mask(texture, garment):
    mask = garment.convert('L')
    mask = ImageOps.invert(mask)

    texture.putalpha(mask)

    return texture

def overlay_texture(garment, texture, opacity=0.5):
    texture = resize_texture(texture, garment.size)
    texture = texture.convert("RGBA")
    garment = garment.convert("RGBA")

    # Apply mask for better blending
    texture = apply_mask(texture, garment)
    blend_image = Image.alpha_composite(garment, texture)
    
    return blend_image

def save_final_image(image, output_path="outputs/final_textured_garment.png"):
    # Save the final image in PNG format to preserve quality
    image.save(output_path, format="PNG")
    print(f"Final image saved at {output_path}")


def main():
    garment_image = load_image(os.path.join(INPUT_DIR, 'garment.jpg'))
    texture_image = load_image(os.path.join(TEXTURE_DIR, 'cotton.jpg'))

 # Apply texture overlay
    textured_garment = overlay_texture(garment_image, texture_image, opacity=0.5)

    # Save the blended texture image
    output_path = os.path.join(OUTPUT_DIR, 'textured_garment.png')
    textured_garment.save(output_path, format="PNG")
    print(f"Intermediate textured image saved at {output_path}")

    # Save the final upscaled image for further processing
    save_final_image(textured_garment)

if __name__ == "__main__":
    main()
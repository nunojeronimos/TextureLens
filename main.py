import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Set the model path and device (CPU or CUDA)
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to on-model and close-up images
on_model_image_path = 'inputs/on-model_garment.jpg'
closeup_image_path = 'textures/cotton1.jpg'

# Initialize the model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

# Helper function to process an image with super-resolution
def process_image(image_path, output_name):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output_path = f'outputs/{output_name}_rlt.png'
    cv2.imwrite(output_path, output)
    print(f"Processed and saved: {output_path}")
    return output_path

# Texture transfer function
def apply_texture(on_model_img_path, texture_img_path):
    # Load and upscale the images
    on_model_img = cv2.imread(on_model_img_path)
    texture_img = cv2.imread(texture_img_path)
    
    # Resize texture to match the on-model image
    texture_resized = cv2.resize(texture_img, (on_model_img.shape[1], on_model_img.shape[0]))
    
    # Convert the texture image to grayscale and apply a high-pass filter to extract fine details
    texture_gray = cv2.cvtColor(texture_resized, cv2.COLOR_BGR2GRAY)
    high_pass_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    texture_detail = cv2.filter2D(texture_gray, -1, high_pass_kernel)

    # Convert the detail image back to color for blending
    texture_detail_colored = cv2.cvtColor(texture_detail, cv2.COLOR_GRAY2BGR)

    # Create a mask to blend only the garment area
    hsv_model = cv2.cvtColor(on_model_img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([10, 30, 30])  # Adjust based on garment color
    upper_color = np.array([30, 255, 255])
    garment_mask = cv2.inRange(hsv_model, lower_color, upper_color)

    # Apply morphological operations to refine the garment mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel)

    # Blend the detail into the garment area using the mask
    garment_with_detail = cv2.addWeighted(on_model_img, 1, texture_detail_colored, 0.2, 0)
    result = np.where(garment_mask[:, :, None] == 255, garment_with_detail, on_model_img)

    # Save the final image
    output_path = 'outputs/augmented_on_model_with_details.png'
    cv2.imwrite(output_path, result)
    print(f"Texture detail blended and saved: {output_path}")
# Process both the on-model and close-up images with super-resolution
on_model_sr_path = process_image(on_model_image_path, 'on_model')
closeup_sr_path = process_image(closeup_image_path, 'closeup')

# Apply the texture to the on-model image
apply_texture(on_model_sr_path, closeup_sr_path)

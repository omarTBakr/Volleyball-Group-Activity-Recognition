from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2 # OpenCV is needed for drawing text on images

# NEW: Import TensorBoard and Torchvision utils
from torch.utils.tensorboard import SummaryWriter
import torchvision

# --- Your existing setup code ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from data_loader import collate_fn , VolleyballDataset  # noqa: E402
from configs.path_config import MAIN_DATASET_DIR, TRAIN_VIDEOS_NUMBERS, VALIDATION_VIDEO_NUMBERS, TEST_VIDEOS_NUMBERS  # noqa: E402

 

# =================================================================
# === NEW VISUALIZATION FUNCTIONS ===
# =================================================================

def denormalize(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def annotate_image(image_tensor, text, mean, std):
    """
    Draws text on a denormalized image tensor.
    
    Args:
        image_tensor (torch.Tensor): A single image tensor of shape [C, H, W].
        text (str): The text to draw.
    
    Returns:
        torch.Tensor: The annotated image tensor.
    """
    # 1. Denormalize the image to get correct colors
    img = denormalize(image_tensor.clone(), mean, std)
    
    # 2. Convert from PyTorch format [C, H, W] to OpenCV format [H, W, C] and scale to 0-255
    img = img.permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8)
    
    # 3. OpenCV works with BGR, but our tensor was RGB. We need to copy to avoid a C-contiguous error.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.ascontiguousarray(img)

    # 4. Draw the text
    position = (10, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0) # Green
    thickness = 2
    cv2.putText(img, text, position, font, font_scale, color, thickness)
    
    # 5. Convert back to PyTorch format [C, H, W] and scale to 0-1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
    
    return img_tensor


def visualize_full_frames(writer, data_loader, num_images=10, mean=None, std=None):
    """
    Fetches a batch of full frames, annotates them with group activity,
    and writes them to TensorBoard as a grid.
    """
    print("Visualizing full frames...")
    
    # Get one batch from the loader
    batch = next(iter(data_loader))
    images = batch['image']
    labels = batch['group_activity']

    num_to_show = min(num_images, len(images))
    
    # Annotate each image in the batch with its label
    annotated_images = []
    for i in range(num_to_show):
        annotated_img = annotate_image(images[i], labels[i], mean, std)
        annotated_images.append(annotated_img)
        
    # Create a grid of images
    grid = torchvision.utils.make_grid(annotated_images, nrow=5) # Display in 2 rows of 5
    
    # Write the grid to TensorBoard
    writer.add_image('Visualization/Full_Frames', grid, 0)
    print(f"Logged {num_to_show} full frames to TensorBoard under 'Visualization/Full_Frames'.")


def visualize_cropped_players(writer, data_loader, mean=None, std=None):
    """
    Fetches one sample of cropped players, annotates them with action labels,
    and writes them to TensorBoard as a grid.
    """
    print("\nVisualizing cropped player images...")
    
    # Get one batch (we only need the first item in it)
    batch = next(iter(data_loader))
    
    # Get the 12 cropped images from the FIRST sample in the batch
    # Shape of images_for_one_sample: [12, C, H, W]
    images_for_one_sample = batch['images'][0] 
    
    # Get the 12 action labels for that same sample
    labels_for_one_sample = batch['player_actions'][0]

    # Annotate each of the 12 crops
    annotated_crops = []
    for i in range(len(images_for_one_sample)):
        annotated_crop = annotate_image(images_for_one_sample[i], labels_for_one_sample[i], mean, std)
        annotated_crops.append(annotated_crop)
        
    # Create a grid of the 12 crops
    grid = torchvision.utils.make_grid(annotated_crops, nrow=6) # Display in 2 rows of 6
    
    # Write the grid to TensorBoard
    writer.add_image('Visualization/Cropped_Players', grid, 0)
    print(f"Logged {len(annotated_crops)} cropped player images to TensorBoard under 'Visualization/Cropped_Players'.")


# --- Main Execution and Visualization ---
if __name__ == '__main__':
    
    # --- 1. Setup Dummy Data (same as the test script) ---
    dummy_root = Path('./dummy_volleyball_dataset')
    # ... (full dummy data setup code from the previous answer is required here) ...
    
    # --- 2. Define Transformations and Constants ---
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    # --- 3. Initialize TensorBoard Writer ---
    # This will create a 'runs/dataset_visualization' directory for the logs.
    writer = SummaryWriter('runs/dataset_visualization')
    print("TensorBoard writer initialized. Logs will be saved in the 'runs/dataset_visualization' directory.")

    # --- 4. Run Full Frame Visualization ---
    full_frame_dataset = VolleyballDataset(mode='train', transform=data_transforms, middle=True)
    full_frame_loader = DataLoader(full_frame_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    visualize_full_frames(writer, full_frame_loader, num_images=10, mean=NORM_MEAN, std=NORM_STD)
    
    # --- 5. Run Cropped Player Visualization ---
    crop_dataset = VolleyballDataset(mode='train', transform=data_transforms, crop=True)
    crop_loader = DataLoader(crop_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    visualize_cropped_players(writer, crop_loader, mean=NORM_MEAN, std=NORM_STD)

    # --- 6. Close the writer and clean up ---
    writer.close()
    
    # --- Cleanup the Dummy Directory ---
    import shutil
    # shutil.rmtree(dummy_root) # You might want to comment this out to inspect dummy data
    
    print("\n--- Visualization complete! ---")
    print("To view the output, run the following command in your terminal:")
    print("\ntensorboard --logdir=runs\n")
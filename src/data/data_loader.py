import shutil
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch  

# Assuming BASE_DIR and configs are set up correctly
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from configs.path_config import MAIN_DATASET_DIR, TRAIN_VIDEOS_NUMBERS, VALIDATION_VIDEO_NUMBERS, TEST_VIDEOS_NUMBERS  # noqa: E402

class VolleyballDataset(Dataset):
    """
    Custom PyTorch Dataset for the Volleyball Dataset with extended features.
    
    Loads data based on the structure: <root>/<video_id>/<frame_id>/<frame_id>.jpg
    
    Can load:
    - A single middle frame (with fallback).
    - A temporal window of frames.
    - A fixed number of cropped player images from the middle frame.
    """
    def __init__(self, mode: str, transform=None, middle: bool = True, crop: bool = False, temporal_window: int = 9):
        """
        Args:
            mode (str): Dataset split ('train', 'validation', or 'test').
            transform (callable, optional): Transform for image samples.
            middle (bool): If True, loads only the middle frame. If False, loads a temporal window.
            crop (bool): If True, overrides 'middle' and loads 12 cropped player images.
            temporal_window (int): Total frames in the temporal window (must be odd).
        """
        assert mode in ['train', 'validation', 'test'], "Mode must be 'train', 'validation', or 'test'"
        assert temporal_window % 2 == 1, "Temporal window must be an odd number."
        
        self.mode = mode
        self.transform = transform
        self.middle = middle
        self.crop = crop
        self.temporal_window = temporal_window
        self.num_player_crops = 12
        
        self.video_id_map = {
            'train': TRAIN_VIDEOS_NUMBERS,
            'validation': VALIDATION_VIDEO_NUMBERS,
            'test': TEST_VIDEOS_NUMBERS
        }
        
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Parses all annotation files and creates a master list of samples.
        """
        samples = []
        video_ids = self.video_id_map[self.mode]
        print(f"Loading samples for '{self.mode}' mode...")
        
        for video_id in video_ids:
            video_dir = MAIN_DATASET_DIR / str(video_id)
            annotations_path = video_dir / 'annotations.txt'
            if not annotations_path.exists(): 
                continue
            
            parsed_frames = self._parse_annotations_file(annotations_path)
            
            for frame_data in parsed_frames:
                # The frame_id is the name of the subfolder
                frame_id_str = Path(frame_data['filename']).stem
                frame_dir = video_dir / frame_id_str
                
                # We only need to store the frame_dir and the labels.
                # The exact image paths will be constructed on-the-fly in __getitem__.
                sample = {
                    'frame_dir': frame_dir,
                    'group_activity': frame_data['group_activity'],
                    'players': frame_data['players']
                }
                samples.append(sample)
                    
        print(f"Loaded {len(samples)} samples.")
        return samples

    def _parse_annotations_file(self, filepath: Path):
        # This helper function is correct and remains the same.
        # ... (code omitted for brevity) ...
        parsed_frames = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                frame_filename, group_activity = parts[0], parts[1]
                player_annotations = []
                i = 2
                while i + 4 < len(parts):
                    try:
                        bbox = [int(p) for p in parts[i:i+4]]
                        action = parts[i+4]
                        player_annotations.append({'action': action, 'bbox': bbox})
                    except ValueError: break
                    i += 5
                parsed_frames.append({
                    'filename': frame_filename,
                    'group_activity': group_activity,
                    'players': player_annotations
                })
        return parsed_frames

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Main method to fetch a data item. It delegates to helper methods based on config."""
        sample_info = self.samples[idx]
        
        if self.crop:
            return self._get_cropped_players_item(sample_info)
        elif not self.middle:
            return self._get_temporal_window_item(sample_info)
        else:
            return self._get_middle_frame_item(sample_info)

    def _get_middle_frame_item(self, sample_info):
        """Loads a single middle frame with fallback logic."""
        frame_dir = sample_info['frame_dir']
        
        # Try to find the target frame directory, or the one after it.
        # This loop handles the "fail safe" you requested.
        image = None
        for offset in range(3): # Try the original folder, then the next two
            # The folder name is the frame ID
            current_frame_id = int(frame_dir.name) + offset
            current_frame_dir = frame_dir.parent / str(current_frame_id)
            image_path = current_frame_dir / f"{current_frame_id}.jpg"
            
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                break
        
        if image is None:
            raise FileNotFoundError(f"Could not find valid image for {frame_dir.name} or its successors.")

        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'group_activity': sample_info['group_activity'], 'players': sample_info['players']}

    def _get_temporal_window_item(self, sample_info):
        """Loads a sequence of 9 frames centered on the target frame."""
        frame_dir = sample_info['frame_dir']
        video_dir = frame_dir.parent
        center_frame_id = int(frame_dir.name)
        half_window = self.temporal_window // 2
        
        frame_ids_to_load = range(center_frame_id - half_window, center_frame_id + half_window + 1)
        
        frames = []
        for fid in frame_ids_to_load:
            image_path = video_dir / str(fid) / f"{fid}.jpg"
            frames.append(image_path)
            
        # This simplified logic loads valid frames and repeats the first/last valid ones for missing edges.
        loaded_frames = [Image.open(p).convert('RGB') if p.exists() else None for p in frames]
        
        first_valid_frame = next((f for f in loaded_frames if f is not None), None)
        last_valid_frame = next((f for f in reversed(loaded_frames) if f is not None), None)
        
        if not first_valid_frame:
            raise FileNotFoundError(f"No valid frames found in window for frame ID {center_frame_id}")
            
        final_frames = [(f or first_valid_frame if i <= len(loaded_frames)//2 else last_valid_frame) for i, f in enumerate(loaded_frames)]

        if self.transform:
            final_frames = [self.transform(frame) for frame in final_frames]
            
        images_tensor = torch.stack(final_frames) # Stack into [T, C, H, W]
        
        return {'images': images_tensor, 'group_activity': sample_info['group_activity'], 'players': sample_info['players']}

    def _get_cropped_players_item(self, sample_info):
        """Loads the middle frame and returns 12 cropped and padded player images."""
        frame_dir = sample_info['frame_dir']
        image_path = frame_dir / f"{frame_dir.name}.jpg"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find image for {frame_dir.name}")
            
        image = Image.open(image_path).convert('RGB')
        players = sample_info['players']
        
        cropped_images = []
        player_actions = []

        for player in players:
            x, y, w, h = player['bbox']
            cropped_img = image.crop((x, y, x + w, y + h))
            cropped_images.append(cropped_img)
            player_actions.append(player['action'])

        cropped_images = cropped_images[:self.num_player_crops]
        player_actions = player_actions[:self.num_player_crops]

        if self.transform:
            cropped_images = [self.transform(img) for img in cropped_images]
        
        # Pad if fewer than 12 players
        num_to_pad = self.num_player_crops - len(cropped_images)
        if num_to_pad > 0:
            if not cropped_images: # Handle case with zero players
                 # Create a dummy image to get shape info
                 dummy_tensor = self.transform(Image.new('RGB', (100, 100))) # type: ignore
                 c, h, w = dummy_tensor.shape
            else:
                 c, h, w = cropped_images[0].shape
            
            padding_image = torch.zeros((c, h, w), dtype=torch.float32)
            padding_action = 'padding'
            
            cropped_images.extend([padding_image] * num_to_pad)
            player_actions.extend([padding_action] * num_to_pad)
            
        images_tensor = torch.stack(cropped_images) # Stack into [12, C, H, W]
        
        return {'images': images_tensor, 'group_activity': sample_info['group_activity'], 'player_actions': player_actions}

 
def collate_fn(batch):
  
    # Filter out None samples if any
    batch = [b for b in batch if b is not None]
    if not batch: 
        return None
    
    # Handle different keys for single vs multiple images
    if 'image' in batch[0]:
        images = [item['image'] for item in batch]
        images = torch.stack(images, 0)
    elif 'images' in batch[0]:
        images = [item['images'] for item in batch]
        images = torch.stack(images, 0)
    else:
        raise KeyError("Batch items must contain either 'image' or 'images' key.")

    # Group activities and player annotations are always lists
    group_activities = [item['group_activity'] for item in batch]
    
    # Handle different keys for player annotations
    if 'players' in batch[0]:
        players_annos = [item['players'] for item in batch]
        return {'image' if 'image' in batch[0] else 'images': images, 'group_activity': group_activities, 'players': players_annos}
    elif 'player_actions' in batch[0]:
        player_actions = [item['player_actions'] for item in batch]
        return {'images': images, 'group_activity': group_activities, 'player_actions': player_actions}
# --- Example Usage and Test Suite ---
if __name__ == '__main__':
    
    # --- 1. Setup a Dummy Dataset Structure for Testing ---
    print("--- Setting up a dummy dataset for testing ---")
    
    # Create a temporary root directory for our test data
    dummy_root = Path('./dummy_volleyball_dataset')
    dummy_root.mkdir(exist_ok=True)
    
    # Define a mock configuration that points to this dummy data
    MAIN_DATASET_DIR = dummy_root
    TRAIN_VIDEOS_NUMBERS = [0] # We only need one video for testing
    VALIDATION_VIDEO_NUMBERS = []
    TEST_VIDEOS_NUMBERS = []

    # Create the dummy video directory
    (dummy_root / "0").mkdir(exist_ok=True)
    
    # Create dummy annotation file
    annotations_content = (
        "1005.jpg r-pass 10 10 20 20 waiting 50 50 30 30 standing\n" # Frame with 2 players
        "1010.jpg l-spike 15 15 25 25 spiking\n" # Frame with 1 player
    )
    with open(dummy_root / "0" / "annotations.txt", "w") as f:
        f.write(annotations_content)
        
    # Create dummy image files for a temporal window around frame 1005
    # Window of 9: from 1001 to 1009
    for i in range(1001, 1010):
        frame_dir = dummy_root / "0" / str(i)
        frame_dir.mkdir(exist_ok=True)
        # Create a small blank white image
        Image.new('RGB', (100, 100), color = 'white').save(frame_dir / f"{i}.jpg")
        
    # Create dummy image files for frame 1010
    (dummy_root / "0" / "1010").mkdir(exist_ok=True)
    Image.new('RGB', (100, 100), color = 'white').save(dummy_root / "0" / "1010" / "1010.jpg")
    
    print("Dummy dataset created.\n")
    
    # --- 2. Define Transformations ---
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)), # Use smaller size for faster testing
        transforms.ToTensor(),
    ])

    # --- 3. Run Tests for each mode ---

    # == Test 1: Middle Frame Mode (middle=True) ==
    print("--- Testing Middle Frame Mode (middle=True) ---")
    try:
        middle_frame_dataset = VolleyballDataset(mode='train', transform=data_transforms, middle=True, crop=False)
        middle_loader = DataLoader(middle_frame_dataset, batch_size=2, collate_fn=collate_fn)
        middle_batch = next(iter(middle_loader))
        
        print("Test PASSED.")
        print(f"Batch image shape: {middle_batch['image'].shape}") # Expected: [2, 3, 64, 64]
        assert middle_batch['image'].shape == (2, 3, 64, 64)
        print(f"Group activities: {middle_batch['group_activity']}")
        print(f"Number of players in first sample: {len(middle_batch['players'][0])}")
    except Exception as e:
        print(f"Test FAILED: {e}")

    # == Test 2: Temporal Window Mode (middle=False) ==
    print("\n--- Testing Temporal Window Mode (middle=False) ---")
    try:
        temporal_dataset = VolleyballDataset(mode='train', transform=data_transforms, middle=False, crop=False, temporal_window=9)
        temporal_loader = DataLoader(temporal_dataset, batch_size=2, collate_fn=collate_fn)
        temporal_batch = next(iter(temporal_loader))
        
        print("Test PASSED.")
        print(f"Batch images shape: {temporal_batch['images'].shape}") # Expected: [2, 9, 3, 64, 64]
        assert temporal_batch['images'].shape == (2, 9, 3, 64, 64)
        print(f"Group activities: {temporal_batch['group_activity']}")
    except Exception as e:
        print(f"Test FAILED: {e}")


    # == Test 3: Cropped Players Mode (crop=True) ==
    print("\n--- Testing Cropped Players Mode (crop=True) ---")
    try:
        crop_dataset = VolleyballDataset(mode='train', transform=data_transforms, crop=True)
        crop_loader = DataLoader(crop_dataset, batch_size=2, collate_fn=collate_fn)
        crop_batch = next(iter(crop_loader))
        
        print("Test PASSED.")
        print(f"Batch images shape: {crop_batch['images'].shape}") # Expected: [2, 12, 3, 64, 64]
        assert crop_batch['images'].shape == (2, 12, 3, 64, 64)
        print(f"Group activities: {crop_batch['group_activity']}")
        print(f"Player actions for first sample: {crop_batch['player_actions'][0]}")
        # Check padding for the second sample which only has 1 player
        assert crop_batch['player_actions'][1][1] == 'padding'
        print(f"Player actions for second (padded) sample: {crop_batch['player_actions'][1]}")
    except Exception as e:
        print(f"Test FAILED: {e}")

    # --- 4. Cleanup the Dummy Directory ---
    print("\n--- Cleaning up dummy dataset ---")
    shutil.rmtree(dummy_root)
    print("Cleanup complete.")
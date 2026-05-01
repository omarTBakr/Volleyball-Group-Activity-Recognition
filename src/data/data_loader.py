import shutil
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch  


from configs.path_config import MAIN_DATASET_DIR
from configs.data_split import TRAIN_VIDEOS_NUMBERS, VALIDATION_VIDEO_NUMBERS, TEST_VIDEOS_NUMBERS  # noqa: E402

class VolleyballDataset(Dataset):
    '''
    Custom Dataset class for loading volleyball video frames 
    n_frames: number of frames to sample from each video
    full_image: whether to return the full images (True) or not (False)
    cropped_image: whether to return the cropped person images (True) or the full frames (False)
    
    if n_frames ==1 : then return the middle frame , if does not exist then return the one after it and so one till it finds one
    if n_frames >1 : then return  (n-1)/2 before the middle and (n-1)/2 after the middle , 
    
    '''
    
    def __init__(self , n_frames = 9 ,full_image = True, cropped_image = False) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.full_image = full_image
        self.cropped_image = cropped_image  
        
        if self.cropped_image and self.full_image:
            raise ValueError("cropped_image and full_image cannot both be True. Please choose one.")
        
        if n_frames % 2 == 0 or  n_frames <=0:
            raise ValueError("n_frames must be positive odd number to have a clear middle frame.")




if __name__ == "__main__":
    pass 
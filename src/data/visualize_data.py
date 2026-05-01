from src.data.data_loader import VolleyballDataset
from matplotlib import pyplot as plt
from configs.path_config import VIDEO_SAMPLE1_DIR , VIDEO_SAMPLE2_DIR , VIDEO_SAMPLE3_DIR , VIDEO_SAMPLE4_DIR , VIDEO_SAMPLE5_DIR
from itertools import islice
import cv2 
import os 

videos_dir = [VIDEO_SAMPLE1_DIR , VIDEO_SAMPLE2_DIR , VIDEO_SAMPLE3_DIR , VIDEO_SAMPLE4_DIR , VIDEO_SAMPLE5_DIR]


def visualize_data( n_pictures = 5 , video = False , n_videos = 2):
    '''
        Visualizes a sample of the data set by displaying a grid of images with their corresponding labels.
        Args:
            n_pictures (int): The number of images to display in the grid (default is 8)
            video (bool): Whether to display videos instead of images (default is False)
            n_videos (int): The number of videos to display (default is 2)
    
    Note : this is the hirarchachy of the dataset :
    dataset/
    ├── main_dataset/
    ├── video_samples/
    │   ├── 7/
    │   ├── 10/
    │   ├── Info.txt/
    ├── volleyball-detection/
    |   |-0
    |        |- 3596
    |           |-action_detection.txt [number.jpg  count xmin ymin xmax ymax confindence class ... ]
    |           |-person_detection.txt
    ├── volleyball_tracking_annotation/
        |- 0
            |- 3596
                |- 3596.txt
    
    
    '''
    if video:
        visualize_videos(n_videos)
    else:
        visualize_images(n_pictures)
      

def visualize_videos(n_videos = 2):
    '''
        Visualizes a sample of the videos in the dataset by displaying them.
        Args:
            n_videos (int): The number of videos to display (default is 2)
    '''
    # setup the video writer 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # VideoWriter(filename, fourcc, fps, frameSize)
    out = cv2.VideoWriter('output.mp4', fourcc, 10, (640, 480))

    for directory in videos_dir[:n_videos]:
         
        for frame in reversed(os.listdir(directory)):
            if frame.endswith(".jpg"):
                frame = cv2.imread(directory / frame)
                frame = cv2.resize(frame, (640, 480))
                out.write(frame)
                cv2.imshow('Live Encoding', frame)
        
            # Press 'q' to stop early
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        
    out.release()
    cv2.destroyAllWindows()


 

def visualize_images(n_pictures=5):
    '''
        Visualizes a sample of the images in the dataset by displaying a grid of images with their corresponding labels.
        Args:
            n_pictures (int): The number of images to display in the grid (default is 5) will display 5 * 5 images = 25 images total , 5 from each video_sample directory 
            will choose frames from (2 before the middle  + middle + 2 after middle ) 
    '''

    assert n_pictures % 2 == 1, "n_pictures must be odd"
    grid = []
    
    for directory in videos_dir:
        # We list all jpgs to find the actual count , note this is valid because they are numbered sequentially , so the middle index will be the middle frame
        all_frames = sorted([f for f in directory.glob("*.jpg")], key=lambda x: int(x.stem))
        if not all_frames: continue # if the directory is empty skip it 
        
        mid_idx = len(all_frames) // 2
        offset = n_pictures // 2
        
        video_sequence = []
        # Calculate indices based on the middle
        for i in range(mid_idx - offset, mid_idx + offset + 1):
            # Check bounds to ensure index exists
            if 0 <= i < len(all_frames):
                img = cv2.imread(str(all_frames[i]))
                if img is not None:
                    # BGR to RGB for Matplotlib
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Keep your horizontal flip
                    video_sequence.append(cv2.flip(img, 1))
        
        if video_sequence:
            grid.append(video_sequence)

    # Visualization
    fig, axes = plt.subplots(len(grid), n_pictures, figsize=(15, 2 * len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            axes[i, j].imshow(grid[i][j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

        
        



if __name__ == "__main__":
    # visualize_data(video=True, n_videos=4)
    visualize_data(video=False, n_pictures=5)
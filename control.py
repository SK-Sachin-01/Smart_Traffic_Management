import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import time 
import matplotlib.pyplot as plt

# Load the YOLO model
model_path = "runs/detect/best1.pt"
model = YOLO(model_path)

# Global variables
timer = [0, 0, 0, 0]
start = 0
max_wait_time = 60

def update_timers(timers):
    """
    Update timers by adding elapsed time since the last update.
    
    Parameters:
    - timers: List of timers to update.
    
    Returns:
    - Updated list of timers.
    """
    current_time = time.time()
    elapsed_time = current_time - update_timers.last_update_time if hasattr(update_timers, 'last_update_time') else 0
    update_timers.last_update_time = int(current_time)  # Update last update time for next call

    return [timer + elapsed_time for timer in timers]

def is_image(input):
    try:
        img = cv2.imread(input)
        if img is not None:
            return True
    except:
        pass
    return False

def make_bounding_box(img):
    if is_image(img):
        image = cv2.imread(img)
    else:
        image = img
    results = model(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image_rgb

def count_classes_in_image(image):
    if is_image(image):
        image = cv2.imread(image)
    results = model(image)
    class_ids = [int(box[-1].item()) for result in results for box in result.boxes.data]
    class_counts = Counter(class_ids)
    class_names = {cls_id: model.names[cls_id] for cls_id in class_counts.keys()}
    class_counts_named = {class_names[cls_id]: count for cls_id, count in class_counts.items()}
    return class_counts_named

weights = {
    'ambulance': 5, 'army vehicle': 5, 'auto rickshaw': 2, 'bicycle': 1,
    'bus': 10, 'car': 3, 'garbagevan': 7, 'human hauler': 6, 'minibus': 8,
    'minivan': 4, 'motorbike': 1, 'pickup': 4, 'policecar': 5, 'rickshaw': 2,
    'scooter': 1, 'suv': 4, 'taxi': 3, 'three wheelers -CNG-': 3, 'truck': 9,
    'van': 4, 'wheelbarrow': 1
}

def calculate_green_light_time(weight):
    """
    Calculate the corresponding time for green lights based on traffic weight.

    Parameters:
    - weight: Weight of traffic (integer or float).

    Returns:
    - green_time: Time for green light (float), constrained by max and min values.
    """
    max_green_time = 30  
    min_green_time = 5 

    m = (max_green_time - min_green_time) / 20
    c = min_green_time
    weight_to_min = 10
   
    green_time = m * (weight-weight_to_min) + c
    green_time = max(min(green_time, max_green_time), min_green_time)

    return green_time

def calculate_weight(img):
    class_count = count_classes_in_image(img)
    weighted_traffic = sum(count * weights.get(cls_name, 1) for cls_name, count in class_count.items())
    return weighted_traffic

weight_arr = [0,0,0,0]

def display_videos_together(video_paths, target_width, target_height, frame_skip):
    global weight_arr
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    print("Opened video captures:", caps)
    if not all([cap.isOpened() for cap in caps]):
        print("Error: Unable to open one or more video files")
        for cap in caps:
            cap.release()
        return

    else:
        frames = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((target_height, target_width, 3), np.uint8)
            else:
                frame = cv2.resize(frame, (target_width, target_height))
            frames.append(frame)

        if len(frames) != len(video_paths):
            return

        img = [make_bounding_box(frame) for frame in frames]
        weight_arr = [calculate_weight(frame) for frame in frames]
        
        while len(img) < 4:
            img.append(np.zeros((target_height, target_width, 3), np.uint8))

        top_row = np.hstack((img[0], img[1]))
        bottom_row = np.hstack((img[2], img[3]))
        combined_frame = np.vstack((top_row, bottom_row))

        plt.imshow(combined_frame)
        plt.axis('off')
        plt.show()
        # plt.pause(0.001)  # Display the plot for a brief moment
        # plt.draw()  # Force redraw to make sure it displays

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# Example usage
video_paths = [
    r'video\2103099-uhd_3840_2160_30fps.mp4',
    r'video\5927708-hd_1080_1920_30fps.mp4',
    r'video\8321868-hd_1920_1080_30fps.mp4'
]

def solve(video_paths, target_width, target_height, frame_skip):
    global start, timer
    while True:
        display_videos_together(video_paths, target_width, target_height, start)
        start += frame_skip
        var = update_timers(timer)
        timer = var
        mx = 0
        ind_mx = 0
        print(weight_arr)
        for i in range(len(timer)):
            if timer[i] > mx:
                ind_mx = i
                mx = timer[i]

        if mx > max_wait_time:
            print(weight_arr[ind_mx])
            timer[ind_mx] = 0
            
            time.sleep(calculate_green_light_time(weight_arr[ind_mx]))

        else:
            mx_weight = max(weight_arr)
            print(calculate_green_light_time(mx_weight))
            time.sleep(calculate_green_light_time(mx_weight))
            timer[weight_arr.index(mx_weight)]=0

solve(video_paths, 320, 180, 40)

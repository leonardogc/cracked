'''conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install easyocr
pip uninstall opencv-python-headless
pip install opencv-python
pip install moviepy


conda install -c conda-forge moviepy
conda install -c conda-forge opencv
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 moviepy opencv -c pytorch -c nvidia -c conda-forge


conda install pytorch torchvision torchaudio pytorch-cuda=11.6 opencv -c pytorch -c nvidia -c conda-forge
'''

'''import easyocr
import time
import cv2

reader = easyocr.Reader(['en'])

img = cv2.imread('img2.png')

start = time.time()

for _ in range(100):
    result = reader.readtext(img)

end = time.time()

print(100/(end-start))'''


'''from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


ffmpeg_extract_subclip('teammate_lowlight.mp4', 158, 170, targetname='teammate_lowlight_cropped.mp4')'''

'''
import cv2

cap = cv2.VideoCapture('Valorant 2022.11.09 - 18.42.52.03.DVR.mp4')

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the total numer of frames in the video.
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Calculate the duration of the video in seconds
duration = frame_count / fps

second = 0
cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000) # optional
success, image = cap.read()

while success and second <= duration:

    # do stuff
    cv2.imshow('image', image)

    #Set waitKey 
    cv2.waitKey(1)

    second += 0.5
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, image = cap.read()


cap.release()
cv2.destroyAllWindows()
'''


import easyocr
import cv2
import numpy as np
import time
from utils import to_lower, is_in
from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm


TIME_STEP = 500
CLIP_TIMEOUT = 30000
ME_HIGHLIGHT_BEFORE = 2500
ME_HIGHLIGHT_AFTER = -2000
ME_LOWLIGHT_BEFORE = 4000
ME_LOWLIGHT_AFTER = -1000
SPECTATE_HIGHLIGHT_BEFORE = 2500
SPECTATE_HIGHLIGHT_AFTER = 0
SPECTATE_LOWLIGHT_BEFORE = 4000
SPECTATE_LOWLIGHT_AFTER = 4000
WINDOW_SIZE = (640/1920, 400/1080)
WINDOW_POS = (1270/1920, 85/1080)
KILLFEED_SEPARATOR = 0.82
# SPECTATE_WINDOW_SIZE = (400/1920, 30/1080)
# SPECTATE_WINDOW_POS = (110/1920, 820/1080)
SPECTATE_WINDOW_SIZE = (360/1920, 135/1080)
SPECTATE_WINDOW_POS = (60/1920, 770/1080)


def get_timestamps(video_path, my_names, spectate_names, debug=False):
    # init reader
    reader = easyocr.Reader(['en'])

    # init capture
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the total numer of frames in the video.
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate the duration of the video in milliseconds
    duration = (frame_count * 1000) / fps

    curr_time = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
    success, image = cap.read()

    h, w = image.shape[0:2]
    slice = [int(WINDOW_POS[1] * h), int((WINDOW_POS[1] + WINDOW_SIZE[1]) * h), int(WINDOW_POS[0] * w), int((WINDOW_POS[0] + WINDOW_SIZE[0]) * w)]
    spectate_slice = [int(SPECTATE_WINDOW_POS[1] * h), int((SPECTATE_WINDOW_POS[1] + SPECTATE_WINDOW_SIZE[1]) * h), int(SPECTATE_WINDOW_POS[0] * w), int((SPECTATE_WINDOW_POS[0] + SPECTATE_WINDOW_SIZE[0]) * w)]

    data = {}

    while success:
        killfeed = image[slice[0]:slice[1], slice[2]:slice[3]]
        spectate = image[spectate_slice[0]:spectate_slice[1], spectate_slice[2]:spectate_slice[3]]

        killfeed_result = reader.readtext(killfeed)
        spectate_result = reader.readtext(spectate)

        detected_spectate_names = []

        for bbox, text, prob in spectate_result:
            text = text.lower()

            gt = is_in(text, spectate_names)

            if gt is not None:
                detected_spectate_names.append(gt)


        for bbox, text, prob in killfeed_result:
            # unpack the bounding box
            tl, tr, br, bl = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            text = text.lower()

            gt = is_in(text, my_names)

            if gt is None:
                gt = is_in(text, detected_spectate_names)

            if gt is not None:
                timestamps = data.get(gt, [[], []])

                if br[0] > int(killfeed.shape[1] * KILLFEED_SEPARATOR):
                    # Got killed
                    timestamps[1].append(curr_time)
                else:
                    # Killed someone
                    timestamps[0].append(curr_time)
                
                data[gt] = timestamps

        if debug:
            print('spectate')
            print()

            for bbox, text, prob in spectate_result:
                # unpack the bounding box
                tl, tr, br, bl = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                cv2.rectangle(spectate, tl, br, (0, 255, 0), 2)

                print(f'{text} - {prob} - {(tl, tr, br, bl)}')

            print()

            print('killfeed')
            print()

            for bbox, text, prob in killfeed_result:
                # unpack the bounding box
                tl, tr, br, bl = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))

                cv2.rectangle(killfeed, tl, br, (0, 255, 0), 2)

                print(f'{text} - {prob} - {(tl, tr, br, bl)}')
            
            print()

            print(data)

            print()

            cv2.line(killfeed, (int(killfeed.shape[1] * KILLFEED_SEPARATOR), 0), (int(killfeed.shape[1] * KILLFEED_SEPARATOR), killfeed.shape[0]), (0, 0, 255), 2)

            cv2.imshow('killfeed', killfeed)
            cv2.imshow('spectate', spectate)

            # Set waitKey
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        curr_time += TIME_STEP
        cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
        success, image = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return data, duration

def timestamps_to_clips(timestamps):
    clips = []

    if len(timestamps) == 0:
        return clips

    timestamps = np.array(timestamps)

    diff = np.array([])

    if len(timestamps) >= 2:
        diff = timestamps[1:] - timestamps[:-1]
    
    clip_separators = np.nonzero(diff > CLIP_TIMEOUT)[0]
    clip_separators = np.append(clip_separators, len(timestamps)-1)
    
    for i in range(len(clip_separators)):
        if i == 0:
            clips.append([timestamps[0], timestamps[clip_separators[i]]])
        else:
            clips.append([timestamps[clip_separators[i-1]+1], timestamps[clip_separators[i]]])
    
    return clips


def transform_timestamps(data):
    transformed_data = {}

    for name, timestamps in data.items():
        highlight_clips = timestamps_to_clips(timestamps[0])
        lowlight_clips = timestamps_to_clips(timestamps[1])

        transformed_data[name] = [highlight_clips, lowlight_clips]
    
    return transformed_data

def export_clips(path, export_path, data, duration, my_names):
    original_file_name = Path(path).stem
    original_file_extension = Path(path).suffix

    for name, timestamps in data.items():
        highlight_clips = timestamps[0]
        lowlight_clips = timestamps[1]

        highlight_export_path = Path(export_path) / name / 'highlights'
        lowlight_export_path = Path(export_path) / name / 'lowlights'

        highlight_export_path.mkdir(parents=True, exist_ok=True)
        lowlight_export_path.mkdir(parents=True, exist_ok=True)

        for count, clip in enumerate(highlight_clips):

            if is_in(name, my_names) is not None:
                clip_start = clip[0] - ME_HIGHLIGHT_BEFORE
                clip_end = clip[1] + ME_HIGHLIGHT_AFTER
            else:
                clip_start = clip[0] - SPECTATE_HIGHLIGHT_BEFORE
                clip_end = clip[1] + SPECTATE_HIGHLIGHT_AFTER

            if clip_start < 0:
                clip_start = 0
            
            if clip_end > duration:
                clip_end = duration
            
            clip_start /= 1000
            clip_end /= 1000

            clip_export_path = highlight_export_path / f'{original_file_name}_clip{count}{original_file_extension}'

            ffmpeg_extract_subclip(path, clip_start, clip_end, targetname=str(clip_export_path))
        
        for count, clip in enumerate(lowlight_clips):
            if is_in(name, my_names) is not None:
                clip_start = clip[0] - ME_LOWLIGHT_BEFORE
                clip_end = clip[1] + ME_LOWLIGHT_AFTER
            else:
                clip_start = clip[0] - SPECTATE_LOWLIGHT_BEFORE
                clip_end = clip[1] + SPECTATE_LOWLIGHT_AFTER

            if clip_start < 0:
                clip_start = 0
            
            if clip_end > duration:
                clip_end = duration
            
            clip_start /= 1000
            clip_end /= 1000

            clip_export_path = lowlight_export_path / f'{original_file_name}_clip{count}{original_file_extension}'

            ffmpeg_extract_subclip(path, clip_start, clip_end, targetname=str(clip_export_path))


def main():
    my_names = ['me', 'notabadbronzie', 'leogc1801']
    spectate_names = ['electricyttrium', 'toli', 'ros', 'foowalksintoabar', 'foowalksintoavar', 'foowalksintoawar', 'foowalksintoacar']

    file_paths = list(Path('').glob('*.mp4'))
    export_path = 'exported'

    my_names = to_lower(my_names)
    spectate_names = to_lower(spectate_names)

    for path in tqdm(file_paths):

        path = str(path)
        
        data, duration = get_timestamps(path, my_names, spectate_names)

        data = transform_timestamps(data)

        export_clips(path, export_path, data, duration, my_names)


if __name__ == '__main__':
    main()


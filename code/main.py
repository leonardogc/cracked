'''
python 3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
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

if __name__ == '__main__':

    reader = easyocr.Reader(['en'])

    img = cv2.imread('img2.png')
    imgs = [img]*4

    start = time.time()

    for i in range(100):
        # result = reader.readtext(img, batch_size=32)
        result = reader.readtext_batched(imgs, batch_size=32)

        print(result[0])
        exit()

        print(i)

    end = time.time()

    print(100/(end-start))

    exit()

    # 9.8 fps batch_size >= 16'''

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
import os
import queue
import multiprocessing


TIME_STEP = 500
CLIP_TIMEOUT = 30000
ME_HIGHLIGHT_BEFORE = 2500
ME_HIGHLIGHT_AFTER = -1000
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
EASY_OCR_BATCH_SIZE = 32
FRAME_QUEUE_SIZE = 16


def skip_ms(cap, time_step):
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time = (cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000) / fps

    while True:
        success = cap.grab()

        curr_time = (cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000) / fps

        if curr_time - start_time >= time_step or not success:
            break
    
    if not success:
        return success, None, None

    success, image = cap.retrieve()
    
    return success, image, curr_time

def put_non_block(queue_obj, item):
    while True:
        try:
            queue_obj.put(item, timeout=0.5)
            return
        except queue.Full:
            pass

def get_non_block(queue_obj):
    while True:
        try:
            item = queue_obj.get(timeout=0.5)
            return item
        except queue.Empty:
            pass


def frame_extractor(frame_queue, video_path):
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
    slice = (int(WINDOW_POS[1] * h), int((WINDOW_POS[1] + WINDOW_SIZE[1]) * h), int(WINDOW_POS[0] * w), int((WINDOW_POS[0] + WINDOW_SIZE[0]) * w))
    spectate_slice = (int(SPECTATE_WINDOW_POS[1] * h), int((SPECTATE_WINDOW_POS[1] + SPECTATE_WINDOW_SIZE[1]) * h), int(SPECTATE_WINDOW_POS[0] * w), int((SPECTATE_WINDOW_POS[0] + SPECTATE_WINDOW_SIZE[0]) * w))

    while success:
        killfeed = image[slice[0]:slice[1], slice[2]:slice[3]]
        spectate = image[spectate_slice[0]:spectate_slice[1], spectate_slice[2]:spectate_slice[3]]

        put_non_block(frame_queue, (killfeed, spectate, curr_time))

        success, image, curr_time = skip_ms(cap, TIME_STEP)
    
    put_non_block(frame_queue, (None, None, duration))

    cap.release()


def get_timestamps(video_path, my_names, spectate_names, debug=False):
    # init reader
    reader = easyocr.Reader(['en'])

    frame_queue = multiprocessing.Queue(maxsize=FRAME_QUEUE_SIZE)

    frame_extractor_process = multiprocessing.Process(target=frame_extractor, args=(frame_queue, video_path), daemon=True)
    frame_extractor_process.start()

    killfeed, spectate, curr_time = get_non_block(frame_queue)

    data = {}

    while killfeed is not None and spectate is not None:

        killfeed_result = reader.readtext(killfeed, batch_size=EASY_OCR_BATCH_SIZE)
        spectate_result = reader.readtext(spectate, batch_size=EASY_OCR_BATCH_SIZE)

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
                timestamps = data.get(gt, ([], []))

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
            '''k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break'''
            
            cv2.waitKey(1)

        killfeed, spectate, curr_time = get_non_block(frame_queue)

    cv2.destroyAllWindows()

    frame_extractor_process.join()

    return data, curr_time

def timestamps_to_clips(timestamps):
    clips = []

    if len(timestamps) == 0:
        return clips

    timestamps = np.array(timestamps)

    diff = np.array([])

    if len(timestamps) >= 2:
        diff = timestamps[1:] - timestamps[:-1]

    # clip_separators saves the index of the timestamp where each clip ends
    
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

        transformed_data[name] = (highlight_clips, lowlight_clips)
    
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


def remove_already_exported(file_paths, export_path):
    already_exported = []
    not_exported = []

    for path, subdirs, files in os.walk(export_path):
        for name in files:
            already_exported.append(name.split('_clip')[0])
    
    already_exported = list(set(already_exported))

    for path in file_paths:
        path = str(path)
        name = Path(path).stem

        if '_ignore' not in name and name not in already_exported:
            not_exported.append(path)
    
    return not_exported


def main():
    my_names = ['me', 'notabadbronzie', 'leogc1801', 'gabe itch']
    spectate_names = ['electricyttrium', 'toli', 'ros', 'foowalksintoabar', 'foowalksintoavar', 'foowalksintoawar', 'foowalksintoacar', 'youmad10']

    my_names = to_lower(my_names)
    spectate_names = to_lower(spectate_names)

    # file_paths can be a list with Path objects or str objects
    file_paths = list(Path('D:\\Clips\\Valorant').glob('*.mp4'))
    export_path = 'D:\\exported'

    file_paths = remove_already_exported(file_paths, export_path)

    no_clips = []

    for path in tqdm(file_paths):
        
        data, duration = get_timestamps(path, my_names, spectate_names)

        data = transform_timestamps(data)

        export_clips(path, export_path, data, duration, my_names)

        if len(data) == 0:
            no_clips.append(path)
    
    if len(no_clips) > 0:
        print('\nWe detected no clips in the following videos:\n')

        for path in no_clips:
            print(path)


if __name__ == '__main__':
    main()

    '''my_names = ['me', 'notabadbronzie', 'leogc1801', 'gabe itch']
    spectate_names = ['electricyttrium', 'toli', 'ros', 'foowalksintoabar', 'foowalksintoavar', 'foowalksintoawar', 'foowalksintoacar', 'youmad10']

    my_names = to_lower(my_names)
    spectate_names = to_lower(spectate_names)

    start = time.time()

    data, duration = get_timestamps('me_highlight.mp4', my_names, spectate_names, debug=True)

    end = time.time()

    print(end - start)'''


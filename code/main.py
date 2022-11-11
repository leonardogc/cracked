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
from difflib import SequenceMatcher
import time

def to_lower(x):
    y = []
    for word in x:
        y.append(word.lower())
    
    return y

def is_in(word, word_list, thre=0.8):
    for x in word_list:
        if SequenceMatcher(a=x, b=word).ratio() > thre:
            return x
    
    return None


TIME_STEP = 500
WINDOW_SIZE = (640/1920, 400/1080)
WINDOW_POS = (1270/1920, 85/1080)
KILLFEED_SEPARATOR = 0.82
# SPECTATE_WINDOW_SIZE = (400/1920, 30/1080)
# SPECTATE_WINDOW_POS = (110/1920, 820/1080)
SPECTATE_WINDOW_SIZE = (360/1920, 135/1080)
SPECTATE_WINDOW_POS = (60/1920, 770/1080)


my_names = ['me', 'notabadbronzie', 'leogc1801']
spectate_names = ['electricyttrium', 'toli', 'ros', 'foowalksintoabar', 'foowalksintoavar', 'foowalksintoawar', 'foowalksintoacar']

my_names = to_lower(my_names)
spectate_names = to_lower(spectate_names)

def get_timestamps(video_path, start_time=0, debug=True):
    # init reader
    reader = easyocr.Reader(['en'])

    # init capture
    cap = cv2.VideoCapture(video_path)

    curr_time = start_time
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
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                break

        curr_time += TIME_STEP
        cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
        success, image = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    return data

start = time.time()
data = get_timestamps('me_highlight.mp4', debug=True)
end = time.time()

print(data)
print(end-start)

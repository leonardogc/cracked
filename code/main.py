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

ffmpeg_extract_subclip('Valorant 2022.11.09 - 18.42.52.03.DVR.mp4', 1, 4, targetname='test.mp4')'''

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

TIME_STEP = 500
WINDOW_SIZE = (640/1920, 400/1080)
WINDOW_POS = (1277/1920, 88/1080)
CONF_THRES = 0.75


def get_timestamps(video_path):
    # init reader
    reader = easyocr.Reader(['en'])

    # init capture
    cap = cv2.VideoCapture(video_path)

    curr_time = 0
    cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
    success, image = cap.read()

    while success:
        h, w = image.shape[0:2]

        image = image[int(WINDOW_POS[1] * h):int((WINDOW_POS[1] + WINDOW_SIZE[1]) * h), int(WINDOW_POS[0] * w):int((WINDOW_POS[0] + WINDOW_SIZE[0]) * w)]

        result = reader.readtext(image)
        
        for bbox, text, prob in result:
            if prob < CONF_THRES:
                continue

            # unpack the bounding box
            tl, tr, br, bl = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            cv2.rectangle(image, tl, br, (0, 255, 0), 2)

            print(f'{text} - {prob} - {(tl, tr, br, bl)}')

        print()

        # do stuff
        cv2.imshow('image', image)

        #Set waitKey
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break

        curr_time += TIME_STEP
        cap.set(cv2.CAP_PROP_POS_MSEC, curr_time)
        success, image = cap.read()

    cap.release()
    cv2.destroyAllWindows()


get_timestamps('Valorant 2022.11.09 - 18.42.52.03.DVR.mp4')



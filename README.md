# Color, Skin, and Finger Detection in Images

| <center>Source Image</center>              | <center>Skin Detection</center>                 |
| ------------------------------------------ | ----------------------------------------------- |
| ![alt text](media/img/person_pointing.jpg) | ![alt text](media/img/person_pointing_skin.jpg) |

_Note: Skin Detection with HSV color space filtering might not work for all skin tones_

## Install requirements

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Webcam fingertip detection

```bash
# runs finger detection on the webcam, Press z when fingers are visible in the 9 squares
python skin_and_finger_detection/video_fingertip_detection.py
```

### Webcam skin detection based on YCrCb Color Space

```bash
# displays the original webcam screen and the skin detection result screen
python skin_and_finger_detection/video_skin_detection.py
```

### Choose a HSV color space filter threshold on an image

```bash
# move sliders to find the optimal threshold for detecting different colors in image
python color_detection/image_hsv_threshold_picker.py -i <IMAGE_PATH>
```

### Filter colors in video using HSV color space filters

```bash
python color_detection/video_color_detection.py -v <VIDEO_PATH>
```

## Reference

-   Fingertip Detection: A Fast Method with Natural Hand <https://arxiv.org/pdf/1212.0134.pdf>

-   Finger Detection and Tracking using OpenCV and Python <https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m>

# Fast Finger Detection in Images

## Install requirements

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Webcam fingertip detection

```bash
$ python skin_and_finger_detection/video_fingertip_detection.py
```

### Webcam skin detection based on YCrCb Color Space

```bash
$ python skin_and_finger_detection/video_skin_detection.py
```

### Choose a HSV color space filter threshold on an image

```bash
$ python color_detection/image_hsv_threshold_picker.py -i <IMAGE_PATH>
```

### Filter colors in video using HSV color space filters

```bash
$ python color_detection/video_color_detection.py -v <VIDEO_PATH>
```

## Reference

-   Fingertip Detection: A Fast Method with Natural Hand <https://arxiv.org/pdf/1212.0134.pdf>

-   Finger Detection and Tracking using OpenCV and Python <https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m>

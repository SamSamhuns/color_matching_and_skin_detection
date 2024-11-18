import numpy as np
import argparse
import cv2


def detect_and_display_skin(vid_src):
    cap = cv2.VideoCapture(vid_src)

    # define the upper and lower boundaries of the YCrCb pixels
    # intensities to be considered 'skin'
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Get pointer to video frames from primary device
        YCrCb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        skinMask = cv2.inRange(YCrCb_converted, min_YCrCb, max_YCrCb)
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # set all non-black pixels to white
        skin[np.where((skin != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # Display the resulting frame
        cv2.imshow('orig', frame)
        cv2.imshow('detected', skin)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--video_path',
                        type=str,
                        help="""Video path where skin detection is done.
                                If no path is provided, cv2 tries to use webcam""")
    args = parser.parse_args()
    vid_src = args.video_path if args.video_path is not None else 0
    detect_and_display_skin(vid_src)


# Running the Program
if __name__ == "__main__":
    main()

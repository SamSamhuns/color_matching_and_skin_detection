import numpy as np
import argparse
import cv2


def detect_color_in_hsv(video_src, min_HSV, max_HSV):
    """
    params:
        video_src: path to video or 0 for webcam
        min_HSV: Lower boundary of HSV color space
        max_HSV: Upper boundary of HSV color space
    """
    video_src = 0 if video_src is None else video_src
    cap = cv2.VideoCapture(video_src)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # # Get pointer to video frames from primary device
        HSV_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(HSV_converted, min_HSV, max_HSV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        color_mask = cv2.erode(color_mask, kernel, iterations=2)
        color_mask = cv2.dilate(color_mask, kernel, iterations=2)
        color_mask = cv2.GaussianBlur(color_mask, (3, 3), 0)
        det_color = cv2.bitwise_and(frame, frame, mask=color_mask)

        # set all non-black pixels to white
        det_color[np.where((det_color != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # Display the resulting frame
        cv2.imshow("detected", det_color)
        cv2.imshow("orig", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        help="""Video path where color matching is done in HSV space.
                                If no path is not provided, cv2 tries to use webcam""",
    )
    args = parser.parse_args()

    # black shorts
    min_HSV = np.array([0, 0, 0], np.uint8)
    max_HSV = np.array([168, 220, 46], np.uint8)

    # red shorts
    min_HSV = np.array([158, 82, 20], np.uint8)
    max_HSV = np.array([179, 243, 200], np.uint8)
    detect_color_in_hsv(args.video_path, min_HSV, max_HSV)


# Running the Program
if __name__ == "__main__":
    main()

import numpy as np
import cv2


def main():
    cap = cv2.VideoCapture(0)

    # define the upper and lower boundaries of the YCrCb pixels
    # intensities to be considered 'skin'
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)
    while(True):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # # Get pointer to video frames from primary device
        YCrCb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        skinMask = cv2.inRange(YCrCb_converted, min_YCrCb, max_YCrCb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # set all non-black pixels to white
        skin[np.where((skin != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # Display the resulting frame
        cv2.imshow('frame', skin)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Running the Program
if __name__ == "__main__":
    main()

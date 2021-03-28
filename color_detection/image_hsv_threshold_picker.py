import cv2
import argparse
import numpy as np


def nothing(x):
    pass


def hsv_threshold_picker(img_path, wait_time=33, mask_val=255):
    """
    params:
        img_src: path to image
        wait_time: cv2.waitKey time, set 30 for Osx
        mask_val: bit value of masked bits, 0=black, 255=white
    """
    image = cv2.imread(img_path)
    output = image

    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('Hue Min', 'image', 0, 179, nothing)
    cv2.createTrackbar('Sat Min', 'image', 0, 255, nothing)
    cv2.createTrackbar('Val Min', 'image', 0, 255, nothing)

    cv2.createTrackbar('Hue Max', 'image', 0, 179, nothing)
    cv2.createTrackbar('Sat Max', 'image', 0, 255, nothing)
    cv2.createTrackbar('Val Max', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('Hue Max', 'image', 179)
    cv2.setTrackbarPos('Sat Max', 'image', 255)
    cv2.setTrackbarPos('Val Max', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('Hue Min', 'image')
        sMin = cv2.getTrackbarPos('Sat Min', 'image')
        vMin = cv2.getTrackbarPos('Val Min', 'image')

        hMax = cv2.getTrackbarPos('Hue Max', 'image')
        sMax = cv2.getTrackbarPos('Sat Max', 'image')
        vMax = cv2.getTrackbarPos('Val Max', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        output[output == 0] = mask_val

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
            print("Min H,S,V = (%d, %d, %d), Max H,S,V = (%d, %d, %d)" % (
                hMin, sMin, vMin, hMax, sMax, vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image_path',
                        type=str,
                        required=True,
                        help="""Image path where color matching is done in HSV space.""")
    args = parser.parse_args()
    hsv_threshold_picker(args.image_path)


if __name__ == "__main__":
    main()

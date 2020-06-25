import numpy as np
import cv2


def main():
    cap = cv2.VideoCapture(0)

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    min_hsv = np.array([0, 48, 80], dtype="uint8")
    max_hsv = np.array([20, 255, 255], dtype="uint8")

    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Conversion operations on frame
        # rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # hsv_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #
        # skinMask = cv2.inRange(hsv_converted, min_hsv, max_hsv)
        #
        # # apply a series of erosions and dilations to the mask
        # # using an elliptical kernel
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # skinMask = cv2.erode(skinMask, kernel, iterations=2)
        # skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        # # # blur the mask to help remove noise, then apply the mask to the frame
        # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        # skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # cv2.imshow('frame', skin)
        # show the skin in the image along with the mask
        # cv2.imshow("images", np.hstack([frame, skin]))

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


        # Setup SimpleBlobDetector parameters.
        # params = cv2.SimpleBlobDetector_Params()
        #
        # # Change thresholds
        # params.minThreshold = 10
        # params.maxThreshold = 200
        #
        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 1500
        #
        # # Filter by Circularity
        # params.filterByCircularity = True
        # params.minCircularity = 0.1

        # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87
        #
        # # Filter by Inertia
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        # detector = cv2.SimpleBlobDetector()

        # Detect blobs.
        # keypoints = detector.detect(skin)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        # im_with_keypoints = cv2.drawKeypoints(skin, keypoints, np.array(
        #     []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #
        # # Show blobs
        # cv2.imshow("Keypoints", im_with_keypoints)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Running the Program
if __name__ == "__main__":
    main()

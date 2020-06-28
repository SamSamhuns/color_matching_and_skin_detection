# Histogram based approach to separate hand from background frame
# Thresholding and filtering used to cancel background
# HSV / YCrCb based color filtering is inaccurate when skin tones and lighting conditions change

import cv2
import time
import numpy as np

hand_hist = None
traverse_point = []
total_rectangle = 9

hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=100, hpercent=100):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def draw_rect(frame):
    """
    Draw 9 rectangles on the frame to state the region
    where the user's hand should be overlayed and return frame

    total_rectangle = 9 (number of rects)
    """
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create a region of interest filter to capture hsv vals in 9 rect area
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                                    hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking_improved(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (34, 34))
    disc = np.float32(disc)  # this disc is for ignoring noise
    disc /= np.count_nonzero(disc) / 2  # normalize filter by size
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 4, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None)
    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def hist_masking(frame, hist):
    """
    Returns the frame masked with the histogram
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    # apply filtering and thresholding to smoothen image
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_trailing_circles(frame, traverse_point, color=[0, 255, 255]):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(
                5 - (5 * i * 3) / 100), color, -1)


def manage_image_opr(frame, hand_hist, improved_method=True):
    if improved_method:
        hist_mask_image = hist_masking_improved(frame, hand_hist)
    else:
        hist_mask_image = hist_masking(frame, hand_hist)
        hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
        hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

    contour_list = contours(hist_mask_image)
    if len(contour_list) == 0:
        return
    max_cont = max(contour_list, key=cv2.contourArea)

    # function to draw contours around skin/hand
    cv2.drawContours(frame, [max_cont], -1, 0xFFFFFF, thickness=4)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)

        print("Centroid : " + str(cnt_centroid) +
              ", farthest Point (Finger Tip) : " + str(far_point))
        cv2.circle(frame, far_point, 6, [0, 0, 255], -1)

        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_trailing_circles(frame, traverse_point)


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)
        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

        # for OpenCV major version < 3, manual calculation of frame rate for video feed might be required
        fps = capture.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

    cv2.destroyAllWindows()
    capture.release()


def calc_framerate(video_capture):
    """
    Manually calculate FPS
    Returns (seconds, fps)
        seconds: total time take to render 120 frames
        fps: frames per second
    """
    num_frames = 120
    print(f"Capturing {num_frames} frames")

    start = time.time()
    for i in xrange(0, num_frames):
        video_capture.read()
    end = time.time()

    seconds = end - start
    fps = num_frames / seconds

    return seconds, fps


if __name__ == '__main__':
    main()

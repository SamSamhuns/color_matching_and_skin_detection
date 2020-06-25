def rgb_to_hsv(rgb):
    """
    rgb: Tuple (R, G, B) Values in the range 0 to 255
    returns hsv as a tuple (H, S, V) H and S are in percents
    """
    R, G, B = rgb
    R, G, B = R / 255, G / 255, B / 255

    Cmax, Cmin = max(R, G, B), min(R, G, B)
    delta = Cmax - Cmin

    if delta == 0:
        H = 0
    elif Cmax == R:
        H = ((G - B) / delta) % 6
    elif Cmax == G:
        H = ((B - R) / delta) + 2
    elif Cmax == B:
        H = ((R - G) / delta) + 4
    H *= 60

    if Cmax == 0:
        S = 0
    else:
        S = delta / Cmax

    V = Cmax
    return (H, S, V)


def rgb_to_YCbCr(rgb):
    """
    rgb: Tuple (R, G, B) Values in the range 0 to 255
    returns YCbCr as a tuple (Y, Cb, Cr) luminance and chrominance
    """
    R, G, B = rgb
    Y = 0.299 * R + 0.287 * G + 0.11 * B
    Cb = B - Y
    Cr = R - Y
    return (Y, Cb, Cr)


if __name__ == "__main__":
    print(rgb_to_hsv((102, 23, 23)))
    print(rgb_to_YCbCr((102, 23, 23)))

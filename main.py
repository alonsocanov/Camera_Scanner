import numpy as np
import argparse
import sys
import image as im
import pytesseract


def main():
    # obtain input arguments
    parser = argparse.ArgumentParser(description='Lane detection script')
    parser.add_argument('--image', type=str, help='Image path to detect lines')
    args = parser.parse_args()
    # if image argument has input
    key_q = False
    win_name = 'Frame'
    while not key_q:
        if args.image:
            file = im.availableFiles(args.image)
            img, corners = im.isImage(file)
        # if no image argument video feed is used
        else:
            img, corners = im.isWebcam(win_name)
        h, w = img.shape[:2]
        # arange corner points
        corners = im.arangePts(corners, (h, w))
        # ratio between height and width of found document
        ratio = im.scannedRatio(corners)
        # destination point for warpping
        dst = im.setDestinationPts((h / ratio, h))
        # warp image
        warp = im.warpImg(img, corners, dst, (h / ratio, h))
        # show image
        im.show(warp, win_name)
        im.check_any_key()

    im.destroyAllWindows()


if __name__ == '__main__':
    main()

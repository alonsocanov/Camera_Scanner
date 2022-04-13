import argparse
import image as im


def main():
    # obtain input arguments
    parser = argparse.ArgumentParser(description='Lane detection script')
    parser.add_argument('--image', type=str, help='Image path to detect lines')
    args = parser.parse_args()
    # if image argument has input
    key_q = False
    win_name = 'Frame'
    path = 'warped.jpg'
    while not key_q:
        if args.image:
            file = im.availableFiles(args.image)
            img, corners = im.isImage(file)
        # if no image argument video feed is used
        else:
            img, corners = im.isWebcam(win_name)
        h, w = img.shape[:2]
        # arange corner points
        corners = im.arangePts(corners, (w, h))
        # ratio between height and width of found document
        ratio = im.scannedRatio(corners)
        # destination point for warpping
        dst, dim = im.setDestinationPts(ratio, (w, h))
        # warp image
        warp = im.warpImg(img, corners, dst, dim)

        warp = im.textDetection(warp)
        # im.saveImg(warp, path)
        # show image
        im.show(warp, win_name)

        im.check_any_key()

    im.destroyAllWindows()


if __name__ == '__main__':
    main()

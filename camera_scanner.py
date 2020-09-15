import cv2
import numpy as np
import os
import argparse
import sys
import glob
import imutils


def resize(img: np.ndarray, rescale_factor: float=None):
    h_max = 600
    h, w = img.shape[:2]
    if not rescale_factor and h_max:
        rescale_factor = h_max / h
    dimesion = (int(w * rescale_factor), int(h * rescale_factor))
    resized_img = cv2.resize(img, dimesion)
    return resized_img, 1 / rescale_factor


def show(img: np.ndarray):
    h_max = 600
    win_name = 'Image'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)
    img_show, _ = resize(img)
    cv2.imshow(win_name, img_show)
    # cv2.imwrite('example_2.jpeg', img_show)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)


def availableFiles(path: str):
    full_path = glob.glob(path, recursive=True)
    if full_path:
        file = full_path[0]
    else:
        file = None
    return file


def checkWebcamAvalability(webcam: cv2.VideoCapture):
    # check if webcam is already opened
    if not webcam.isOpened():
        sys.exit("Error opening webcam")


def check(c: str='q'):
    # check if desired key was pressed
    if cv2.waitKey(1) & 0xFF == ord(c):
        return True
    return False


def arangePts(pts: np.ndarray, dim: tuple):
    # arange source point by closeners to corners
    h, w = dim
    img_pts = [[0, 0], [w, 0], [w, h], [0, w]]
    img_pts = np.array(img_pts)
    sorted_idx = list()
    for pos in img_pts:
        norm = np.linalg.norm(pts - pos, axis=1)
        sorted_idx.append(np.argmin(norm))
    return pts[sorted_idx].astype(np.float32)


def scannedRatio(pts: np.ndarray):
    # norm of height and width of found corners
    w_norm = np.linalg.norm(pts[1, :] - pts[0, :], axis=0)
    h_norm = np.linalg.norm(pts[1, :] - pts[2, :], axis=0)
    # ratio between two norms
    ratio = h_norm / w_norm
    return ratio


def isImage(file):
    # read image
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # resize image and obtain its resized factor
    img_res, factor = resize(img)
    h_res, w_res = img_res.shape[:2]
    # convert to gray scale
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    # blur image with Gaussian Filter to reduce image intensity changes
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # find edges with Canny edge detector
    edges = cv2.Canny(blur, threshold1=50, threshold2=200, apertureSize=3)
    # dilatate edges to make them more aparent
    dilate = cv2.dilate(edges, (5, 5), iterations=1)
    # find contours, sort them and take the 5 biggest
    cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # if found contours
    if cnts:
        # initialized source detected points
        src_pts = None
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if contour with 4 sides (for documents)
            if len(approx) == 4:
                src_pts = approx
                break
        if isinstance(src_pts, np.ndarray):
            # draw contour in resized image and in original image
            cv2.drawContours(img_res, [src_pts], -1, (0, 255, 0), 5)
            cv2.drawContours(img, [(src_pts * factor).astype(int)], -1, (0, 255, 0), 5)
    # squeeze array if it has a third dimension
    if len(src_pts.shape) == 3 and src_pts.shape[1] == 1:
        src_pts = np.squeeze(src_pts, axis=1)
    # factor source point to actual image scale
    src_pts = src_pts * factor
    return img, src_pts


def isWebcam():
    # open video capture
    video = cv2.VideoCapture(0)
    # check if webcam isnt being used
    checkWebcamAvalability(video)
    # setup video feed at pixel x = 20, y = 20
    win_name = 'WebCam'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)
    # obtain first frame
    ret, img = video.read()
    h, w = img.shape[:2]
    # resise image and the resize factor
    img_res, factor = resize(img)
    h_res, w_res = img_res.shape[:2]
    # save video
    # out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (w_res, h_res))

    # maximum corners when no document is detected
    max_h = int(h_res * .98)
    max_w = int(w_res * .8)
    min_h = h_res - max_h
    min_w = w_res - max_w
    outer = [[min_w, min_h], [max_w, min_h], [max_w, max_h], [min_w, max_h]]
    default_contour = np.array(outer)
    # area of the maximum corners
    area_img = cv2.contourArea(default_contour)
    # initialize the output image
    out_img = None
    # number of frames that the image has to be stable in order to take frame
    NUM_STABLE_FRAMES = 20
    # counter for number of frames
    stable = 0
    # start video feed analysis
    while video.isOpened():
        # obtain image frame
        ret, img = video.read()
        # resize frame
        img_res, _ = resize(img)
        # convert to gray scale
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        # blur image with gaussian filter for less intensity changes
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # find edges with Canny edge detector
        edges = cv2.Canny(blur, threshold1=50, threshold2=200, apertureSize=3)
        # dilatate edges to make them more aparent
        dilate = cv2.dilate(edges, (5, 5), iterations=1)
        # find contours, sort them and take the 5 biggest
        cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        # if found contours
        if cnts:
            # initialized source detected points
            src_pts = None
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if contour with 4 sides (for documents)
                if len(approx) == 4:
                    src_pts = approx
                    break
            # if a square shape was detected
            if isinstance(src_pts, np.ndarray):
                # area of square shape found
                area = cv2.contourArea(src_pts)
                # if area is at least 80 % of maxumum corners
                if area > area_img * .8:
                    # set green contour
                    color = (0, 255, 0)
                    # if the stable counter has been stable for NUM_STABLE, take that frame for analisis
                    if stable == NUM_STABLE_FRAMES:
                        out_img = img
                        break
                    # add to stable counter
                    stable += 1
                # if area is smaller tah 80% of maximum corners
                else:
                    # set orange contour
                    color = (0, 165, 255)
                    stable = 0
            # if no square shaped was d
            else:
                color = (0, 0, 255)
                src_pts = [[min_w, min_h], [max_w, min_h], [max_w, max_h], [min_w, max_h]]
                src_pts = np.array(src_pts)
                stable = 0
            # draw contours depending in the color
            cv2.drawContours(img_res, [src_pts], -1, color, 5)
        # save video feed
        # out.write(img_res)
        # show current frame
        cv2.imshow(win_name, img_res)
        # check if key q was pressed to quit
        if check('q'):
            if isinstance(out_img, np.ndarray):
                break
            else:
                error = 'No image large enough'
                sys.exit(error)
    # close camera feed and all windows
    video.release()
    cv2.destroyAllWindows()
    # squeeze array if it has a third dimension
    if len(src_pts.shape) == 3 and src_pts.shape[1] == 1:
        src_pts = np.squeeze(src_pts, axis=1)
    # factor source point to actual image scale
    src_pts = src_pts * factor
    return out_img, src_pts


def main():
    # obtain input arguments
    parser = argparse.ArgumentParser(description='Lane detection script')
    parser.add_argument('--image', type=str, help='Image path to detect lines')
    args = parser.parse_args()
    # if image argument has input
    if args.image:
        file = availableFiles(args.image)
        if not file:
            error = ' '.join(['Could not find file:', args.path])
            sys.exit(error)
        img, corners = isImage(file)
    # if no image argument video feed is used
    else:
        img, corners = isWebcam()

    h, w = img.shape[:2]
    # arange corner points
    corners = arangePts(corners, (h, w))
    # ratio between height and width of found document
    ratio = scannedRatio(corners)
    # destination point for warpping
    dst = [[0, 0], [h / ratio, 0], [h / ratio, h], [0, h]]
    dst = np.array(dst, dtype=np.float32)
    # matrix transform
    M = cv2.getPerspectiveTransform(corners, dst)
    # warped image from actual size
    warp = cv2.warpPerspective(img, M, (int(h / ratio), h))
    # concatente images to see input and output images
    img_2 = np.concatenate([img, warp], axis=1)
    # show images
    show(img_2)


if __name__ == '__main__':
    main()

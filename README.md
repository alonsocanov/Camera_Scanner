# Camera Scanner

This repository is a classic document scanner using a provided image or a webcam.  

## Run code

First activate the virtual environment if you use one, make sure that you have required libraries and run one of the the folowing commands:  

1) Using the webcam
```bash
python3 path/to/directory/camera_scanner.py
```

2) Using a provided image
```bash
python3 path/to/directory/camera_scanner.py --image path/to/image/img.jpg
```
If you downloaded full repossitory and you are currently  repository run:  
```bash
python3 camera_scanner.py --image data/img_1.jpeg
```

## Required libraries
- cv2
- numpy
- sys
- os
- glob
- imutils
- argparse

## References
Took inspiration from this [post](https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/).


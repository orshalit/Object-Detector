How to install:
From the directory of the project-

$ python3 -m virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

Optional:
If not all of the images inside the image's folder are in jpg format, run:

$ python jpegConversion

How to run:

For single image:
$ python yolo.py --image images/1.jpg --yolo yolo-coco

For folder:
$ python yolo.py --folder images --yolo yolo-coco


Remove # from lines 163/164 to pop image with bbox's on the original image
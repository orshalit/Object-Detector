from __future__ import print_function
import os
from PIL import Image
import glob

# covert all images that are not jpg format to jpg
def convert_to_png():
    print('convert all images from jpeg to png and all channels to RGB')   #iterate over dir, change to png format
    for file in glob.glob("images/*"):
        # print('convert: ',file)
        if file.endswith(".jpg"):
            continue
        else:
            print('convert: ', file)
            f, e = os.path.splitext(file)
            outfile = f + ".jpg"
            try:
                Image.open(file).save(outfile)
                os.remove(file)
            except IOError:
                print("cannot convert", file)

    print('end of conversion')

convert_to_png()

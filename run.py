import pandas as pd
import os
from img_prep import crop_face, add_watermark

img_dir = './data/15Dec/'
new_images = [i for i in os.listdir(img_dir)]
print(len(new_images))

new_images_path = [img_dir+i for i in new_images]

cropped_images_path = ["./data/cropped_imgages/"+n.split('.')[0]+"_cropped.png" for n in new_images]

wm_images_path = [img_dir+"wm_images/"+n for n in new_images]

watermark_image_path = "./findsuri_Watermark3.png"
for img,cimg, wmimg in zip(new_images_path, cropped_images_path, wm_images_path):
#     print(img)
    try:
        crop_face(img, cimg)

        add_watermark(cimg, 
                      watermark_image_path, 
                      wmimg, 
                      position="center", 
                      opacity=0.1)
    except:
        with open('faulty_face_detection.csv', 'a') as f:
            f.write(img+'\n')
        continue

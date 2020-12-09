import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


## import a series of .png images which shown after each other result in the wanted animation
png_dir = 'C:/Users/Florian/Box Sync/Projects/UTe2/CT scan/animation'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))


## this is when you want to change the background color
for im in images:
    for i in im:
        for j in i:
            if j[2] > 200:
                j[0] = 255
                j[1] = 255
                j[2] = 255




## save .gif file
imageio.mimsave('C:/Users/Florian/Box Sync/Projects/UTe2/CT scan/animation2.gif', images, fps=55)
import matplotlib
matplotlib.use("TkAgg")
from PIL import Image
import matplotlib.pylab as pylab


path = "image.jpg"

img = Image.open(open(path))

size_list = [16,20,24,28,32,36,40,48,64]
img_list = []

for size in size_list:
    img_list.append(img.resize((size,size), Image.ANTIALIAS))

img.show()

pylab.subplot(1,9,1)
pylab.imshow(img)

for image in img_list:
    pylab.subplot(1, 9, img_list.index(image)+1)
    pylab.axis('off')
    pylab.imshow(image)
pylab.show()

for image in img_list:
    image.show()

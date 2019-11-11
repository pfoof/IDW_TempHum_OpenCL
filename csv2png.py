from PIL import Image
import sys

tresh = 30.0

img = []

ff = sys.argv[1]

csv = open(ff, "r")
lines = csv.readlines()
for l in lines:
    imgl = [x for x in l.split(",")]
    imgl = imgl[:-1]
    imgl = [float(x) for x in imgl]
    img.append( imgl )
csv.close()

im = Image.new( 'RGB', (len(img[0]), len(img)), "black")
pixels = im.load()
for i in range(im.size[0]):
    for j in range(im.size[1]):
        t = img[j][i]
        if t < 0:
            t = max(-tresh,t)
            r = 0
            g = 255
            b = int(255*(1.0-(t/-tresh)))
        else:
            t = min(t,tresh)
            r = int(255* (t/tresh))
            g = int(255*(1.0-(t/tresh)))
            b = 0
        pixels[i,j] = (r,g,b)

im.save('out.png')



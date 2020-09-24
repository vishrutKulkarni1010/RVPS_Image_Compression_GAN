import cv2
import matplotlib.pyplot as plt
import glob
for name in glob.glob('train/*'):
  img = cv2.imread(name, cv2.IMREAD_COLOR)
  print('Original size',img.shape)
  height = 512
  width = 1024
  dim = (width, height)
  res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
  # Checcking the size
  print("RESIZED", res.shape)
  if cv2.imwrite(name[7:],res) :
    print("saved "+name[7:])

from hashlib import new
from PIL import Image
def letterbox_image(image, size):
    # 对图片进行resize，使图片不失真。在空缺的地方进行padding
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

img = Image.open("./data/images/2.jpg")
new_image = letterbox_image(img,[640,640])
new_image.show()
new_image.save('./2.jpg')

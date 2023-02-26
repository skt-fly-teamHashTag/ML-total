import cv2
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def poo_make_image(message, img):
    # font setting
    year = "2023"
    font = ImageFont.truetype('font/thum1.otf', size=70)
    background_color = 'rgb(0,0,0)'
    txt_color = 'rgb(255,212,0)'
    year_color = 'rgb(229,244,250)'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    image_poo = Image.open('img/poo.png')
    image_poo2 = Image.open('img/poo2.png')

    poo_w,poo_h = image_poo.size
    poo_w2,poo_h2 = image_poo2.size
    image_poo = image_poo.resize((int(poo_w * 0.3), int(poo_h * 0.3)))
    image_poo2 = image_poo2.resize((int(poo_w2 * 0.08), int(poo_h2 * 0.08)))

    draw = ImageDraw.Draw(image)


    # Text wraper to handle long text
    # 40자를 넘어갈 경우 여러 줄로 나눔
    lines = textwrap.wrap(message, width=40)
    lines2 = textwrap.wrap(year, width=40)
  
    # text size and img size
    img_w,img_h = image.size
    txt_w,_ = draw.textsize(message, font=font)
    year_w,_ = draw.textsize(year, font=font)

    # center anchor
    x_text = int(img_w / 2 - (txt_w + year_w) / 2 + 3)
    y_text = int(img_h / 5 * 4 + 20)

    image.paste(image_poo, (-10, int(img_h * 0.63) + 10), image_poo)
    image.paste(image_poo2, (int(img_w * 0.63) + 90, int(img_h * 0.6) + 10), image_poo2)

    # 각 줄의 내용을 적음
    for line in lines:
        draw.text((x_text, y_text), line, font=font, fill=(background_color))
        rx_text, ry_text = x_text - 3, y_text - 3
        draw.text((rx_text, ry_text), line, font=font, fill=(txt_color))

    for line in lines2:
        draw.text((x_text + txt_w, y_text), line, font=font, fill=(background_color))
        rx_text, ry_text = x_text -3 + txt_w, y_text - 3
        draw.text((rx_text, ry_text), line, font=font, fill=(year_color))
    
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def partition_make_image(message,img):
    # font setting
    font = ImageFont.truetype('font/thum2.ttf', size=70)
    background_color = 'rgb(0,0,0)'
    txt_color = 'rgb(255,255,255)'
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    draw = ImageDraw.Draw(image)

    # Text wraper to handle long text
   # 40자를 넘어갈 경우 여러 줄로 나눔
    lines = textwrap.wrap(message, width=40)
  
    # text size and img size
    img_w,img_h = image.size
    txt_w,_ = draw.textsize(message, font=font)

    # center anchor
    x_text = int(img_w / 2 - (txt_w) / 2 + 168)
    y_text = int(img_h / 5 * 4) - 20

    # 각 줄의 내용을 적음
    for line in lines:
        draw.text((x_text, y_text), line, font=font, fill=(background_color))
        rx_text, ry_text = x_text - 2, y_text - 2
        draw.text((rx_text, ry_text), line, font=font, fill=(txt_color))

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def love_make_image(message, img):
    # font setting
    background_color = 'rgb(0,0,0)'
    font = ImageFont.truetype('font/thum3.ttf', size=80)
    txt_color = 'rgb(255,255,255)'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    frame = Image.open('img/frame.png')
    flower = Image.open('img/flower.png')
    flower2 = Image.open('img/flower2.png')

    draw = ImageDraw.Draw(image)

    # Text wraper to handle long text
    # 40자를 넘어갈 경우 여러 줄로 나눔
    lines = textwrap.wrap(message, width=40)
    
    # text size and img size
    img_w, img_h = image.size
    txt_w, txt_h = draw.textsize(message, font=font)
    f_w, f_h = flower.size
    f_w2, f_h2 = flower2.size

    flower = flower.resize((int(f_w * 0.1), int(f_h * 0.1)))
    flower2 = flower2.resize((int(f_w2 * 0.06), int(f_h2 * 0.06)))
    frame = frame.resize((800, 450))

    # center anchor
    x_text = (img_w - txt_w) // 2
    y_text = (img_h - txt_h) // 2 + 80

    image.paste(frame, (0, 0), frame)
    image.paste(flower2, (x_text + txt_w - 40, y_text - txt_h // 2), flower2)
    image.paste(flower, (x_text - 60, y_text - txt_h // 2 + 30), flower)

    # 각 줄의 내용을 적음
    for line in lines:
        draw.text((x_text, y_text), line, font=font, fill=(background_color))
        rx_text, ry_text = x_text - 2,y_text - 2
        draw.text((rx_text, ry_text), line, font=font, fill=(txt_color))

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

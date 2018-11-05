import argparse
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'
FONT = ImageFont.truetype(FONT_PATH, 13, encoding='unic')

def convert(text, size=128, margin=4, save_path=None):
    char_width, char_height = FONT.getsize('\u2588') # full block

    linewidth = (size - margin) // char_width
    lineheight = (size - margin) // char_height

    lines = textwrap.wrap(text, width=linewidth)

    # img = Image.new('RGB', (char_width*linewidth + 2*margin, char_height*lineheight + 2*margin), 'white')
    img = Image.new('RGB', (size, size), 'white')

    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        if i == lineheight:
            break
        draw.text((margin, margin + char_height*i), line, 'black', FONT)

    if save_path is not None:
        img.save(save_path)

    return img

def convert_all(texts, directory):
    os.makedirs(directory, exist_ok=True)
    for i, text in enumerate(texts):
        convert(text, save_path=os.path.join(directory, '{}.png'.format(i)))

if __name__ == '__main__':
    convert('testing 1 2 3 hello how about you this is really fun let me see', save_path='img.png')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('infile', type=argparse.FileType('r'))
    # parser.add_argument('--dir', type=str, default='data')
    # args = parser.parse_args()

    # texts = args.infile.read().strip().split('\n')
    # convert_all(texts, args.dir)


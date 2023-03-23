import argparse
import PIL

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data", help="Path to image data directory")
    parser.add_argument("--xml_dir", type=str, default="data", help="Path to xml data directory")
    args = parser.parse_args()

def conver_png_to_jpg(image):
    image = image.convert('RGB')
    return image
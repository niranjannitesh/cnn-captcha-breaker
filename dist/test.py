import model
import img
import sys
import os

c_model = model.init_model()

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print('Usage: python3 test.py folder')
    sys.exit()

for filename in os.listdir(path):
    captcha = img.load_captcha_file(f'{path}/{filename}')
    char_array = img.cut_captcha(captcha)
    text = model.get_captcha_from_array(c_model, char_array)
    print(text)
    os.rename(f'{path}/{filename}', f'{path}/{text}.png')

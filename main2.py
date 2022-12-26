import keras
import sys, os
import numpy as np
import math
from PIL import Image
from keras.models import load_model
import streamlit as st
import cv2


imsize = (64, 64)
testpic     = "dog1.jpeg"
keras_param = "model/cnn20.h5"

def load_image(path):
        img = Image.open(path)
        img = img.convert('RGB')
        # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
        img = img.resize(imsize)
        # 画像データをnumpy配列の形式に変更
        img = np.asarray(img)
        img = img / 255.0
        return img

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def main():
    st.title("犬猫判別器")
    model = load_model(keras_param)
    # upload Image2
    upload_image = st.file_uploader("画像をアップしよう！", type="jpg")
    if upload_image:
        image = Image.open(upload_image)
        img_array = np.array(image)
        st.image(img_array, width=100)
        img = pil2cv(image)
        img = cv2.resize(img, (64, 64))

        # predict
        prd = model.predict(np.array([img]))
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            st.write(math.floor(prd[0][0]*100), "%の確率で犬です！！")
        elif prelabel == 1:
            st.write(math.floor(prd[0][1]*100), "%の確率で猫です！！")
    


if __name__ == '__main__':
    main()
import os
import time
import numpy as np
import cv2
from ImageProcessor import *

def ClearFrame(frame):
    # renk paletini siyah beyaza çeviriyoruz
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # otsu eşitleme
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Eşik değerin altındaki pikselleri beyaza çeviriyoruz
    binary_image = np.zeros_like(gray_image)
    binary_image[gray_image <= thresh] = 255

    # dilation
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel, iterations=5)

    return cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

def SliceAndProcess(im, images, slices):
    height, width = im.shape[:2]
    slice_height = int(height / slices)

    for i in range(slices):
        part = slice_height * i
        crop_img = im[part:part + slice_height, 0:width]
        images[i].image = crop_img
        images[i].process()

def RepackImages(images):
    return np.concatenate([image.image for image in images], axis=0)

# tüm işlenmiş karelerin kaydedileceği dizi
frames = []
# ekranı N parçaya bölmek için sabit değişken
N_SLICES = 4
# bölünmüş kareleri barındıracak dizi
Images = []
for q in range(N_SLICES):
    Images.append(ImageProcessor())

testVideoPath = "lineFollower.mp4"

def main():
    cap = cv2.VideoCapture("../video/" + testVideoPath)

    if not cap.isOpened():
        print("Unable to open video!")
        exit()

    process_times = []
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # örnek videoumuz istediğimizin tam tersi renklerde olduğu için renkleri ters çeviriyoruz
        frame = 255 - frame

        start = time.process_time()
        img = ClearFrame(frame)
        if img is not None:
            SliceAndProcess(img, Images, N_SLICES)
            direction = 0
            for k in range(N_SLICES):
                direction += Images[k].direction

            fm = RepackImages(Images)
            frames.append(fm)

            # mevcut karede geçen süreyi ms olarak kaydediyoruz
            process_times.append((time.process_time() - start) * 1000)
            cv2.line(
                fm,
                (width // 2 + direction // 3, height),
                (width // 2 + direction // 3, 0),
                (0, 0, 255),
                2
            )
            cv2.imshow("Computer Vision", fm)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    print("Average process time per frame: " + str(sum(process_times) / len(process_times)) + "ms")
    cv2.destroyAllWindows()

    if not os.path.exists("../result/"):
        os.mkdir("../result/")

    print("Writing processed frames to result/" + testVideoPath.rpartition(".")[0] + "_CV2.mp4")
    out = cv2.VideoWriter(
        "../result/" + testVideoPath.rpartition(".")[0] + "_CV2.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        40,
        (width, height)
    )
    for frame in frames:
        out.write(frame)
    out.release()

if __name__=="__main__":
    main()
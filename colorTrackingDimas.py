# Nama  : Dimas Ahmat Husin
# NIM   : C.431.21.0023
# Kelas : Teknik Elektro A Sore

# Tugas Aplikasi Kecerdasan Buatan pelacakan warna

import numpy as np
import cv2

#membuka video gambar
cap = cv2.VideoCapture(0)

while(1):
    #membaca frame gambar
    _, img = cap.read()

    #mengkonversi frame gambar ke HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #mendefinisikan jangkauan warna merah (HSV)
    red_lower = np.array([157,87,111], np.uint8)
    red_upper = np.array([180,255,255], np.uint8)

    #mendefinisikan jangkauan warna hijau (HSV)
    green_lower = np.array([40,95,110], np.uint8)
    green_upper = np.array([82,255,255], np.uint8)

    #mendefinisikan jangkauan warna biru (HSV)
    blue_lower = np.array([99,115,140], np.uint8)
    blue_upper = np.array([130,255,255], np.uint8)

    #mendefinisikan jangkauan warna kuning (HSV)
    yellow_lower = np.array([24,60,200], np.uint8)
    yellow_upper = np.array([60,255,255], np.uint8)

    #mendefinisikan jangkauan warna oranye (HSV)
    orange_lower = np.array([8,130,170], np.uint8)
    orange_upper = np.array([23,255,255], np.uint8)

    #mendefinisikan jangkauan warna ungu (HSV)
    purple_lower = np.array([131,100,140], np.uint8)
    purple_upper = np.array([156,255,255], np.uint8)

    #mencari jangkauan warna di dalam frame gambar
    red = cv2.inRange(hsv, red_lower, red_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    orange = cv2.inRange(hsv, orange_lower, orange_upper)
    purple = cv2.inRange(hsv, purple_lower, purple_upper)

    #mengurangi noise pada gambar (morphological transformation, dilation)
    kernal = np.ones((5,5),"uint8")

    red = cv2.dilate(red,kernal)
    res = cv2.bitwise_and(img, img, mask=red)

    green = cv2.dilate(green,kernal)
    res = cv2.bitwise_and(img, img, mask=green)

    blue = cv2.dilate(blue,kernal)
    res = cv2.bitwise_and(img, img, mask=blue)

    yellow = cv2.dilate(yellow,kernal)
    res = cv2.bitwise_and(img, img, mask=yellow)

    orange = cv2.dilate(orange,kernal)
    res = cv2.bitwise_and(img, img, mask=orange)

    purple = cv2.dilate(purple,kernal)
    res = cv2.bitwise_and(img, img, mask=purple)

    #melacak warna merah
    (contours, hierarcy) = cv2.findContours(
        red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #menggambar kotak dengan warna merah
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "Merah", (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 0, 255))

    #melacak warna hijau
    (contours, hierarcy) = cv2.findContours(
        green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #menggambar kotak dengan warna hijau
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Hijau", (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 255, 0))

    #melacak warna biru
    (contours, hierarcy) = cv2.findContours(
        blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #menggambar kotak dengan warna biru
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, "Biru", (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (255, 0, 0))

    #melacak warna kuning
    (contours, hierarcy) = cv2.findContours(
        yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #menggambar kotak dengan warna kuning
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, "Kuning", (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (0, 255, 255))

    #melacak warna oranye
    (contours, hierarcy) = cv2.findContours(
        orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #menggambar kotak dengan warna oranye
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (3, 152, 252), 2)
            cv2.putText(img, "Oranye", (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (3, 152, 252))

    #melacak warna ungu
    (contours, hierarcy) = cv2.findContours(
        purple, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #menggambar kotak dengan warna ungu
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (252, 3, 198), 2)
            cv2.putText(img, "Ungu", (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (151, 3, 198))

    #Menampilkan gambar
    cv2.imshow("Deteksi Warna ---- Tekan tombol Q untuk mengakhiri", img)

    k = cv2.waitKey(10)
    #tekan tombol Q untuk keluar dari aplikasi
    if k & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


import cv2
from matplotlib import pyplot as plt

def process(i):
    # Read image
    img = cv2.imread(f'image{i}.jpg')

    # To Gray Scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title(f'image - {i} - Grayscale')
    cv2.imwrite(f'image{i}-gray.jpg', img)
    plt.figure()
    plt.hist(img.flatten(),256,(0,256))
    plt.title(f'image - {i} - Histogram')

    # Histogram Equalisation
    hist_eq = cv2.equalizeHist(img)
    plt.figure()
    plt.imshow(hist_eq, cmap='gray')
    plt.axis("off")
    plt.title(f'image{i} - Histogram Equalized')
    cv2.imwrite(f'image{i}-he.jpg', hist_eq)
    plt.figure()
    plt.hist(hist_eq.flatten(),256,(0,256))
    plt.title(f'image - {i} - Equlaized Histogram')

    # Adaptive Histogram Equalisation
    adap_hist_eq = cv2.createCLAHE(clipLimit=5).apply(img)
    plt.figure()
    plt.imshow(adap_hist_eq, cmap='gray')
    cv2.imwrite(f'image{i}-ahe.jpg', adap_hist_eq)
    plt.axis("off")
    plt.title(f'image - Histogram Adaptive Equalized')
    plt.figure()
    plt.hist(adap_hist_eq.flatten(),256,(0,256))
    plt.title(f'image - {i} - Adaptive Equalized Histogram')
    return (hist_eq, adap_hist_eq)


def thresh(eq_img, eq_thresh, adap_eq_img, adap_eq_thresh, i):

    # Thresholding the Equlaized Image
    _, hist_eq_thresh = cv2.threshold(eq_img, eq_thresh, 
        255, cv2.THRESH_BINARY)

    plt.figure()
    plt.imshow(hist_eq_thresh, cmap='gray')
    plt.axis("off")
    plt.title(f'image - Histogram Equalized Thresholded')
    cv2.imwrite(f'image{i}-he_th.jpg', hist_eq_thresh)

    # Thresholding the Adaptive Equlaized Image
    _, adap_hist_eq_thresh = cv2.threshold(adap_eq_img, adap_eq_thresh, 
        255, cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(adap_hist_eq_thresh, cmap='gray')
    plt.axis("off")
    plt.title(f'image - Adaptive Histogram Equalized Thresholded')
    cv2.imwrite(f'image{i}-ahe_th.jpg', adap_hist_eq_thresh)

# Dim Illumination:
eq_img_1, adap_eq_img_1 = process(1)
thresh(eq_img_1, 60, adap_eq_img_1, 50, 1)

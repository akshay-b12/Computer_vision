import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
#from pylab import *
import cv2
import random

def optical_flow(I1g, I2g, window_size=2, tau=1e-2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1 = np.array(I1g)
    I2 = np.array(I2g)
    S = np.shape(I1)
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + \
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            A = [ [ np.sum(Ix*Ix),np.sum(Ix*Iy)], [np.sum(Iy*Ix), np.sum(Iy*Iy)] ]
            A = np.array(A)
    
            b = [-np.sum(Ix*It), -np.sum(Iy*It)]
            b = np.array(b)
            
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            U,D,V_T = np.linalg.svd(A.T.dot(A))
            if np.min(D) < tau:
                u[i,j]=0
                v[i,j]=0                
            
            else:
                try:
                    nu = np.linalg.inv(A).dot(b)
                    u[i,j]=nu[0]
                    v[i,j]=nu[1]
                except Exception as e:
                    print(A,D,e)

    return np.dstack((u, v))
    
def draw_flow(img, img_gray, flow):
    x = np.arange(0, img_gray.shape[1], 1)
    y = np.arange(0, img_gray.shape[0], 1)
    x, y = np.meshgrid(x, y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    #plt.subplots(1, 2, 1)
    #fig = plt.imshow(img_gray, cmap='gray', interpolation='bicubic')
    ax1.imshow(img_gray, cmap='gray', interpolation='bicubic')
    ax1.axis('off')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    step = img_gray.shape[0] / 20
    step = int(step)
    ax1.quiver(x[::step, ::step], y[::step, ::step],flow[::step, ::step, 0], flow[::step, ::step, 1],
               color='r', pivot='middle', headwidth=1, headlength=5)
    
    hsv = np.zeros_like(img)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #print(np.shape(ang), np.shape(hsv))
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    ax2.axis("off")
    #ax1.subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    #ax1.show()
    prvs = next
    #plt.axis('off')
    #plt.show()
    
def draw_hsv(flow, prev, change=True):
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    if change:
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return hsv

def track_moving_objects(frame, prev):
    def gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flow = optical_flow(gray(prev), gray(frame), 24)
    hsv1 = draw_hsv(flow, prev)
    gray1 = cv2.cvtColor(hsv1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray1, 25, 0xFF, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 15 and h > 15 and w < 900 and h < 680:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0xFF, 0), 4)
    return frame


cap = cv2.VideoCapture("vtest.avi")
ret, frame1 = cap.read()
count = -1
out = cv2.VideoWriter('Obj_detect.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
while True:
    ret, frame2 = cap.read()
    count += 1
    if count % 20 != 0:
        continue
    if ret is False:
        break
    objected_frame = track_moving_objects(frame2, frame1)
    out.write(frame2)
    '''
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    '''
    prvs = next
cap.release()
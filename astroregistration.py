
#%%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
from scipy.ndimage import *

from sfunc import *

path = '/Users/victorstrijbis/Documents/Astronomie/processing4/'
A = sorted(glob.glob(path + 'IMG*'))
im1 = plt.imread(A[0]).astype(np.short)

#%%

def eq_image(im):
    b, g, r = cv2.split(im.astype(np.uint8))
    # Apply histogram equalization to each channel
    b_eq = cv2.equalizeHist(b).astype(np.uint8)
    g_eq = cv2.equalizeHist(g).astype(np.uint8)
    r_eq = cv2.equalizeHist(r).astype(np.uint8)

    equalized_arr = np.array(cv2.merge([b_eq, g_eq, r_eq]))

    return equalized_arr

def rough_register(im1,im2,steps=200):
    im1_norm = norm_image(im1)
    im2_norm = norm_image(im2)

    im1_eq = eq_image(im1_norm)
    im2_eq = eq_image(im2_norm)

    im1_bool = np.median((im1>np.mean(im1)),axis=-1); im2_bool = np.median((im2>np.mean(im2)),axis=-1)
    im1_bool = im1; im2_bool = im2

    max_shift = 1000; steps = steps
    shift_list = np.arange(-max_shift,max_shift+1,max_shift//steps)
    dsx = []; dy = 0

    for dx in shift_list:
        alt = np.roll(im2_eq, shift=(dy, dx), axis=(0, 1))

        ref = im1_eq[:,:,1]>250
        alt = alt[:,:,1]>250

        dsx.append(np.sum(np.sum(ref,0)*np.sum(alt,0)))

    dsx = gaussian_filter1d(dsx, sigma=1)

    dxmin = shift_list[np.where(dsx == np.max(dsx))[0][0]]
    plt.figure()
    plt.plot(shift_list,dsx)


    max_shift = 800; steps = int(0.8*steps)
    shift_list = np.arange(-max_shift,max_shift+1,max_shift//steps)
    dx = dxmin; dsy = []

    for dy in shift_list:
        alt = np.roll(im2_eq, shift=(dy, dx), axis=(0, 1))

        ref = im1_eq[:,:,1]>250
        alt = alt[:,:,1]>250

        dsy.append(np.sum(np.sum(ref,1)*np.sum(alt,1)))

    dsy = gaussian_filter1d(dsy, sigma=1)

    plt.figure()
    plt.plot(shift_list,dsy)

    dymin = shift_list[np.where(dsy == np.max(dsy))[0][0]]
    return dymin, dxmin

def precise_register(dymin,dxmin,im1,im2,max_shift=10):

    max_shift_x = max_shift; max_shift_y = max_shift

    best_mse = float('inf')
    for dy in np.arange(-max_shift_y,max_shift_y+1,1):
        print([best_mse,dy])
        for dx in np.arange(-max_shift_x,max_shift_x+1,1):

            alt = np.roll(im2, shift=(dymin+dy, dxmin+dx), axis=(0, 1))
            mse = calculate_mse(im1[max_shift_x:-max_shift_x,max_shift_y:-max_shift_y,:],alt[max_shift_x:-max_shift_x,max_shift_y:-max_shift_y,:])

            if mse < best_mse:

                print(mse)
                best_mse = mse

                ty = dy+dymin
                tx = dx+dxmin

    return ty, tx

#%%

txs = []; tys = [];
for ii in np.arange(1,len(A)):
    print("Registring image {}".format(ii))
    im2 = plt.imread(A[ii]).astype(np.short)
    im2 = np.roll(im2, shift=(0, 0), axis=(0, 1))

    plt.figure()
    plt.imshow(im1)

    plt.figure()
    plt.imshow(im2)

    plt.figure()
    plt.imshow((im1-im2)**2)

    dymin, dxmin = rough_register(im1,im2, 400)
    ty, tx = precise_register(dymin, dxmin, im1, im2, 20)

    txs.append(tx); tys.append(ty)

#%%

images = np.zeros((im1.shape[0],im1.shape[1],im1.shape[2],len(A))).astype(np.short)
images[:,:,:,0] = im1.astype(np.short)

for ii in np.arange(1,len(A)):
    im2 = plt.imread(A[ii]).astype(np.short)
    alt = np.roll(im2, shift=(tys[ii-1], txs[ii-1]), axis=(0, 1))
    images[:,:,:,ii] = alt.astype(np.short)

    plt.figure()
    plt.imshow(alt)
    plt.title(natural_string(ii))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path + 'registered_' + natural_string(ii) + '.png',dpi=300)

#%%

# imfin = norm_image(images[:,:,:,0],1)
# for ii in np.arange(1,len(A)):
#     imfin = imfin * norm_image(images[:,:,:,ii],1)
#%%

ii = 0

plt.figure(figsize=(30,20))

plt.subplot(1, 2, 1)
plt.title("Original Image {}".format(ii))
plt.imshow(images[:,:,:,ii])
plt.axis('off')

im_final = norm_image(np.median(images,axis=-1))

plt.subplot(1, 2, 2)
plt.title("Stacked image")
plt.imshow(im_final)
plt.axis('off')

plt.tight_layout()

plt.savefig(path + 'comparison_med.png',dpi=300)

cv2.imwrite(path + 'output_med.png', im_final)


plt.figure(figsize=(30,20))

plt.subplot(1, 2, 1)
plt.title("Original Image {}".format(ii))
plt.imshow(images[:,:,:,ii])
plt.axis('off')

im_final = norm_image(np.mean(images,axis=-1))

kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
im = cv2.filter2D(im_final, -1, kernel)

cv2.imwrite(path + 'output1_med.png', im)

plt.subplot(1, 2, 2)
plt.title("Stacked image")
plt.imshow(im_final)
plt.axis('off')

plt.tight_layout()

plt.savefig(path + 'comparison_mean.png',dpi=300)

cv2.imwrite(path + 'output_mean.png', im_final)

kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
im = cv2.filter2D(im_final, -1, kernel)

cv2.imwrite(path + 'output1_mean.png', im)

# %%

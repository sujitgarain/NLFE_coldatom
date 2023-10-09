import numpy as np                  # for numeric operation
import scipy.interpolate as spi     # for interpolating various callibrations
import numpy.fft as fft             # for fast fourier transform (alternativelity scipy.fft also works)
import scipy.ndimage as ndi         # for image spatial filters
import scipy.optimize as spo        # for curve fitting
import matplotlib.pyplot as plt     # making plots
import matplotlib.gridspec as gsp   # to customize image layouts
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sif_parser as sf             # read sif files
import pandas as pd                 # read and write dataframes with csv files
import glob                         # search directory for existing files with filters
import warnings                     # to raise warnings
import copy                         # to have copy and deepcopy
import cv2 as cv                    # open computer vision functions for feature matching
import ray                          # for parallel computatation
import time                         # for bechmarking 
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, fftfreq     # for fourier transform and filtering


# Global parameters
NUM_FLUOROSCENT_IMG = 5     # Number of fluoroscence images
NUM_ATOM_IMG = 1     # Number of images with atoms
NUM_PROBE_IMG= 10    # Number of images with probe but without atoms (Total images captured with probe is nAi+nPi)
NUM_BACKGROUND_IMG = 1     # Number of background images captured
NUM_TOTAL_IMG = NUM_FLUOROSCENT_IMG  + (NUM_ATOM_IMG  + NUM_PROBE_IMG) + NUM_BACKGROUND_IMG       # Total number of images in a sif file

def get_sifdata(filePath):
    data, info = sf.np_open(filePath)
    return data

def get_sif(folderName, fileName, cur, rep):
    filePath = folderName + "\\" + fileName + r"{:3.3f}".format(cur) + "_" + str(rep) + ".sif"
    data, info = sf.np_open(filePath)
    return data

def get_images(sifData):
    """
    Needs the sif data
    Gives images as separate list
    """
    fImgs = sifData[: NUM_FLUOROSCENT_IMG]  # list of fluoroscent images
    aImgs = sifData[NUM_FLUOROSCENT_IMG : NUM_FLUOROSCENT_IMG + NUM_ATOM_IMG]   # the image with atom SHOULD BE only ONE IMAGE
    pImgs = sifData[NUM_FLUOROSCENT_IMG + NUM_ATOM_IMG : NUM_FLUOROSCENT_IMG + NUM_ATOM_IMG + NUM_PROBE_IMG]     # images with probe without atom
    bImgs = sifData[NUM_FLUOROSCENT_IMG + NUM_ATOM_IMG + NUM_PROBE_IMG :]       # list of bakground images 
    
    if len(fImgs)  + len(aImgs) + len(pImgs) + len(bImgs) != NUM_TOTAL_IMG:
        warnings.warn("Number of images in sif file do not match the desribed number")
        
    return fImgs, aImgs, pImgs, bImgs



def get_mean_images(sifData, showImages = False):
    fImgs, aImgs, pImgs, bImgs = get_images(sifData)

    mfImg = np.mean(fImgs, axis=0)
    maImg = np.mean(aImgs, axis=0)
    mpImg = np.mean(pImgs, axis=0)
    mbImg = np.mean(bImgs, axis=0)

    dImg = maImg-mpImg
    
    if showImages == True:

        fig = plt.figure(tight_layout=True, figsize=(16,8))

        gs = gsp.GridSpec(nrows=2, ncols=4)

        ax1 = fig.add_subplot(gs[0, 0])
        d1 = make_axes_locatable(ax1)
        cax1 = d1.append_axes('right', size='5%', pad=0.05)

        ax2 = fig.add_subplot(gs[0, 1])
        d2 = make_axes_locatable(ax2)
        cax2 = d2.append_axes('right', size='5%', pad=0.05)

        ax3 = fig.add_subplot(gs[1, 0])
        d3 = make_axes_locatable(ax3)
        cax3 = d3.append_axes('right', size='5%', pad=0.05)

        ax4 = fig.add_subplot(gs[1, 1])
        d4 = make_axes_locatable(ax4)
        cax4 = d4.append_axes('right', size='5%', pad=0.05)

        ax5 = fig.add_subplot(gs[:, 2:4])
        d5 = make_axes_locatable(ax5)
        cax5 = d5.append_axes('right', size='5%', pad=0.05)

        fig.suptitle("Power = " + str(pwr) +"mW" + "   VCO =" + str(vco) + "   " + "rep="+ str(r))

        aImg_p = ax1.imshow(maImg, origin='lower', cmap="jet")
        fig.colorbar(aImg_p, cax=cax1, orientation='vertical')
        ax1.set_title("with Atoms and Probe")

        pImg_p = ax2.imshow(mpImg, origin='lower', cmap="jet")
        fig.colorbar(pImg_p, cax=cax2, orientation='vertical')
        ax2.set_title("with Probe only")

        fImg_p = ax3.imshow(mfImg, origin='lower', cmap="jet")
        fig.colorbar(fImg_p, cax=cax3, orientation='vertical')
        ax3.set_title("Fluoroscence")

        bImg_p = ax4.imshow(mbImg, origin='lower', cmap="jet")
        fig.colorbar(bImg_p, cax=cax4, orientation='vertical')
        ax4.set_title("Background")

        dImg_p = ax5.imshow(dImg, origin='lower', cmap="jet")
        fig.colorbar(dImg_p, cax=cax5, orientation='vertical')
        ax5.set_title("Difference Imaging")
        
        plt.show()
        
    return dImg, mfImg, maImg, mpImg, mbImg

def smooth_image(img, s):
    img_lpx = ndi.gaussian_filter(img, sigma=s) # Low pass gaussian filter in position domain
    imgk = fft2(img_lpx)   # image in momentum domain
    img_lpk = ndi.fourier_gaussian(imgk, sigma=s)  # Low pass filter in momentum domain
    img_x = ifft2(img_lpk)
    imgx = np.real(img_x)
    return imgx

def sif2img8(img):
    # convert the sif images to 8 bit images for easy processing with open cv
    sif_img_px_max  = 2**15 -1 
    img8 = img*(1/(sif_img_px_max ))*255
    return img8

def get_contours(sifData, showImages=False):

    dImg, mfImg, maImg, mpImg, mbImg = get_mean_images(sifData, showImages=False)
    # Images averaged over number of repeated readings

    mpImg8 = sif2img8(mpImg)    # converting images with probe as 8 bit images
    mbImg8 = sif2img8(mbImg)    # converting background images as 8 bit images

    halfProbeIntensity= 0.5*np.percentile(mpImg8.ravel(), 97)      # Extracting half of probe intensity avoiding spurious peaks

    readNoise = np.percentile(mbImg8, 95)       # Extracting maximum of readNoise avoiding peaks
    if halfProbeIntensity < readNoise:
        warnings.warn("data is not good")

    cmpImg8 = cv.GaussianBlur(src=mpImg8, ksize=(5,5),sigmaX=3,sigmaY=3, borderType=cv.BORDER_DEFAULT) # Gaussian blur of 3 pixels radius with 5 px by 5 px kernel
    res, imgTh = cv.threshold(cmpImg8, 5*readNoise, 255, cv.THRESH_BINARY)  # Setting the values to maximum with and without probe Image needs to be 5 times brighter than the noise

    img8Th = cv.normalize(imgTh, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U) # Normalize the binary classified image with threshold and convert it to 8 bits

    contours, hierarchy = cv.findContours(img8Th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # Find the contours

    if showImages == True:
        mpImgc = copy.deepcopy(mpImg8)
        canvas = np.zeros_like(mpImg8)
        cv.drawContours(canvas, contours, -1, 253, 1)
        cv.drawContours(mpImgc, contours, -1, int(np.max(mpImgc)), 3)

        fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))
        ax1.imshow(mpImg8, cmap="jet", origin='lower')
        ax1.set_title("Input Raw Image")
        ax2.imshow(img8Th, cmap="jet", origin='lower')
        ax2.set_title("Binarized with Threshold")
        ax3.imshow(canvas, cmap="jet", origin='lower')
        ax3.set_title("Edge Detection")
        ax4.imshow(mpImgc, cmap="jet", origin='lower')
        ax4.set_title("Comparision")
        fig.suptitle("Contour Extraction")
        plt.show()

    if len(contours) != 2 :
        raise Exception("Two unique contours not found")
        
    mask = cv.GaussianBlur(src=img8Th, ksize=(3,3),sigmaX=1,sigmaY=1, borderType=cv.BORDER_DEFAULT)
    smoothMask = smooth_image(mask, 1)
    smoothMaskNormed = smoothMask/255
    binarizeMask = np.round(smoothMaskNormed)
    
    return contours, binarizeMask , smoothMaskNormed


def extract_bound(contour, canvas=[], showImages=False):
    
    M = cv.moments(contour) # get the moments of the image matrix
    
    if M['m00'] != 0:
        xc = int(M['m10']/M['m00']) # x centroid
        yc = int(M['m01']/M['m00']) # x centroid

        (xg,yg),radius = cv.minEnclosingCircle(contour)  # enclosing circle gives geometric center and the radius
        xg, yg = int(xg), int(yg)   # making the geometric center as integers
        center = (xg,yg)    # tuple with the geometric center
        radius = int(radius)    # rounding off the radius

        if showImages==True:
            if len(canvas)!=0:
                cv.circle(canvas,center,radius,200,3)
                cv.drawContours(canvas, [contour], -1, 127, 2)
                cv.circle(canvas, (xc, yc), 10, 255, -1)
                plt.figure()
                plt.imshow(canvas, cmap="jet", origin="lower")
                plt.title("extracting circular bound")
                plt.show()
            else:
                warnings.warn("Canvas is not provided. Hence not showing the images")

        return xc, yc, xg, yg, radius
    else:
        raise Exception("image is blank and bounds cannot be extracted")
        
def get_croppedImages(dImg, contours, showImages=False):
    if len(contours) ==2:
        xc, yc, xg, yg, radius = np.transpose([extract_bound(contour) for contour in contours])
        r = np.max(radius)
        xc1, xc1 = xc
        yc1, yc2 = yc
        xg1, xg2 = xg
        yg1, yg2 = yg

        d1 = dImg[yg1-r:yg1+r, xg1-r:xg1+r]
        d2 = dImg[yg2-r:yg2+r, xg2-r:xg2+r]

        blurRadius = np.max([np.max(np.abs(xc-xg)), np.max(np.abs(yc-yg)), 2])

        cv.GaussianBlur(src=d1, ksize=(3,3),sigmaX=blurRadius,sigmaY=blurRadius, borderType=cv.BORDER_DEFAULT)
        cv.GaussianBlur(src=d2, ksize=(3,3),sigmaX=blurRadius,sigmaY=blurRadius, borderType=cv.BORDER_DEFAULT)
        
        if showImages==True:
            fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8), tight_layout=True)

            a1 = make_axes_locatable(ax1)
            cax1 = a1.append_axes('right', size='5%', pad=0.05)
            a2 = make_axes_locatable(ax2)
            cax2 = a2.append_axes('right', size='5%', pad=0.05)
            a3 = make_axes_locatable(ax3)
            cax3 = a3.append_axes('right', size='5%', pad=0.05)
            a4 = make_axes_locatable(ax4)
            cax4 = a4.append_axes('right', size='5%', pad=0.05)

            d1p = ax1.imshow(d1, cmap="jet", origin='lower')
            ax1.set_title("Image 1")
            fig.colorbar(d1p, cax=cax1, orientation='vertical')
            d2p = ax2.imshow(d2, cmap="jet", origin='lower')
            fig.colorbar(d2p, cax=cax2, orientation='vertical')
            ax2.set_title("Image 2")
            ddp = ax3.imshow(d1-d2, cmap="jet", origin='lower')
            fig.colorbar(ddp, cax=cax3, orientation='vertical')
            ax3.set_title("Difference")
            dsp = ax4.imshow(d1+d2, cmap="jet", origin='lower')
            fig.colorbar(dsp, cax=cax4, orientation='vertical')
            ax4.set_title("Sum")
            fig.suptitle("Cropping")
            plt.show()

        return [d1, d2]
    else:
        raise Exception("Two unique contours not found")
    
def smooth_image(dI, s=3):
    dIs = ndi.gaussian_filter(dI, sigma=s)
    dIf = fft2(dIs)
    dIfg = ndi.fourier_gaussian(dIf, sigma=s)
    dIfgi = ifft2(dIfg)
    cdI = np.real(dIfgi)
    return cdI

def centroid(M, order=6, showImages=False):

    sM = smooth_image(M, 5)

    xL,yL = np.shape(M)
    x = np.arange(xL)
    y = np.arange(yL)
    Mo = np.abs(sM)**order
    cx = np.sum(x*np.sum(Mo, axis=1))/np.sum(Mo)
    cy = np.sum(y*np.sum(Mo, axis=0))/np.sum(Mo)
    if showImages==True:
        plt.figure()
        plt.imshow(sM, cmap="jet", origin="lower")
        plt.colorbar()
        plt.plot([cy], [cx], 'o', c='k', mfc='None', ms=4, mew=4)
        plt.plot([cy], [cx], '*', c='w', ms=3)
        plt.title("Smoothened Image")
        plt.show()

    return int(cy), int(cx)

def gauss2d(x, y, mx, my, a, b, c, A, B):
    r2 = a*(x-mx)**2 +  2*b*(x-mx)*(y-my) + c*(y-my)**2
    return B + A *np.exp(-r2)

def gauss2dfit(xy, mx, my, a, b, c, A, B):
    x,y = xy
    g = gauss2d(x, y, mx, my, a, b, c, A, B)
    return g.ravel()

def gaussParams(a, b, c):
    th = 0.5*np.arctan(2*b/(a-c))
    sX = (2*(a*np.cos(th)**2 + 2*b*np.cos(th)*np.sin(th)+ c*np.sin(th)**2))**(-1/2)
    sY = (2*(a*np.sin(th)**2 - 2*b*np.cos(th)*np.sin(th)+ c*np.cos(th)**2))**(-1/2)
    return [th, sX, sY]

def fit_img(d, xc=0, yc=0, abc = [3.76811892e-03, 3.63267180e-04, 8.71083758e-04], showImages=False):
    a,b,c = abc
    xL, yL = np.shape(d)
    x = np.arange(xL)
    y = np.arange(yL)
    X,Y = np.meshgrid(x,y)

    if xc+yc==0:
        xc = int(xL/2)
        yc = int(yL/2)

    xdata = (X,Y)
    ydata = d.ravel()
    sign = 1 if np.abs(np.max(ydata))>np.abs(np.min(ydata)) else -1
    popt, pcov = spo.curve_fit(gauss2dfit, xdata, ydata, p0=[xc,yc, a,b,c, sign*(np.max(ydata)-np.min(ydata)), np.mean(ydata)])
    perr = np.sqrt(np.diag(pcov))
    
    f = gauss2d(X,Y, *popt)

    if showImages ==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X.ravel(), Y.ravel(), d.ravel(), s=0.01, c='k', alpha=0.3)
        ax.plot_surface(X, Y, f, alpha=0.7, cmap='rainbow')
        ax.contour(X, Y, f, zdir='z', offset=-50, cmap="rainbow")
        ax.contour(X, Y, f, zdir='x', offset=1.25*np.min(f),  cmap="rainbow")
        ax.contour(X, Y, f, zdir='y', offset=200,  cmap="rainbow")
        ax.set(xlim=(0,xL), ylim=(0,yL), zlim=(-10, np.max(d)*1.5), xlabel='X', ylabel='Y', zlabel='Z', title="current=")
        plt.show()

        plt.imshow(gauss2d(X,Y, *popt), origin="lower")
        plt.show()

    return popt, perr


def get_splitting_ratio(refData, contours, binarizeMask):
    fImgs_REF, aImgs_REF, pImgs_REF, bImgs_REF = get_images(refData)

    splitting_ratio_fi = []
    A1Arr = []
    A2Arr = []
    x1Arr = []
    x2Arr = [] 
    for i in range(5):
        fImg = fImgs_REF[i] - np.mean(bImgs_REF, axis=0)
        f1_REF, f2_REF = get_croppedImages(fImg, contours)
        popt_f1, pcov_f1 = fit_img(f1_REF)
        popt_f2, pcov_f2 = fit_img(f2_REF)
        x1,y1, a1,b1,c1, A1, B1 = popt_f1
        #print(popt_f1)
        x2,y2, a2,b2,c2, A2, B2 = popt_f2
        α1, σx1, σy1  = gaussParams(a1, b1, c1)
        α2, σx2, σy2  = gaussParams(a2, b2, c2)
        vol1 = A1*np.pi/np.sqrt(a1*c1-b1**2)
        vol2 = A2*np.pi/np.sqrt(a2*c2-b2**2)
        #print(A2/A1, "  " , σx1/σx2 , "   ",  σy1/σy2 )
        splitting_ratio_fi.append(vol1/vol2)
        A1Arr.append(A1)
        A2Arr.append(A2)
        x1Arr.append(x1)
        x2Arr.append(x2)

    pImg_REF = np.mean(pImgs_REF, axis=0) - np.mean(bImgs_REF, axis=0)
    p1_REF, p2_REF = get_croppedImages(pImg_REF, contours)
    bm1, bm2 = get_croppedImages(binarizeMask, contours)
    pwr1 = np.sum(p1_REF)/np.sum(bm1)
    pwr2 = np.sum(p2_REF)/np.sum(bm2)
    splitting_ratio_p = pwr1/pwr2

    splitting_ratio_f = np.mean(splitting_ratio_fi)

    return [splitting_ratio_f, splitting_ratio_p, np.mean(A1), np.mean(A2), np.mean(x1), np.mean(x2)]

def get_rotation(dI, I0, θp):
    return 2 * θp + 0.5*np.arccos((I0*np.cos(4*θp)-2*dI)/I0)


def get_values(sifData, trans, θp, contours, binarizeMask, AFref, xref):
    
    try:
        trans1, trans2 = trans
        X1, X2 = xref
        AF1, AF2 = AFref
        #P1, P2 = Pref
        θp1, θp2 = θp
        #sifData = get_sif(Zcur=zC, repetition=r, Xcur=xC)
        fImgs, aImgs, pImgs, bImgs = get_images(sifData)


        popt_f1A = []
        popt_f2A = []
        for i in range(NUM_FLUOROSCENT_IMG):
            fImg = fImgs[i] - np.mean(bImgs, axis=0)
    
            f1, f2 = get_croppedImages(fImg, contours)
            x0f1, y0f1 = centroid(f1)
            x0f2, y0f2 = centroid(f2)
            popt_f1, pcov_f1 = fit_img(smooth_image(f1, 3), xc=x0f1, yc=y0f1)
            popt_f2, pcov_f2 = fit_img(smooth_image(f2, 3), xc=x0f2, yc=y0f2)
            x1,y1, a1,b1,c1, A1, B1 = popt_f1
            x2,y2, a2,b2,c2, A2, B2 = popt_f2
            #α1, σx1, σy1  = gaussParams(a1, b1, c1)
            #α2, σx2, σy2  = gaussParams(a2, b2, c2)
            #vol1 = A1*np.pi/np.sqrt(a1*c1-b1**2)
            #vol2 = A2*np.pi/np.sqrt(a2*c2-b2**2)
            popt_f1A.append([x1,y1, a1,b1,c1, A1, B1])
            popt_f2A.append([x2,y2, a2,b2,c2, A2, B2])

        xf1,yf1, af1,bf1,cf1, Af1, Bf1 = np.mean(popt_f1A, axis=0)
        xf2,yf2, af2,bf2,cf2, Af2, Bf2 = np.mean(popt_f2A, axis=0)

        vf1 = Af1*np.pi/np.sqrt(af1*cf1-bf1**2)
        vf2 = Af2*np.pi/np.sqrt(af2*cf2-bf2**2)

        dImg = aImgs[0] - pImgs[0]
        d1, d2 = get_croppedImages(dImg, contours)
        ds = trans1*d1 + trans2*d2
        d1s = d1-ds
        d2s = d2-ds

        x0d1, y0d1 = centroid(np.abs(d1s))
        x0d2, y0d2 = centroid(np.abs(d2s))
        
        popt_d1, pcov_d1 = fit_img(smooth_image(d1s,3), xc=x0d1, yc=y0d1)
        popt_d2, pcov_d2 = fit_img(smooth_image(d2s,3), xc=x0d1, yc=y0d2)
        xd1,yd1, ad1,bd1,cd1, Ad1, Bd1 = popt_d1
        xd2,yd2, ad2,bd2,cd2, Ad2, Bd2 = popt_d2

        vd1 = Ad1*np.pi/np.sqrt(ad1*cd1-bd1**2)
        vd2 = Ad2*np.pi/np.sqrt(ad2*cd2-bd2**2)

        """
        return [xf1, yf1, xf2, yf2, xd1, yd1, xd2, yd2,
                af1,bf1,cf1, Af1, Bf1, af2,bf2,cf2, Af2, Bf2,
                ad1,bd1,cd1, Ad1, Bd1, ad2,bd2,cd2, Ad2, Bd2, 
                vf1, vf2, vd1, vd2]
        """
        pImg = pImgs[0] - bImgs[0]
        p1, p2 = get_croppedImages(pImg, contours)
        bm1, bm2 = get_croppedImages(binarizeMask, contours)
        pwr1 = np.sum(p1)/np.sum(bm1)
        pwr2 = np.sum(p2)/np.sum(bm2)

        fnorm = np.mean([Af1/AF1 , Af2/AF2])
        #pnorm = np.mean([pwr1/AF1 , Af2/AF2])
        dI1r = Ad1/(fnorm*pwr1)
        dI2r = Ad2/(fnorm*pwr2)
        px = 0.016   # pixel size
        θfr1 = 2 * θp1 - 0.5*np.arccos(np.clip(np.cos(4*θp1)-(2*dI1r), -1, 1))
        θfr2 = 2 * (-1*θp2) + 0.5*np.arccos(np.clip(np.cos(4*(-1*θp2))-(2*dI2r), -1, 1))
        θfr = (θfr1+θfr2)/2
        shiftx1 = (xf1-X1)*px
        shiftx2 = (xf2-X2)*px
 
    except:
        θfr = np.nan
        shiftx1 = np.nan
        shiftx2 = np.nan
        dI1r = np.nan
        dI2r = np.nan


    return [θfr, dI1r, dI2r, shiftx1, shiftx2]


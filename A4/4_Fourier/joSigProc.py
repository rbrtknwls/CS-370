# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:13:35 2015

@author: jorchard
"""

import numpy as np
#import scipy.io.wavfile
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
#from scipy import ndimage, misc
#from scipy.signal import gaussian
gray = plt.cm.gray


def TimeSamples(f, Omega):
    N = len(f)
    L = float(N) / Omega
    return np.linspace(0., L, N)


def ShiftedFreqSamples(f, Omega):
	if len(np.shape(f))==1:
		N = len(f)
		L = N / float(Omega)
		ctr = np.floor(N/2.)
		shifted_omega = (np.arange(N) - ctr) / L
	else:
		dims = np.array(np.shape(f), dtype=float)
		L = dims / np.array(Omega, dtype=float)
		ctr = np.floor(dims/2.)
		shifted_omega = []
		for idx in range(len(dims)):
			shifted_omega.append((np.arange(dims[idx]) - ctr[idx]) / L[idx])
	return shifted_omega


def FreqSamples(f, Omega):
    N = len(f)
    L = N / float(Omega)
    return np.arange(N)/L
    #return np.linspace(0., N/L, N)

def PlotFT_raw(shifted_omega, F, fig=None, subplot=None, color=None, clf=True):
	#plt.figure(fig)
	#plt.subplot(subplot[0], subplot[1], subplot[2])
	if clf:
		plt.clf()
	if len(np.shape(F))==1:
		if color==None:
			plt.plot(shifted_omega, abs(F));
		else:
			plt.plot(shifted_omega, abs(F), color=color);
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Modulus')
	else:
		#im = plt.imshow(np.flipud(np.log(abs(F)+1)), cmap='gray')
		im = plt.imshow(np.log(abs(F)+1), cmap='gray')
		plt.grid('on', color='0.25')
		plt.plot([shifted_omega[1][0], shifted_omega[1][-1]], [0, 0], 'r:')
		plt.plot([0, 0], [shifted_omega[0][0], shifted_omega[0][-1]], 'r:')
		im.set_extent([shifted_omega[1][0], shifted_omega[1][-1], -shifted_omega[0][-1], -shifted_omega[0][0]])
		plt.title('Frequency Domain')
		plt.xlabel('x Freq (Hz)')
		plt.ylabel('y Freq (Hz)')


def PlotFT(f, Omega=None, fig=None, subplot=None, color=None, clf=True):
	if Omega==None:
		Omega = np.shape(f)
	if len(np.shape(f))==1:
		F = fftshift(fft(f))
	else:
		F = fftshift(fft2(f))
	if color==None:
		PlotFT_raw(ShiftedFreqSamples(f, Omega), F, fig=fig, subplot=subplot, clf=clf)
	else:
		PlotFT_raw(ShiftedFreqSamples(f, Omega), F, fig=fig, subplot=subplot, color=color, clf=clf)
	

def PlotSignal(f, Omega=None, fig=None, subplot=None, clf=True):
	#plt.figure(fig)
	#plt.subplot(subplot[0], subplot[1], subplot[2])
	if clf:
		plt.clf()
	if Omega==None:
		Omega = np.shape(f)
	if len(np.shape(f))==1:
		tt = TimeSamples(f, Omega)
		plt.plot(tt, f)
		plt.title('Time Domain')
		plt.xlabel('Time (s)')
		plt.ylabel('Amplitude')
	else:
		#im = plt.imshow(np.flipud(f), cmap=gray)
		im = plt.imshow(f, cmap=gray)
		#im.set_extent([0, tt[-1]-1, 0, tt[-1]-1])
		plt.title('Spatial Domain')
		plt.xlabel('x')
		plt.ylabel('y')

def PlotFTandSignal(f, DFT=None, clf=True):
	if clf:
		plt.clf()
	plt.subplot(1,2,1)
	if hasattr(DFT, '__iter__'):
		PlotFT(f, clf=clf)
	else:
		print(np.shape(DFT))
		PlotFT_raw(ShiftedFreqSamples(f, np.shape(f)), DFT, clf=clf)
	plt.subplot(1,2,2)
	PlotSignal(np.real(f), clf=clf)

def FilterSignal(f, thresh, band='low'):
	filt = Boxcar(len(f), thresh)
	F = fftshift(fft(f))
	if band=='low':
	    G = F*filt
	else:
	    G = F*(1.-filt)
	g = np.real(ifft(ifftshift(G)))
	return g

def Boxcar(length, radius):
	ctr = np.floor(length/2)
	rr = range(length) - ctr
	filt_mask = abs(rr)<radius
	filt = np.zeros(length)
	filt[filt_mask] = 1.0
	return filt

def Circle(dims, radius):
    ctr = np.floor(np.array(dims)/2)
    rr, cc = np.mgrid[-ctr[0]:(dims[0]-ctr[0]),
                      -ctr[1]:(dims[1]-ctr[1])]
    filt_mask = np.sqrt(rr**2 + cc**2)<radius
    filt = np.zeros(dims)
    filt[filt_mask] = 1.0
    return filt


def FilterImage(f, thresh, band='low'):
    filt = Circle(np.shape(f), thresh)
    F = fftshift(fft2(f))
    if band=='low':
        G = F*filt
    else:
        G = F*(1.-filt)
    g = np.real(ifft2(ifftshift(G)))
    return g






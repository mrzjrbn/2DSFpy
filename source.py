#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:35:09 2020

@author: imasfararachma
"""
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
#import matplotlib.gridspec as gridspec
from scipy import fftpack
#here is a class that define the source

class sources:
    
    def __init__(self,dt,nt,fc):
        self.dt = dt
        self.nt = nt
        self.fc = fc
    #input are dt (sampling time),nt (number of sampe), and fc ()
              
    def ricker(self,a,plot):
        tsour=1/self.fc
        t = np.linspace(0,self.nt-1,self.nt)*self.dt
        t0=tsour*1.5
        T0=tsour*1.5
        tau=math.pi*(t-t0)/T0
        fs=(1-a*tau*tau)*np.exp(-2*tau*tau)
        
        self.fs = fs
        
        if plot == True:
            fig = plt.figure(figsize=(8,2),dpi = 300,constrained_layout=True)
            gs  = fig.add_gridspec(1, 2)
            
            #plot time series
            ax1 = fig.add_subplot(gs[0,0])
            ax1.plot(t,fs,color='black',linewidth=0.2)
            ax1.fill_between(t, fs,where=fs>=0,color='red',alpha=0.3)
            ax1.fill_between(t, fs,where=fs<=0,color='blue',alpha=0.3)

            ax1.set_xlabel('time[s]')
            ax1.set_ylabel('Amplitude')
            plt.xlim((0,np.max(t)))
    
            
            #compute frequency series
            waveletf = fftpack.fft(fs)
            freqs = (fftpack.fftfreq(len(fs))*(1/self.dt))
            
            #plot frequency series
            ax2 = fig.add_subplot(gs[0,1])
            ax2.plot(freqs,np.abs(waveletf),color='blue',linewidth=0.2)
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('|FFT|')
            ax2.fill_between(freqs, np.abs(waveletf),where=np.abs(waveletf)>=0,color='blue',alpha=0.3)
            plt.xlim((0,self.fc+100))
            
            
            
        return t,fs


class forward:
    
    def __init__(self,velocity,density,dx):
        
        self.velocity = velocity 
        self.density = density
        self.dx = dx
   
    def pml(self,npml):
        #most left
        velpml = self.velocity
        
        left  = velpml[:,0]   
        leftarray = np.zeros((len(velpml),npml))
        leftarray[...]=np.array(left)[:,None]
        
        #most right
        right  = velpml[:,-1]   #most right
        rightarray = np.zeros((len(velpml),npml))
        rightarray[...]=np.array(right)[:,None]
        
        #update velocity
        velpml = np.hstack((leftarray,velpml,rightarray))
        
        #uppest
        up = velpml[0,:]
        uparray = np.zeros((npml,len(velpml[1])))
        uparray[...]=np.array(up)[None,:]
        
        #lowest
        down = velpml[-1,:]
        downarray = np.zeros((npml,len(velpml[1])))
        downarray[...]=np.array(down)[None,:]
        
        #update velocity model
        velpml = np.vstack((uparray,velpml,downarray))
        
        self.velpml = velpml
        self.npml = npml
        print('--- > %d points are added to all sides'%(self.npml))
        return velpml

    def FDpar(self):
        fc = np.min(self.velocity)/(20*self.dx)
        fmax=np.min(self.velocity)/(6*self.dx)
        dt = 0.81*(0.606*self.dx/np.max(self.velocity))
        print('FD parameters:')
        print('|fc  | dominant frequency of ricker wavelet = %.6f Hz' %(fc))
        print('|fmax| maximum frequency of our data        = %.6f Hz' %(fmax))
        print('|dt  | maximum sampling time                = %.6f s' %(dt))
             
    def ApplyPML(self,pmlfac,pmlexp):
   
        #velocity now is velocity + pml
        vlc  = self.velpml
        npml = self.npml
        #get the new nx and ny
        nx = len(vlc)#
        ny = len(vlc[1])
        
        #get the new nx and ny for staggered grid 
        ny2 = ny+1
        nx2 = nx+1
        
        #devine velocity for later computation
        #vp0 = np.min(vlc)
        vp = vlc
        vp = vp*vp
        
        #devine density for later computation
        rho = self.density
        
        #initiate container for absorbing boundary condition
        qx = np.zeros((nx,ny))
        qy = np.zeros((nx,ny))

        #Applying PML
        for a in range(npml):
            qx[a,:] = pmlfac*(npml-a-1)**pmlexp      #left
            qx[nx-a-1,:] = pmlfac*(npml-a-1)**pmlexp #Right
            qy[:,a] = pmlfac*(npml-a-1)**pmlexp      #top
            qy[:,ny-a-1] = pmlfac*(npml-a-1)**pmlexp #bottom
        
        #Applying absorbing boundary condition to the velocity + pml model 
        qx = np.hstack((qx[:,0].reshape(nx,1),qx))
        qx = np.vstack((qx[0,:].reshape(1,ny+1),qx)) 
        qy = np.hstack((qy[:,0].reshape(nx,1),qy))
        qy = np.vstack((qy[0,:].reshape(1,ny+1),qy)) 
        
        #assigning value 
        self.qx = qx
        self.qy = qy
        self.nx = nx
        self.ny = ny
        self.nx2 = nx2
        self.ny2 = ny2
        self.vp  = vp
        self.rho = rho
        
        print('--- > absorbing boundaries applied to PML')
        print('--- > PML factor = %.3f  |   PML exponent = %.3f' %(pmlfac,pmlexp))      
    
    def plotmodel(self,sx,sy,recx,recz):
        velplot = self.velocity
        plt.figure(num=None,figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')
        plt.style.use('seaborn-paper')
        plt.imshow(velplot,cmap="RdBu_r")#,extent=[0,(len(velplot[0])*self.dx)-self.dx,(len(velplot[1])*self.dx)-self.dx,0])
        plt.colorbar(fraction=0.02, pad=0.06,shrink = 0.4 , orientation="vertical",label="velocity [m/s]")
        #plt.plot(sx*self.dx,sy*self.dx,'r*')
        plt.plot(sx,sy,'r*')
        plt.plot(recx,recz,'gv',markersize = 2)
        #plt.plot(recx*self.dx,recz*self.dx,'gv',markersize = 2)
        plt.xlabel('x [m]/dx')
        plt.ylabel('z [m]/dx')
        plt.title('Velocity model')
        plt.rcParams.update({'font.size': 6})
        
    def solve(self,recx,recz,sx,sy,t,fs,plotmov):
        
        

        qx = self.qx
        qy = self.qy 
        nx = self.nx 
        ny = self.ny 
        
        nx2 = self.nx2 
        ny2 = self.ny2
        vp  = self.vp 
        rho = self.rho 
        
        isx = sy+self.npml
        isy = sx+self.npml
        irx = recx+self.npml
        iry = recz+self.npml
        
        
        # Initialize fields
        px = np.zeros((nx2,ny2))
        py = np.zeros((nx2,ny2))
        ux = np.zeros((nx2,ny2))
        uy = np.zeros((nx2,ny2))
        
        # spatial spacing
        dx = self.dx 
        dy = dx
        
        # time spacing
        nt = len(t)
        dt = t[1]-t[0]
        
        # all stored results
        Ptot = np.zeros((nt,len(irx)))
        Px   = np.zeros_like(Ptot)
        Py   = np.zeros_like(Ptot)
        Vx   = np.zeros_like(Ptot)
        Vy   = np.zeros_like(Ptot) 
        Vxx  = np.zeros_like(Ptot)
        Vxy  = np.zeros_like(Ptot)
        Vyy  = np.zeros_like(Ptot)
        Vyx  = np.zeros_like(Ptot)
        
        
        #Vxxx  = np.zeros_like(Ptot)
        #Vxyy  = np.zeros_like(Ptot)
        
        
        
        # max and min  for plotting propagation
        amax = np.max(fs)*dt*0.5
        amin = np.min(fs)*dt*0.5
        
        if plotmov == True:
            
            fig = plt.figure(figsize = (10,6))
            ax  = fig.add_subplot(111)
            plt.ion()
            fig.show()
            fig.canvas.draw()
            
        for b in tqdm(np.arange(1,nt)):
            
            # Inject source funtion
            px[isx,isy] = px[isx,isy] + dt*0.5*fs[b];
            py[isx,isy] = py[isx,isy] + dt*0.5*fs[b];
            
            #Update px
            diffop = (ux[1:nx2,0:ny] - ux[0:nx,0:ny])/dx
            pmlop  = qx[1:nx2,1:ny2]*px[1:nx2,1:ny2]
            px[1:nx2,1:ny2] = px[1:nx2,1:ny2] - (np.multiply(dt,pmlop + np.multiply((rho*vp),diffop)))
            
            #Update py
            diffop = (uy[0:nx,1:ny2] - uy[0:nx,0:ny])/dy
            pmlop  = qy[1:nx2,1:ny2]*py[1:nx2,1:ny2]
            py[1:nx2,1:ny2] = py[1:nx2,1:ny2] - (np.multiply(dt,pmlop + np.multiply((rho*vp),diffop)))
            
            #Update ux
            diffop = (px[1:nx2,1:ny2] - px[0:nx,1:ny2] + py[1:nx2,1:ny2] - py[0:nx,1:ny2])/dx;
            pmlop = np.multiply(0.5,(qx[1:nx2,1:ny2]+qx[0:nx,1:ny2])*ux[0:nx,0:ny])
            ux[0:nx,0:ny] = ux[0:nx,0:ny] - (np.multiply(dt/rho,[pmlop + diffop]))
            
            #Update uy
            diffop = (px[1:nx2,1:ny2] - px[1:nx2,0:ny] + py[1:nx2,1:ny2] - py[1:nx2,0:ny])/dy;
            pmlop = np.multiply(0.5,(qy[1:nx2,1:ny2]+qy[1:nx2,0:ny])*uy[0:nx,0:ny])
            uy[0:nx,0:ny] = uy[0:nx,0:ny] - (np.multiply(dt/rho,[pmlop + diffop]))
            
            #total pressure
            Ptot[b,:] = px[iry,irx] + py[iry,irx]
            
            #pressure x y
            Px[b,:] = px[iry,irx]
            Py[b,:] = py[iry,irx]
            
            #velocity x y
            Vx[b,:] = uy[iry,irx]
            Vy[b,:] = ux[iry,irx]
            
            #uxxx = np.gradient(np.sqrt(ux**2 + ux**2),axis=0)
            #Vxxx[b,:] = uxxx[iry,irx]
            
            #uxyy = np.gradient(np.sqrt(ux**2 + uy**2),axis=1)
            #Vxyy[b,:] = uxyy[iry,irx]
            
            
            #velocity gradient
            uxx = np.gradient(uy,axis=0)
            uxy = np.gradient(uy,axis=1)
    
            uyy = np.gradient(ux,axis=1)
            uyx = np.gradient(ux,axis=0)
    
            #velocity gradient
            Vxx[b,:]  =  uxx[iry,irx]
            Vxy[b,:]  =  uxy[iry,irx]
            
            Vyy[b,:]  =  uyy[iry,irx]
            Vyx[b,:]  =  uyx[iry,irx]
        
        
            if plotmov == True:
                if b%20 == 10:
                    ax.clear()
                    ax.imshow(self.velpml,cmap="RdBu_r",alpha=0.9)
                    ax.imshow(px+py, interpolation='none',aspect='auto',alpha=0.5,cmap="binary",vmin= amin,vmax = amax)#,extent=[0,8000-dx,3000-dx,0])
                    #ax.imshow(uy+ux, interpolation='none',aspect='auto',alpha=0.5,cmap="binary")#,vmin= amin,vmax = amax)#,extent=[0,8000-dx,3000-dx,0])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(self.npml,ny-self.npml)
                    ax.set_ylim(nx-self.npml,self.npml)
                    #plt.yticks([])

                    

                    fig.canvas.draw()
                    #plt.imshow(self.velpml,cmap="RdBu_r",alpha=0.9)
                    #plt.imshow(px+py, interpolation='none',aspect='auto',alpha=0.5,cmap="binary")#,extent=[0,8000-dx,3000-dx,0])
                    #plt.clim(amin,amax)
                    #ax.set_clim(amin,amax)
                    #plt.xlim(self.npml,ny-self.npml)
                    #plt.ylim(nx-self.npml,self.npml)
                    #plt.xticks([])
                    #plt.yticks([])
                    #plt.draw()
                    #plt.pause(.001)
                    #plt.show()
                    #plt.clf()
                
        Results = {
                  "Px":Px,
                  "Py":Py,
                  "Ptot":Ptot,
                 "Vx":Vx,
                 #"Vxxx":Vxxx,
                 #"Vxyy":Vxyy,
                 "Vy":Vy,
                 "Vxx":Vxx,
                 "Vxy":Vxy,
                 "Vyy":Vyy,
                 "Vyx":Vyx
                 }
        
        return Results


class data:
    def __init__(self,fs,wavelet,t):
        self.fs = fs
        self.wavelet = wavelet
        self.t  = t 
    
    def deconv(self,data,cutoff,order,fc,plotdeconv):
        t = self.t
        wavelet = self.wavelet
        
        #filterring using butterworth filter
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
     
        
        #defube frequency of the wavelet
        waveletf  = fftpack.fft(wavelet) 
        signalf   = fftpack.fft(data) 
        freqs     = (fftpack.fftfreq(len(wavelet))*(self.fs))
        
        Gf  = fftpack.ifft((signalf)/(waveletf+0.005*np.max(waveletf)));#Gf[0:5]=0
        Gft =   lfilter(b, a, Gf)
        Gftf = fftpack.fft(Gft)
        
        #convolve 
        traceconv = waveletf*Gftf; traceconv = fftpack.ifft(traceconv)
        
        if plotdeconv == True:
            
            fig = plt.figure(figsize=(9,7),dpi = 300,constrained_layout=True)
            gs  = fig.add_gridspec(3, 3)
                        
            #plot time series
            ax1 = fig.add_subplot(gs[0,0:2])
            ax1.plot(t,data,color='black',linewidth=1,label='modelled')
            ax1.plot(t,traceconv,'--r',linewidth=1,label='convolved')
            ax1.fill_between(t, data,where=data>=0,color='red',alpha=0.3)
            ax1.fill_between(t, data,where=data<=0,color='blue',alpha=0.3)
            plt.xlim((0,np.max(t)))
            plt.title('signal')
            plt.legend()
            
            #plot frequency series
            ax2 = fig.add_subplot(gs[0,-1])
            ax2.plot(freqs,np.abs(signalf),color='blue',linewidth=0.2)
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('|FFT|')
            ax2.fill_between(freqs, np.abs(signalf),where=np.abs(signalf)>=0,color='blue',alpha=0.3)
            plt.xlim((0,fc+50))
            plt.title('signal spectrum')
            
            #plot time series
            ax3 = fig.add_subplot(gs[1,0:2])
            ax3.plot(t,wavelet,color='black',linewidth=0.2)
            ax3.fill_between(t, wavelet,where=wavelet>=0,color='red',alpha=0.3)
            ax3.fill_between(t, wavelet,where=wavelet<=0,color='blue',alpha=0.3)
            plt.xlim((0,np.max(t)))
            plt.title('wavelet')
            
            #plot frequency series
            ax4 = fig.add_subplot(gs[1,-1])
            ax4.plot(freqs,np.abs(waveletf),color='blue',linewidth=0.2)
            ax4.set_xlabel('Frequency [Hz]')
            ax4.set_ylabel('|FFT|')
            ax4.fill_between(freqs, np.abs(waveletf),where=np.abs(waveletf)>=0,color='blue',alpha=0.3)
            plt.xlim((0,fc+50))
            plt.title('wavelet spectrum')
            
            #plot time series
            ax5 = fig.add_subplot(gs[2,0:2])
            ax5.plot(t,Gft,color='black',linewidth=0.2)
            ax5.fill_between(t, Gft,where=Gft>=0,color='red',alpha=0.3)
            ax5.fill_between(t, Gft,where=Gft<=0,color='blue',alpha=0.3)
            plt.xlim((0,np.max(t)))
            plt.title('Green s function')
            
            #plot frequency series
            ax6 = fig.add_subplot(gs[2,-1])
            ax6.plot(freqs,np.abs(Gftf),color='blue',linewidth=0.2)
            ax6.set_xlabel('Frequency [Hz]')
            ax6.set_ylabel('|FFT|')
            ax6.fill_between(freqs, np.abs(Gftf),where=np.abs(Gftf)>=0,color='blue',alpha=0.3)
            plt.xlim((0,fc+50))
            plt.title('Green s function spectrum')
            plt.show()

       

        

                    







        

        

        

        

        

        

        

        

        

        

        

        

        

        

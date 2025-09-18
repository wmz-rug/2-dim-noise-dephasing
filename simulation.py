#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:38:45 2024

@author: Meng-Zhi Wu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import scipy.signal as signal

#%%%
#--------------------------------------
# Trajectories and Transfer Function
#--------------------------------------
# trajectories

omega0 = 2*np.pi
tf = 2*np.pi/omega0

u1 = 2.2
u2 = 1.8
v1 = 1.1
v2 = 0.9

N = 1000
t = np.linspace(0, tf, N)
x1 = 2*u1*np.sin(omega0*t/2)**2
x2 = 2*u2*np.sin(omega0*t/2)**2
y1 = 2*v1*np.sin(omega0*t/2)**2
y2 = 2*v2*np.sin(omega0*t/2)**2

#%%
# plot trajectories

xd = x1 - x2
yd = y1 - y2
plt.plot(t, xd, label="$\Delta x$")
plt.plot(t, yd, label="$\Delta y$")
plt.xlabel(r"$\omega_0 t/2\pi$", fontsize=16)
plt.yticks([2*(u1-u2),2*(v1-v2)],["$2\Delta u$", "$2\Delta v$"])
plt.ylabel("Superposition Size", fontsize=16)
plt.legend()

#%% 
# plot dimensionless transfer function

omega = 2*np.pi*np.logspace(-1, 2.01, N)
F = 4*omega0**4*np.sin(np.pi*omega/omega0)**2 / (omega**3-omega0**2*omega)**2
plt.loglog(omega/(2*np.pi), F)
plt.xlabel("Frequency/Hz", fontsize=16)
plt.ylabel("Transfer Function $F_0(\omega)$", fontsize=16)
plt.ylim([1e-18,1e1])
plt.grid(color='gray', alpha=0.3)
ax = plt.gca()
ax.annotate("$4\pi^2/\omega_0^2$", xy=(-0.05,1.05), xycoords="axes fraction", ha='center', fontsize=12)
#plt.savefig('F0.png', bbox_inches='tight')


#%%%
#------------------------------------------
# Preparation for Simulation 
#------------------------------------------
# test uniform random number seed

np.random.seed(0)
x = np.random.randn(102400)
y = np.random.randn(102400)

f, Pxy = signal.csd(x, y, fs=1.0, nperseg=256)
f, Px = signal.welch(x, fs=1.0, nperseg=256)
f, Py = signal.welch(y, fs=1.0, nperseg=256)
coh = abs(Pxy)**2/ (Px*Py)
plt.loglog(f[1:], coh[1:])
plt.show()

#%%
# random seed and Runge-Kutta algorithms

def generate_noise(N, dim, paras):
    # create random numbers by using the Linux system file "/dev/random"
    with open('/dev/random', 'rb') as f:
        random_bytes = f.read(dim*4*N)
    random_numbers = np.frombuffer(random_bytes, dtype=np.uint32)
    u1 = random_numbers[0::2]/(2**32)
    u2 = random_numbers[1::2]/(2**32)
    # Box-Muller transform to create two independent Gaussian variables
    A, dt = paras
    noise1 = A*np.sqrt(dt) * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    noise2 = A*np.sqrt(dt) * np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    noises = np.stack((noise1, noise2), axis=1)
    return noises

def runge_kutta_4(f, y0, t, noises, paras):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    if noises.shape != (n, 2):
        raise TypeError("The shape of noises is incorrect!")
    
    for i in range(1, n):
        h = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1], noises[i-1, :], paras)
        noises_half = (noises[i-1, :]+noises[i, :])/2
        k2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k1, noises_half, paras)
        k3 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k2, noises_half, paras)
        k4 = f(t[i-1] + h, y[i-1] + h*k3, noises[i, :], paras)
        y[i] = y[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

#%%
# theoretical formulae

def dynamical_equations(t, z0, noisesT, paras):
    x, vx, y, vy = z0
    noise1T, noise2T = noisesT
    omega0, gamma, K = paras
    
    dxdt = vx
    dvxdt = K*y - gamma*vx - omega0**2*x + noise1T
    dydt = vy
    dvydt = K*x - gamma*vy - omega0**2*y + noise2T 
    return np.array([dxdt, dvxdt, dydt, dvydt])

def Sxx_th(freq, A, omega0, gamma, K):
    return A**2*((omega0**2-(2*np.pi*freq)**2)**2 + (2*np.pi*freq)**2*gamma**2 + K**2) / (((omega0**2-(2*np.pi*freq)**2)**2 - (2*np.pi*freq)**2*gamma**2 - K**2)**2 + 4*(2*np.pi*freq)**2*gamma**2*(omega0**2-(2*np.pi*freq)**2)**2) / (2*np.pi**4)

def Sxy_th_re(freq, A, omega0, gamma, K):
    return 2*A**2*K * (omega0**2 - (2*np.pi*freq)**2) / (((omega0**2-(2*np.pi*freq)**2)**2 - (2*np.pi*freq)**2*gamma**2 - K**2)**2 + 4*(2*np.pi*freq)**2*gamma**2*(omega0**2-(2*np.pi*freq)**2)**2) / (2*np.pi**4)

def sigmaUU_th(freq, A, omega0, gamma, K):
    return 4*np.pi**2/(2*np.pi*freq) * A**2*1 / ((omega0**2-K-(2*np.pi*freq)**2)**2+gamma**2*(2*np.pi*freq)**2) / (2*np.pi**4)

def sigmaVV_th(freq, A, omega0, gamma, K):
    return 4*np.pi**2/(2*np.pi*freq) * A**2*1 / ((omega0**2+K-(2*np.pi*freq)**2)**2+gamma**2*(2*np.pi*freq)**2) / (2*np.pi**4)


#%%%
#------------------------------------------
# Simulation on Noise and FFT Analysis
#------------------------------------------
# Noise class

class Noise:
    def __init__(self,**kwargs):
        N = kwargs.get('N', 100000)
        self.N = int(N)
        self.dt = kwargs.get('dt', 0.01)
        self.time = np.linspace(0, self.N*self.dt, self.N)
        
        self.Omega0 = kwargs.get('Omega0', 1)
        self.K = kwargs.get('K', 1e-1)
        self.gamma = kwargs.get('gamma', 1e-2)
        self.A = kwargs.get('A', 1e-2)
        
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        self.U = (self.x+self.y)/np.sqrt(2)
        self.V = (self.x-self.y)/np.sqrt(2)
        self.vx = np.zeros(N)
        self.vy = np.zeros(N)
        self.x0 = kwargs.get('x0', 0.0)
        self.y0 = kwargs.get('y0', 0.0)
        self.vx0 = kwargs.get('vx0', 0.0)
        self.vy0 = kwargs.get('vy0', 0.0)
        
        self.MCdata = False
        
    def Monte_Carlo(self):
        noises = generate_noise(self.N, 2, [self.A, self.dt])
        paras = [self.Omega0, self.gamma, self.K]
        init_values = [self.x0,self.vx0,self.y0,self.vy0]
        sol = runge_kutta_4( dynamical_equations, init_values, self.time, noises, paras)
        self.x = sol[:, 0]
        self.vx = sol[:, 1]
        self.y = sol[:, 2]
        self.vy = sol[:, 3]
        self.U = (self.x+self.y)/np.sqrt(2)
        self.V = (self.x-self.y)/np.sqrt(2)
        self.__default_PSD()
        self.MCdata = True
        
    def __default_PSD(self):
        self.f_x, self.Pxx = signal.welch(self.x, fs=1/self.dt, window='hann', nperseg=self.N/10, noverlap=self.N/20)
        self.f_y, self.Pyy = signal.welch(self.y, fs=1/self.dt, window='hann', nperseg=self.N/10, noverlap=self.N/20)
        self.f_xy, self.Pxy = signal.csd(self.x, self.y, fs=1/self.dt, window='hann', nperseg=self.N/10, noverlap=self.N/20)
        freq, Puu = signal.welch(self.U, fs=1/self.dt, window='hann', nperseg=self.N/10, noverlap=self.N/20)
        freq, Pvv = signal.welch(self.V, fs=1/self.dt, window='hann', nperseg=self.N/10, noverlap=self.N/20)
        self.freq = freq
        self.sigmaUU = 4*np.pi**2/(2*np.pi*freq) * Puu
        self.sigmaVV = 4*np.pi**2/(2*np.pi*freq) * Pvv

#%%
# plot methods

def plot_trajectory(noise=Noise(N=100000, dt=0.02), figName=None):
    if not noise.MCdata:
        noise.Monte_Carlo()
        
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(noise.time[noise.N//5:noise.N//2], noise.x[noise.N//5:noise.N//2], label='x')
    ax.plot(noise.time[noise.N//5:noise.N//2], noise.y[noise.N//5:noise.N//2], label='y')
    ax.set_xlabel('Time/s', fontsize=20)
    ax.set_ylabel('Position/m', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    if figName is None:
        plt.show()
    else:
        plt.savefig(figName, bbox_inches='tight')

def plot_velocity(noise=Noise(N=100000, dt=0.02), figName=None):
    if not noise.MCdata:
        noise.Monte_Carlo()
        
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(noise.time[noise.N//5:noise.N//2], noise.vx[noise.N//5:noise.N//2], label='$v_x$')
    ax.plot(noise.time[noise.N//5:noise.N//2], noise.vy[noise.N//5:noise.N//2], label='$v_y$')
    ax.set_xlabel('Time/s', fontsize=20)
    ax.set_ylabel('Velocity/(m/s)', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    if figName is None:
        plt.show()
    else:
        plt.savefig(figName, bbox_inches='tight')
    
def plot_PSD(noise=Noise(N=100000, dt=0.02), figName=None):
    if not noise.MCdata:
        noise.Monte_Carlo()
    
    psd_x_th = Sxx_th(noise.f_x, noise.A, noise.Omega0, noise.gamma, noise.K)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.loglog(2*np.pi*noise.f_x[1:], (2*np.pi*noise.f_x[1:])**4*noise.Pxx[1:], label='simulation')
    ax.loglog(2*np.pi*noise.f_x[1:], (2*np.pi*noise.f_x[1:])**4*psd_x_th[1:], label='analytic')
    ax.set_xlabel('Frequency/($\Omega_0/2\pi$)', fontsize=20)
    ax.set_ylabel(r'Power Spectrum Density $S_{aa}/(m/(s^2\cdot\sqrt{\text{Hz}}$))', fontsize=16)
    ax.set_ylim([2e-14,1e-2])
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16, loc='lower right')
    ax.grid(color='gray', alpha=0.3)
    if figName is None:
        plt.show()
    else:
        plt.savefig(figName, bbox_inches='tight')
    
    
def plot_crossPSD(noise=Noise(N=100000, dt=0.02), figName=None, yscale=1e-5):
    if not noise.MCdata:
        noise.Monte_Carlo()
        
    psd_x_th = Sxx_th(noise.f_x, noise.A, noise.Omega0, noise.gamma, noise.K)
    cor_xy_re_th = Sxy_th_re(noise.f_xy, noise.A, noise.Omega0, noise.gamma, noise.K)
    
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.semilogx(2*np.pi*noise.f_xy[1:], (2*np.pi*noise.f_xy[1:])**4*np.real(noise.Pxy[1:]), label='simulation')
    ax.semilogx(2*np.pi*noise.f_xy[1:], (2*np.pi*noise.f_xy[1:])**4*cor_xy_re_th[1:], '--', label='analytic')
    ax.set_xlabel('Frequency/($\Omega_0/2\pi$)', fontsize=20)
    ax.set_ylabel(r'Cross PSD $S_{a_xa_y}/(m/(s^2\cdot\sqrt{\text{Hz}}$))', fontsize=18)
    ax.set_yscale('symlog', linthresh=yscale)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16, loc='lower right')
    ax.grid(color='gray', alpha=0.3)
    if figName is None:
        plt.show()
    else:
        plt.savefig(figName, bbox_inches='tight')
    
#%%
# simulate and plot x(t), y(t)

noise = Noise(N=20000, dt=0.05, K=0.5)
noise.Monte_Carlo()

plot_trajectory(noise)
#plot_trajectory(noise, 'xyt.png')
plot_velocity(noise)
#plot_velocity(noise, 'vxvyt.png')

#%%
# PSD and cross-PSD

noise = Noise(N=100000, dt=0.05, K=0.9)
#noise = Noise(N=1000000, dt=0.02)
noise.Monte_Carlo()
plot_PSD(noise)
#plot_PSD(noise, figName='Saa.png')
plot_crossPSD(noise, yscale=1.1e-5)
#plot_crossPSD(noise, figName='Saxay.png')

#%%
# middle gamma

noise = Noise(N=100000, dt=0.05, K=0.9, gamma=0.3)
noise.Monte_Carlo()

#plot_trajectory(noise)
#plot_trajectory(noise, 'xyt_middle.png')
#plot_velocity(noise)
#plot_velocity(noise, 'vxvyt_middle.png')

plot_PSD(noise)
#plot_PSD(noise, figName='Saa_middle.png')
plot_crossPSD(noise, yscale=4e-7)
#plot_crossPSD(noise, figName='Saxay_middle.png', yscale=4e-7)

#%%
# large gamma

noise = Noise(N=100000, dt=0.05, K=0.9, gamma=1.5)
noise.Monte_Carlo()

#plot_trajectory(noise)
#plot_trajectory(noise, 'xyt_large.png')
#plot_velocity(noise)
#plot_velocity(noise, 'vxvyt_large.png')

plot_PSD(noise)
#plot_PSD(noise, figName='Saa_large.png')
plot_crossPSD(noise, yscale=2e-8)
#plot_crossPSD(noise, figName='Saxay_large.png', yscale=2e-8)


#%%%
#----------------------------------
# Dephasing Factor sigma2
#----------------------------------
# different damping factor gamma

freq = noise.freq[1:]
sigma1 = sigmaVV_th(freq, 1e-2, 1, 0.1, 0.9)
sigma2 = sigmaVV_th(freq, 1e-2, 1, 1, 0.9)
sigma3 = sigmaVV_th(freq, 1e-2, 1, 10, 0.9)

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.loglog(2*np.pi*freq, (2*np.pi*freq)**3*sigma1, label='$\gamma=0.1\Omega_0$')
ax.loglog(2*np.pi*freq, (2*np.pi*freq)**3*sigma2, label='$\gamma=1\Omega_0$')
ax.loglog(2*np.pi*freq, (2*np.pi*freq)**3*sigma3, label='$\gamma=10\Omega_0$')
ax.set_xlabel('Test Mass Frequency $\omega_0$/$\Omega_0$', fontsize=20)
ax.set_ylabel(r'Phase Variance $\sigma^2$', fontsize=18)
ax.tick_params(labelsize=16)
ax.legend(fontsize=16, loc='upper right')
ax.grid(color='gray', alpha=0.3)
#plt.savefig('sigma_gamma.png', bbox_inches='tight')
plt.show()

#%%
# different coupling k

freq = noise.freq[1:]
sigma1 = sigmaVV_th(freq, 1e-2, 1, 0.1, 0.9)
sigma2 = sigmaVV_th(freq, 1e-2, 1, 0.1, 0)
sigma3 = sigmaVV_th(freq, 1e-2, 1, 0.1, -0.9)

fig, ax = plt.subplots(1, figsize=(8, 6))
#ax.loglog(2*np.pi*freq, (2*np.pi*freq)**3*sigma1, label='$k=0.9\Omega_0^2$')
#ax.loglog(2*np.pi*freq, (2*np.pi*freq)**3*sigma2, label='$k=0$')
#ax.loglog(2*np.pi*freq, (2*np.pi*freq)**3*sigma3, label='$k=-0.9\Omega_0^2$')
ax.semilogy(2*np.pi*freq, (2*np.pi*freq)**3*sigma1, label='$k=0.9\Omega_0^2$')
ax.semilogy(2*np.pi*freq, (2*np.pi*freq)**3*sigma2, label='$k=0$')
ax.semilogy(2*np.pi*freq, (2*np.pi*freq)**3*sigma3, label='$k=-0.9\Omega_0^2$')
ax.set_xlim([0,3])
ax.set_xlabel('Test Mass Frequency $\omega_0$/$\Omega_0$', fontsize=20)
ax.set_ylabel(r'Phase Variance $\sigma^2$', fontsize=18)
ax.tick_params(labelsize=16)
ax.legend(fontsize=16, loc='upper right')
ax.grid(color='gray', alpha=0.3)
#plt.savefig('sigma_k.png', bbox_inches='tight')
plt.show()

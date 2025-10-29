import numpy as np


def func(freq, angle):
    """Example: complex-valued response"""
    return np.exp(1j * 2 * np.pi * freq) * np.cos(np.deg2rad(angle))

def func_resonance(freq, angle):
    """Lorentz-like resonance in frequency, smooth cos variation in angle"""
    return 1.0 / (1 - (freq/5.0)**2 + 0.05j) * np.cos(angle)

def func_double_resonance(freq, angle):
    """Two nearby resonances, complex-valued, smooth angle"""
    r1 = 1.0 / (1 - (freq/3.0)**2 + 0.03j)
    r2 = 0.5 / (1 - (freq/7.0)**2 + 0.02j)
    return (r1 + r2) * np.sin(angle*2)

def func_angle_peak(freq, angle):
    """Angle of peak shifts with frequency"""
    theta0 = np.pi/4 + 0.1*np.sin(freq)
    return np.exp(-((angle - theta0)/0.1)**2) * np.exp(1j*freq)

def func_interference(freq, angle):
    """Rapid oscillations in both freq and angle"""
    return np.sin(2*np.pi*freq) * np.cos(5*angle) + 1j*np.cos(2*np.pi*freq)*np.sin(3*angle)

def func_sharp_resonance(freq, angle):
    """Combination of sharp Lorentz peak and smooth background"""
    lorentz = 1.0 / (1 - (freq/4.0)**6 + 0.01j)
    background = 0.2*np.cos(angle)
    return lorentz + background

def func_non_separable(freq, angle):
    """Angle and frequency interact nonlinearly"""
    return np.exp(-((freq-5*np.sin(angle))**2)/2) * np.exp(1j*freq*angle)

import numpy as np
import scipy.fftpack
import scipy.integrate
import matplotlib.pyplot as plt


def normalize(psi_x,x):
    dx = x[1]-x[0]
    n = np.sum(psi_x*np.conj(psi_x))*dx # metoda prostokatow, wstyd
    print(n,dx)
    return psi_x/np.sqrt(n)


def potential(x):
    return - 0.5*x*x


def potential_energy(psi_x, x):
    dx = x[1]-x[0]
    return np.sum(psi_x*np.conj(psi_x)*potential(x))*dx


def kinetic_energy(psi_x, x):
    dx = x[1]-x[0]
    dpsi_x = scipy.fftpack.diff(psi_x)
    return -np.sum(dpsi_x*np.conj(dpsi_x))*dx/2.0


def evolve_imaginary(psi_x,x,dt):
    period = np.max(x) - np.min(x)
    k = scipy.fftpack.fftfreq(psi_x.size,period)
    
    # move to reciprocal space
    psi_k = scipy.fftpack.fft(psi_x)
    
    # multiply by e^(Tdt)
    psi_k = np.exp(-0.5*k**2*dt)*psi_k
    
    # move to posiitons space
    psi_x = scipy.fftpack.ifft(psi_k)
    
    #multiply by e^(Vdt)
    psi_x = np.exp(-potential(x)*dt)*psi_x

    ############################################ 
     # move to reciprocal space
    psi_k = scipy.fftpack.fft(psi_x)
    
    # multiply by e^(Tdt)
    psi_k = np.exp(-0.5*k**2*dt)*psi_k
     
    # move to posiitons space
    psi_x = scipy.fftpack.ifft(psi_k)
    ############################################ 

    # normalize
    psi_x = normalize(psi_x,x)
    return normalize(psi_x,x)


def evolve_real(psi_x,x,dt):
    period = np.max(x) - np.min(x)
    k = scipy.fftpack.fftfreq(psi_x.size,period)
    
    psi_k = scipy.fftpack.fft(psi_x)
    
    psi_k = np.exp(0.5*1j*k**2*dt)*psi_k
    
    psi_x = scipy.fftpack.ifft(psi_k)
    
    psi_x = np.exp(1j*potential(x)*dt)*psi_x
    
    return psi_x


def make_evolution(psi_x,x,time,dt):
    timesteps = int(time/dt)
    ekin_list = []
    epot_list = []
    etot_list = []

    psi_x = evolve_imaginary(psi_x, x, dt)
    for i in range(timesteps):
        psi_x = evolve_real(psi_x,x, dt)
        ekin = kinetic_energy(psi_x,x)
        epot = potential_energy(psi_x,x)
        #print('ekin:',ekin,'  epot:',epot,'  etot:',ekin+epot,' virial:',2*np.abs(ekin-epot))
        ekin_list.append(ekin)
        epot_list.append(epot)
        etot_list.append(ekin + epot)

    return range(timesteps), ekin_list, epot_list, etot_list


# lattice
xmin = -5.25
xmax =  5.25
Nx = 2048
x = np.linspace(xmin,xmax,Nx)

dt = 1e-4

# initial conditions
psi_x = np.exp(-0.5*x**2)*np.pi**(-0.25)
#plt.plot(x,psi_x*np.conj(psi_x),label='not normalized')
psi_x = normalize(psi_x,x)
psi_x = normalize(psi_x,x)

"""
plt.title('Groundstate')
plt.plot(x,psi_x*np.conj(psi_x),label='normalized')

plt.legend()
plt.show()
plt.close('all')
"""

E_k = kinetic_energy(psi_x, x)
E_p = potential_energy(psi_x, x)

print(E_k, E_p, E_k + E_p)

temp = make_evolution(psi_x,x,.1,dt)
plt.scatter(temp[0], temp[1])
plt.scatter(temp[0], temp[2])
plt.scatter(temp[0], temp[3])
plt.show()


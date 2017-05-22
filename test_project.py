import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation


def normalize(psi_x, x, dx):
    n = np.sum(psi_x*np.conj(psi_x))*dx  # metoda prostokatow, wstyd
    return psi_x/np.sqrt(n)


def potential(x):
    return 0.5*x*x


def potential_energy(psi_x, x, dx):
    return np.sum(psi_x*np.conj(psi_x)*potential(x))*dx


def kinetic_energy(psi_x, x):
    # dpsi_k = scipy.fftpack.fft(psi_x)
    # return -np.sum(dpsi_x*np.conj(dpsi_x))*dx/2.0
    # p = np.trapz(np.conj(dpsi_k)*k*dpsi_k,k,dk)
    p = np.sum(np.conj(psi_x)*-1j*np.gradient(psi_x))
    return p**2/2


def evolve_imaginary(psi_x, x, dt):
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


def evolve_real(psi_x, x, dt):
    psi_k = scipy.fftpack.fft(psi_x)
    
    psi_k = np.exp(0.5*1j*k**2*dt)*psi_k
    
    psi_x = scipy.fftpack.ifft(psi_k)
    
    psi_x = np.exp(1j*potential(x)*dt)*psi_x
    
    return psi_x


def make_evolution(psi_x, x, time, dt):
    timesteps = int(time/dt)
    ekin_list = []
    epot_list = []
    etot_list = []

    psi_x = evolve_imaginary(psi_x, x, dt)
    for i in range(timesteps):
        psi_x = evolve_imaginary(psi_x, x, dt)
        ekin = kinetic_energy(psi_x, x)
        epot = potential_energy(psi_x, x)
        # print('ekin:',ekin,'  epot:',epot,'  etot:',ekin+epot,' virial:',2*np.abs(ekin-epot))
        ekin_list.append(ekin)
        epot_list.append(epot)
        etot_list.append(ekin + epot)

    # return range(timesteps), ekin_list, epot_list, etot_list
    return psi_x

# lattice
x_min = -5.25
x_max = +5.25
Nx = 2048
x = np.linspace(x_min, x_max, Nx)
dx = x[1]-x[0]
dt = 1e-4

period = np.max(x) - np.min(x)

dk = 2 * np.pi / period
k = -0.5 * Nx * dk + dk * np.arange(Nx)

# initial conditions
psi_x = np.exp(-0.5*x**2)*np.pi**(-0.25)
# plt.plot(x,psi_x*np.conj(psi_x),label='not normalized')
psi_x = normalize(psi_x, x, dx)
psi_x = normalize(psi_x, x, dx)

psi_x = (psi_x*np.exp(-1j * k[0] * x) * dx / np.sqrt(2 * np.pi))

psi_x = normalize(psi_x, x, dx)
psi_x = normalize(psi_x, x, dx)


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

# psi_x = make_evolution(psi_x,x,10,dt)
# plt.scatter(temp[0], temp[1])
# plt.scatter(temp[0], temp[2])
# plt.scatter(temp[0], temp[3])
fig = plt.figure()

ims = []

for i in range(200):
    ims.append(plt.plot(x, normalize(psi_x*np.exp(1j * k[0] * x) * np.sqrt(2 * np.pi) / dx, x), color='0'))
    psi_x = make_evolution(psi_x, x, 0.01, dt)

imp_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
# psi_x*= np.exp(1j * k[0] * x) *np.sqrt(2 * np.pi) / dx
# plt.plot(x,normalize(psi_x*np.exp(1j * k[0] * x) *np.sqrt(2 * np.pi) / dx,x))
# plt.plot(x, 0.7/(np.cosh(x)**2))

plt.show()

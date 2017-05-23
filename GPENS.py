import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def potential(x_):
    return 0.5*x_**2


class GPENS:
    def __init__(self, x, psi, dt, end_time):
        self.x = x
        self.psi = psi
        self.dt = dt
        self.end_time = end_time

        self.x_min = x[0]
        self.x_max = x[-1]
        self.dx = abs(x[1] - x[0])

        period = np.max(x) - np.min(x)
        self.dk = 2 * np.pi / period
        self.k = -0.5 * self.x.size * self.dk + self.dk * np.arange(self.x.size)

        self.normalize(self.psi)
        self.normalize(self.psi)
        self.psi = (self.psi*np.exp(-1j * self.k[0] * self.x) * self.dx / np.sqrt(2 * np.pi))
        self.normalize(self.psi)
        self.normalize(self.psi)

        self.time_steps = int(self.end_time/self.dt)
        self.energy = np.zeros((3, self.time_steps)).astype(np.float32)
        self.E_k = None
        self.E_p = None
        self.E_tot = None

    def normalize(self, psi):
        # sum function using rectangles method
        n = np.sum(psi*np.conj(psi))*self.dx
        return psi/np.sqrt(n)

    def update_energy(self):
        self.E_k = (np.sum(np.conj(self.psi)*-1j*np.gradient(self.psi)))**2/2
        self.E_p = np.sum(self.psi*np.conj(self.psi)*potential(self.x))*self.dx
        self.E_tot = self.E_k + self.E_p

    def evolve_imaginary(self):
        # move to reciprocal space
        psi_k = np.fft.fft(self.psi)

        # multiply by e^(Tdt)
        psi_k = np.exp(-0.5*self.k**2*self.dt)*psi_k

        # move to positions space
        psi_x = np.fft.ifft(psi_k)

        # multiply by e^(Vdt)
        psi_x = np.exp(-potential(self.x)*self.dt)*psi_x

        ############################################
        # move to reciprocal space
        psi_k = np.fft.fft(psi_x)

        # multiply by e^(Tdt)
        psi_k = np.exp(-0.5*self.k**2*self.dt)*psi_k

        # move to posiitons space
        psi_x = np.fft.ifft(psi_k)
        ############################################

        # normalize
        psi_x = self.normalize(psi_x)
        return self.normalize(psi_x)

    def evolve_real(self):
        psi_k = np.fft.fft(self.psi)

        psi_k = np.exp(0.5*1j*self.k**2*self.dt)*psi_k

        psi_x = np.fft.ifft(psi_k)

        return np.exp(1j*potential(self.x)*self.dt)*psi_x

    def make_evolution(self):
        self.psi = self.evolve_imaginary()
        for index in range(self.time_steps):
            self.psi = self.evolve_imaginary()
            self.update_energy()
            self.energy[0, index] = self.E_k
            self.energy[1, index] = self.E_p
            self.energy[2, index] = self.E_tot

    def animation(self, steps, end_time):
        ims = []
        for i in range(steps):
            ims.append(plt.plot(self.x,
                                self.normalize(self.psi*np.exp(1j * self.k[0] * self.x) * np.sqrt(2 * np.pi) / self.dx),
                                color='0'))
            self.make_evolution(end_time)

        return ims

x_min = -5.25
x_max = +5.25
Nx = 2048
x_temp = np.linspace(x_min, x_max, Nx)
psi_x_temp = np.exp(-0.5*x_temp**2)*np.pi**(-0.25)
test = GPENS(x_temp, psi_x_temp, Nx, 1e-4)

fig = plt.figure()
imp_ani = animation.ArtistAnimation(fig, test.animation(2, 1e-2), interval=50, repeat_delay=3000, blit=True)
plt.show()

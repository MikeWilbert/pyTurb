# pyTurb

## Introduction

In this project we will explore how to simulate 2-dimensional forced turbulence by means of the pseudo-spectral method.  
We will give a coarse introduction to the physical models of 2-dimensional turbulence followed by a discussion on the numerical methods suitable to efficiently simulate this natural phenomenon.  
The numerical methods discussed here are implemented in the python module *pyTurb*, which is available at my [github](https://github.com/MikeWilbert) page. To produce simulations with a practically relevant resolution, *pyTurb* uses the module *cupy* to accelerate the code on a single GPU.

<video width="720" height="720" controls poster="./../images/step_0960.jpg">
      <source src="./../images/movie.mp4" type="video/mp4">
</video> 

Above we see a simulation produced with *pyTurb*. It has a resolution of 1024 points in each spatial direction. For the shown 80 turn over times it took about 75 seconds on a A100 Nvidia GPU. It shows 2-dimensional turbulence forced from rest at a wavenumber of $k_f = 20$.

## pyTurb

### Download

The pyTurb project can be downloaded in the following ways:

ssh: </br>
`git clone https://github.com/MikeWilbert/pyTurb.git`

https: </br>
`git clone git@github.com:MikeWilbert/pyTurb.git`

### Test run

To start a first test run go to the code directory

`cd pyturb_2d/code`

There you will find the python module *pyTurb* as well as an example code that usesthe module to run a simulation. Additionally you will find the files *plot_stats.py* and *plot_spectra.py* which display makroscopic fluid quantities and the energy spectra produced printed during the simulations, respectively.

Before you run the execution script with

`python NS_GPU.py`

you might want to change the output directory as well as the resolution. You can also change other parameters. The patrameters for the simulations are described in detail in the next section.

### Functions

In the following the functions included in *pyTurb* are briefly described.

#### `init(N_, k_a_, k_f_, dk_f_ c_res_, out_dir_)`

Reads the parameters necessary to specify the simulation. This function needs to be called before any other *pyTurb* function.

| parameter | description |
| --- | --- |
| `N_` | # of grid points per direction |
| `k_a_` | linear friction wavenumber |
| `k_f_` | middle of forcing wavenumber band|
| `dk_f_` | width of forcing wavenumber band |
| `c_res_` | ratio of maximum resolved wavenumber to diffusion wavenumber $k_{max} / k_{\nu}$. Good value: 3, acceptable value: 1.5 |
| `out_dir_` | output directory |

#### `step()`

Performs a single time step.

#### `print_vtk()`

Prints the vorticity field in the vtk format to the output directory specified in the `init` function. That format constits of an XML header which describes the following binary data stored in the file. VTK files can be read e.g. by *Paraview* for visualization purposes.

#### `print_spectrum()`

Prints the energy spectrum and the corresponding simulation time to the file 'spectra.csv' that can be found in the output directory specified in the `init` function.

#### `print_stats()`

Prints the mean energy and dissipation rate and the corresponding simulation time to the file 'stats.csv' that can be found in the output directory specified in the `init` function.

<hr>

For an example file to run a simulation with *pyTurb* we refer to the file 'code/NS_GPU.py'

## 2D Turbulence

In this section, the basic ideas and formulas behind the theory of 2-dimensional turbulence will be discussed.

### Vorticity equation

We start by considering the incompressible Navier-Stokes equations with an additional linear friction and a random force:

$$
\begin{align}
\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u} &= - \nabla p + \nu \Delta \mathbf{u} - \alpha \mathbf{u} + \mathbf{f} \\
\nabla \cdot \mathbf{u} &= 0
\end{align}
$$

| term | name | function |
| --- | --- | --- |
| $\mathbf{u}$ | convection / non-linearity | velocity transports itself |
| $\nabla p$ | pressure gradient | enforces incompressibility |
| $\nu \Delta \mathbf{u}$ | viscous diffusion | dissipates energy at small scales |
| $\alpha \mathbf{u}$ | linear friction | removes energy at large scales |
| $\mathbf{f}$ | random force | inserts energy at a prescribed scale |

If we consider only 2-dimensional systems, it is more practical to formulate the incompressible Navier-Stokes equations in terms of the vorticity $\mathbf{\omega} = (\nabla \times \mathbf{u})$. 

Before taking the curl of the momentum equation (1), it is useful to reformulate the non-linearity by applying the identity

$$ \frac{1}{2} \nabla |\mathbf{u}|^2 = \mathbf{u} \cdot \nabla \mathbf{u} + \mathbf{u} \times ( \nabla \times \mathbf{u} )$$

With that we find an evolution equation for the vorticity

$$
\begin{equation}
\partial_t \omega + \nabla \times ( \omega \times \mathbf{u} ) = \nu \Delta \mathbf{\omega} - \alpha \mathbf{\omega}.
\end{equation}
$$

Note that the vorticity is divergence free by definition $\left(\nabla \cdot ( \nabla \times \mathbf{u} ) = 0\right)$. Thus, no pressure term is needed in this formulation.

If we consider only 2 spatial dimensions, equations (3) can be simplified even further.

Assume the fluid velocity $\mathbf{u}$ is restricted to the x-y plane, then the vorticity vector $\mathbf{\omega}$ only has a non-zero component in the z-direction $\mathbf{\omega} = \omega \, \mathbf{\hat{e}}_z$.

Using the vector identity

$$ \nabla \times ( \mathbf{A} \times \mathbf{B} ) = \mathbf{B} \cdot \nabla \mathbf{A} - \mathbf{A} \cdot \nabla \mathbf{B} + \mathbf{A} ( \nabla \cdot \mathbf{B} ) - \mathbf{B} ( \nabla \cdot \mathbf{A} ) $$

and using the solenoidality of the velocity and the vorticity $(\nabla \cdot \mathbf{u} = \nabla \cdot \mathbf{\omega} = 0)$, we can write the non-linear term as

$$ \nabla \times ( \mathbf{ \omega } \times \mathbf{u} ) = \mathbf{u} \cdot \nabla \mathbf{\omega} - \mathbf{\omega} \cdot \nabla \mathbf{u} .$$

In the 2-dimensional case, the first term on the right-hand-side vanishes and we find the 2-dimensional version of the vorticity equation

$$
\begin{equation}
\partial_t \omega + \mathbf{u} \cdot \nabla \mathbf{\omega} = \nu \Delta \omega - \alpha \omega.
\end{equation}
$$

Note that this is only a scalar equation for the z-component of the vorticity.

Since the velocity field is divergence-free, it can also be written as the curl of a vector potential. In the 2-dimensional case this vector potential only has z-component that we will call the stream function $\psi$. As for the vorticity in 2D, we write $\mathbf{\psi} = \psi \mathbf{\hat{e}}_z$.

By expressing the velocity by the stream function $( \mathbf{u} = \nabla \times \mathbf{\psi} )$ we find the stream function formulation of the 2D vorticity equation

$$
\begin{equation}
\boxed{
\partial_t \omega = \nabla \psi \times \nabla \omega + \nu \Delta \omega - \alpha \omega.
}
\end{equation}
$$

Finally, it would be more consistent to express the stream function $\psi$ by the vorticity $\omega$ instead of the velocity vector $\mathbf{u}$.  
This can be achieved by applying the vector identity $\nabla \times ( \nabla \times \mathbf{A} ) = \nabla (\nabla \cdot \mathbf{A}) - \Delta \mathbf{A}$ to the definitions of $\omega$ and $\psi$.

$$ \mathbf{\omega} = \nabla \times \mathbf{u} = \nabla \times ( \nabla \times \mathbf{\psi}) = \nabla (\nabla \cdot \mathbf{\psi}) - \Delta \mathbf{\psi} $$

Due to the solenoidality of the stream function $(\nabla \cdot \mathbf{\psi} = 0)$ we can write

$$ \mathbf{\omega} = - \Delta \mathbf{\psi} $$

In the 2D case $\mathbf{\omega} = \omega \mathbf{\hat{e}}_z$ and $\mathbf{\psi} = \psi \mathbf{\hat{e}}_z$, the latter reads

$$ \omega = - \Delta \psi .$$

Thus, if we solve for the 2-dimensional vorticity field $\omega$, we can use the last relation to compute the corresponding stream function $\psi$.

### Energy balance

Since we are interested in turbulence that is stationary on average, it is useful to derive the energy balance of our system (4).

For that we start by taking the scalar product of the fluid velocity $\mathbf{u}$ with equation (1) and integrate over all space. For simplicity we assume that the fluid domain is either infinite or has periodic boundary conditions, so that we can neglect surface terms.

$$
\begin{equation}
\int d^2x \, \mathbf{u} \cdot \partial_t \mathbf{u} + \mathbf{u} \cdot ( \mathbf{u} \cdot \nabla \mathbf{u} ) = \int d^2x \, - \mathbf{u} \cdot \nabla p + \nu \mathbf{u} \cdot \Delta \mathbf{u} - \alpha \mathbf{u}^2 + \mathbf{u} \cdot \mathbf{f}
\end{equation}
$$

The first term on the left-hand side is the time derivative of the kinetic energy of the fluid

$$ \partial_t \int d^2x \, \frac{1}{2} |\mathbf{u}|^2 .$$

The second term on the left-hand side describes the advection of momentum. Its integrand is the divergence of the momentum flux tensor and thus does not contribute to the kinetic energy

$$ \int d^2x \, \mathbf{u} \cdot ( \mathbf{u} \cdot \nabla \mathbf{u} ) = \int d^2x \, \nabla \cdot \left( \frac{1}{2} |\mathbf{u}|^2 \mathbf{u} \right) = 0 .$$

Similarly, the first term on the right-hand side does not contribute either, as its integrand is the divergence of the pressure

$$ \int d^2x \, - \mathbf{u} \cdot \nabla p = \int d^2x \, \nabla \cdot ( p \mathbf{u} ) = 0 .$$

For the viscous term we have

$$ \nu \int d^2x \, \mathbf{u} \cdot \Delta \mathbf{u} = \nu \int d^2x \, ( \partial_i u_j ) ( \partial_i u_j ) = \nu \int d^2x \, | \nabla \mathbf{u} |^2 = \varepsilon .$$

This describes the dissipation of kinetic energy into heat at small scales and is usually referred to as the energy dissipation rate $\varepsilon$.

Finally, for the forcing term we can write

$$ \int d^2x \, \mathbf{u} \cdot \mathbf{f} = I .$$

This term describes the injection of kinetic energy into the fluid.

Putting everything together we find an energy balance of the form

$$
\begin{equation}
\boxed{
\partial_t \int d^2x \, \frac{1}{2} |\mathbf{u}|^2 = I - \varepsilon - \alpha \int d^2x \, \mathbf{u}^2
}
\end{equation}
$$

In the stationary state, the time derivative on the left-hand side must vanish. The energy input is then balanced by dissipation into heat by viscous forces as well as dissipation by the linear friction term.

### Random forcing

In 2-dimensional turbulence the energy is transported from small to large scales and ultimately dissipated at the largest possible scale by linear friction.

Thus, to study a steady state turbulence, we have to insert energy at a small, controlled scale by means of a random force.

For the numerical implementation of the vorticity equation (4), it is convenient to specify the random force in Fourier space, i.e. at a well-defined length scale.

We start by writing the Fourier transformation of the velocity field

$$ \mathbf{u} (\mathbf{x}, t) = \frac{1}{L^2} \sum_{\mathbf{k}} \mathbf{u}_{\mathbf{k}} (t) \exp \left( i \mathbf{k} \cdot \mathbf{x} \right) $$

The Fourier modes $\mathbf{u}_{\mathbf{k}}$ are complex-valued vectors that fulfill the reality condition $\mathbf{u}_{\mathbf{k}} = \mathbf{u}^*_{-\mathbf{k}}$ so that the velocity field $\mathbf{u} (\mathbf{x}, t)$ is real.

In order to make the velocity field incompressible, the forcing modes have to be perpendicular to the wavevector $\mathbf{k}$,

$$ \mathbf{k} \cdot \mathbf{u}_{\mathbf{k}} = 0 .$$

In our 2-dimensional system, this means we have to specify only one (complex-valued) degree of freedom per forcing mode $\mathbf{k}$.

We write the modes in terms of their absolute value and a phase

$$ \mathbf{u}_{\mathbf{k}} (t) = \sum_{k_0 \leq | \mathbf{k} | < k_0 + \delta k} | \mathbf{u}_{\mathbf{k}} (t) | \exp ( i \phi_{\mathbf{k}} (t) ) \mathbf{e}_{\mathbf{k}} $$

with $\mathbf{e}_{\mathbf{k}} \cdot \mathbf{k} = 0$.

At each forcing step, we draw the phases $\phi_{\mathbf{k}}$ randomly and uniformly from the interval $[ 0, 2\pi )$.

Furthermore, the absolute values of the forcing modes are drawn from a normal distribution

$$ \langle \mathbf{u}_{\mathbf{k}} (t) \mathbf{u}^*_{\mathbf{k}} (t) \rangle = f_0 .$$

Thus, the total forcing amplitude is specified by $f_0$ while the forcing length scale is determined by $k_0$.

## Numerical methods

In this section we will consider the numerical methods used in *pyTurb* to solve equations (5) & (6).  
*pyTurb* is based the Fourier pseudo-spectral approach, which is very suitable for problems on periodic domains with infititely smooths solutions.

### Psuedo-spectral method

The main trick of the Fourier pseudo-spectral method is to exploit the fact that derivatives in physical space become multiplications with the wavevector in Fourier space.  
This can be easily shown for the one-dimensional case.
The basic idea of the Fourier transform is that every complex and integrable function $f(x)$ can be described by a superposition of plane waves, i.e.
$$ f(x) = \int \hat{f}(k) \, \exp(i\,k\,x) \, \text{d}k =: \mathcal{F}^{-1}(\hat{f})(x).$$  
$\mathcal{F}^{-1}$ is known as the *inverse Fourier transform*.
Here, the complex amplitude $\hat{f}$, which includes the amplitude and phase of the corresponding plane wave, is called the *Fourier coefficient* of $f$.  
Since the Fourier basis, i.e. the set consisting of the functions $\exp(i\,k\,x)$ for all real $k$, is orthonogonal, the Fourier coefficients can be expressed as 
$$ \hat{f}(k) = \frac{1}{2\,\pi} \int f(x) \, \exp(-i\,k\,x) \, \text{d}x =: \mathcal{F}(f)(k).$$
$\mathcal{F}$ is then called the *Fourier transform*.  

Now, consider the derivative of a function. Then we find
$$ f'(x) = \frac{\text{d}}{\text{d}x} \int \hat{f}(k) \, \exp(i\,k\,x) \, \text{d}k = \int i\,k\, \hat{f}(k) \, \exp(i\,k\,x) \, \text{d}k = \mathcal{F}^{-1}(i\,k\,\hat{f}(k)) = \mathcal{F}^{-1}(i\,k\,\mathcal{F(f)(k)}) $$
Thus, to take the derivative of f, we can transform to Fourtier space, multiply with $i\,k$ and then transform back to physical space. This procedure can easily be extended to vector valued functions:

$$\begin{align*}
  \mathcal{F}(\nabla \phi(\mathbf{x})) &= i \mathbf{k} \hat{\phi}(k)\\
  \mathcal{F}(\nabla \cdot \mathbf{u}(\mathbf{x})) &= i \mathbf{k} \cdot \hat{\mathbf{u}}(k)\\
  \mathcal{F}(\nabla \times \mathbf{u}(\mathbf{x})) &= i \mathbf{k} \times \hat{\mathbf{u}}(k)\\
  \mathcal{F}(\Delta \mathbf{u}(\mathbf{x})) &= - |\mathbf{k}|^2 \hat{\mathbf{u}}(k)
\end{align*}$$

Note also, that in Fourier space the Poisson equation can be solved quiet easily as the Laplace operator $\Delta$ can be simply inverted by dividing by $-|{k}|^2$.

Since we want to solve equations (5) & (6) on a computer, we are restricted to a finite number spacial samples of $f$ and thus we will deal with a finite number of Fourier modes. This leads from the Fourier integral to the discrete Fourier transform (*DFT*).

Let's consider the finite space interval $L = [0, 2\pi)$ and discretize it by $N$ points in space $x_j$ that are an equal distance $\Delta x = L/N$ apart. By now evalutating $f$ at these discrete points $f_j = f(x_j)$, with $x_j = j \, \Delta x, j = 0, \dots, N-1$, we define the discrete Fourier transform and its inverse as
$$DFT(f_j) := \sum_{j=0}^{N-1} f_j\,\exp(-i\,k\,x) =: \hat{f}_k $$
$$DFT^{-1}(\hat{f}_k) := \frac{1}{N} \sum_{k=-N/2}^{N/2-1} \hat{f}_k\,\exp(i\,k\,x) = f_j$$

The *DFT* assumes that the function $f$ can be described by a finite number of plane waves, which are infinetly smooths, periodic functions. Therefore, also $f$ needs to be periodic and infintely smooth. If that assumption is not fullfilled, it will show in the form of unphysical oscillations or a refelction of unresolvable modes into the spectrum.

The naive computation of the *DFT* gives a number of operations of the order $\mathcal{O}(N^2)$ (every $j$ with every $k$). This will become very expensive in terms of computation time if we want to compute high resolutions in multiple dimensions. 
This unconvencience is overcome by an algorithms called the *Fast Fourier Transform* (*[FFT](10.1090/S0025-5718-1965-0178586-1)*), which achieves a number of operations of $\mathcal{O}(N\,\log N)$, i.e. nearly optimal, by utilizing a divide-and-conquer approach based on the periodicty of the Fourier base.  

So far, the pseudo-spectral method can be summarized as follows: Compute the right-hand-side of equation (6) at discrete equidistant points by transforming the initial data to Fourier space using the FFT and compute the derivatives by multiplications with the wavevector. If the r.h.s. is evaluated this leaves us with an ordinary differential equation in time, for that a great variety of numerical methods exists. Also the stream function can easiy be computed in Fourier space by inverting the Laplace operator.  
The only thing we have not considered yet is the non-linear term. In our case it consists of the multiplication of two gradients. Unfortunately, multiplications in Fourier space become convultions in physical space. In discrete space, this is again an operation of order $\mathcal{O}(N^2)$. This can be avoided by first calculating the derivatives in Fourier space, then transforming to physical space and perform the multiplications there. Since we use the FFT for the transformations, we are back at $\mathcal{O}(N \log N)$.  
The main idea of the pseudo-spectral method can be summarized as follows:  
Compute derivatives in Fourier space, calculate multiplications in real space and transform between those two views by the efficient FFT.

### Dealiasing

Another issue associated with the multiplication in real space is that we need to double the number of Fourier modes to properly represent the result.
$$\begin{align*}
  f_j \, g_j &= \left( \sum_{k=-N/2}^{N/2-1} \hat{f}_k \exp( i\,x_j \, k ) \right) \, \left( \sum_{m=-N/2}^{N/2-1} \hat{g}_m \exp( i\,x_j \, m ) \right) \\
  &= \sum_{k,m=-N/2}^{N/2-1} \hat{f}_k \hat{g}_m \exp( i\,x_j \, (k+m) ) \\
\end{align*} 
$$
Thus, the resulting interval of discrete wavenumbers will be $[-2 N/2, 2 (N/2-1)]$. But we will surely not double the spatial resultion after every multiplication. Keeping the resultion fixed to the wavenumbers $[-N/2, N/2-1]$ then results in the unresolved wavenumbers being refected back into the other side of the resolvable spectrum, producing erroneous results. This can be seen by the following consideration.
$$ \begin{align*}
  
 \exp( i\, x_j\, k ) &= \exp( i\, 2\pi (j\,k) / N) \cdot 1 = \exp( i\, 2\pi j\,k / N) \cdot \exp( \pm i\, 2\pi \,  j ) = \exp( i\, 2\pi (j\, (k \pm N) / N) \\
 &= \exp( i\, x_j\, ( k \pm N ) )
 \end{align*}$$

E.g. if we have $N = 8$, the resolvable wavenumbers are in the discrete interval $[-4, 3]$. The wavenumber interval produced by the multiplication is $[-8, 6]$. The amplitude $\hat{f}_5$ will then be added to the amplitude $\hat{f}_{-1}$.
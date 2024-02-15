# PyTurb_2D

Python Code simulating Navier-Stokes / MHD turbulence on a single GPU using the pseudo-spectral method.

## Project Description

This project aims at creating a tool to experiment with different techiques before implementing them into larger frameworks (e.g. SpecTurb, MuPhy2).

## Problem
> $\partial_t \omega = \mathbf{u} \cdot \nabla \omega + \nu \Delta \omega - \alpha \omega$  
> $\mathbf{u} = - \Delta^{-1} [ \nabla \times (\mathbf{\omega \, \hat{e}_z}) ]$

- $\omega$ : vorticity
- $\mathbf{u}$ : velocity
- $\nu$ : viscosity
- $\alpha$ : linear friction coefficient

## Scales

### Input parameters

>- $N$ : Resolution per direction
>- $\epsilon$ : Energy rate put into the system
>- $k_f$ : Forced mode
>- $k_\alpha$ : Smallest mode (integral scale) due to linear forcing >restricting the inverse cascade
>- $c_{res}$ : Ratio of maximum wavenumber due to resolution $k_{max}$ and wavenumber of disspation scale $k_\nu$

### Derived quantities

#### maximum wavenumber
>$k_{max} = \frac{1}{3} N$ [2/3-dealiasing] or   
>$k_{max} = \frac{2}{5} N$ [Hou&Li-dealiasing]

#### spatial resolution
>$\frac{k_{max}}{k_\nu} = c_{ref} \rightarrow \nu = c_{ref}²\,\epsilon^\frac{1}{3}\,k_f^\frac{2}{3}\,k_{max}^{-\frac{1}{2}}$

#### friction force
>$k_\alpha = \epsilon^{-\frac{1}{2}}\,\alpha^{\frac{3}{2}} \rightarrow \alpha = k_\alpha^{\frac{2}{3}}\,\epsilon^{\frac{1}{3}}$
# pyTurb_2D

## Setup

### Connect to and setup lars
`ssh [user]@davinci.gpucluster.ruhr-uni-bochum.de`

`module purge` </br>
`module load nvhpc` </br>
`module load anaconda` </br>

### Install python libs

`conda create --name [name]` </br>
`conda activate [name]` </br>
`conda install [lib_name]`

- numpy
- cupy
- pandas
- shutil
- scipy
- pyevtk

E.g. `conda install cupy`

### Download pyTurb_2D

ssh: </br>
`git clone git@gitlab.tp1.ruhr-uni-bochum.de:mw/pyturb_2d.git`

https: </br>
`git clone https://gitlab.tp1.ruhr-uni-bochum.de/mw/pyturb_2d.git`

### Make test run

- `cd pyturb_2d/code`
- Change path to desired output directory in *NS_GPU.py*
- run script with `python NS_GPU.py`1
- Additionally, change parameters in *NS_GPU.py* </br>(E.g. k_f or k_a)

## 2D Turbulence

### Vorticity equation

Incompressible Navier-Stokes equations with additional linear friction force:

$$  \begin{align}
  
 \partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u} = - \nabla p &+ \nu \Delta \mathbf{u} - \alpha \mathbf{u} \\ \nabla \cdot \mathbf{u} &= 0 
\end{align}$$

| term | name | function |
| --- | --- | --- |
|  $\mathbf{u}$ | convection / non-linearity | veclocity transports itself
|  $\nabla p$ | pressure gradient | enforces incompressibility |
|  $\nu \Delta \mathbf{u}$ | viscous diffusion | dissipates energy at small scales |
|  $\alpha \mathbf{u}$ | linear friction | removes energy at large scales |

If we consider only 2-dimensional systems, it is more practical to formulate the incompressible Navier-Stokes equations in terms of the vorticity $\mathbf{\omega} = (\nabla \times \mathbf{u})$. 

Before taking the curl of the momentum equation (1), it is usefull to reformulate the non-linearity by applying the identity

$$ \frac{1}{2} \nabla |\mathbf{u}|^2 = \mathbf{u} \cdot \nabla \mathbf{u} + \mathbf{u} \times ( \nabla \times \mathbf{u} )$$

With that we find an evolution equation for the vorticity

$$\begin{equation}
   \partial_t \omega + \nabla \times ( \omega \times \mathbf{u} ) = \nu \Delta \mathbf{\omega} - \alpha \mathbf{\omega}.
  \end{equation}$$


Note that the vorticity is divergence free by definition $\left(\nabla \cdot ( \nabla \times \mathbf{u} ) = 0\right)$. Thus, no pressure term is needed in this formulation.

If we consider only 2 spatial dimensions, equations (3) can be simplified even further.

Assuming the fluid velocity $\mathbf{u}$ restricted to the x-y plane, then the vorticity vector $\mathbf{\omega}$ only has a non-zero component in the z-direction $\mathbf{\omega} = \omega \, \mathbf{\hat{e}}_z$.

Using the vector identity

$$ \nabla \times ( \mathbf{A} \times \mathbf{B} ) = \mathbf{B} \cdot \nabla \mathbf{A} - \mathbf{A} \cdot \nabla \mathbf{B} + \mathbf{A} ( \nabla \cdot \mathbf{B} ) - \mathbf{B} ( \nabla \cdot \mathbf{A} ) $$

and using the solenoidality of the velocity and the vorticity $(\nabla \cdot \mathbf{u} = \nabla \cdot \mathbf{\omega} = 0)$, we can write the non-linear term as

$$ \nabla \times ( \mathbf{ \omega } \times \mathbf{u} ) = \mathbf{u} \cdot \nabla \mathbf{\omega} - \mathbf{\omega} \cdot \nabla \mathbf{u} .$$

In the 2-dimensional case, the first term on the right-hand-side vanishes and we find the 2-dimensional version of the vorticity equation

$$\begin{equation}
   \partial_t \omega + \mathbf{u} \cdot \nabla \mathbf{\omega} = \nu \Delta \omega - \alpha \omega.
  \end{equation}$$

Note that this is only a scalar equation for the z-component of the vorticity.

Since the velocity field is divergence-free, it can also be written as the curl of a vector potential. In the 2-dimensional case this vector potential only has z-component that we wil call the stream function $\psi$.
As for the vorticity in 2D, we write $\mathbf{\psi} = \psi \mathbf{\hat{e}}_z$.

By expressing the veclocity by the stream function $( \mathbf{u} = \nabla \times \mathbf{\psi} )$ we find the stream fruntion formulation of the 2D vorticity equation

$$\begin{equation}
\boxed{
   \partial_t \omega = \nabla \psi \times \nabla \omega + \nu \Delta \omega - \alpha \omega.
}
\end{equation}$$

Finally, it would be more consistent to express the stream function $\psi$ by the vorticity $\omega$ instead of the velocity vector $\mathbf{u}$.  
This can be achieved by applying the vector identity $\nabla \times ( \nabla \times \mathbf{A} ) = \nabla (\nabla \cdot \mathbf{A}) -  \Delta \mathbf{A}$ to the definitions of $\omega$ and $\psi$.

$$ \mathbf{\omega} = \nabla \times \mathbf{u} = \nabla \times ( \nabla \times \mathbf{\psi}) = \nabla (\nabla \cdot \mathbf{\psi}) -  \Delta \mathbf{\psi} $$

Since the $\mathbf{\psi}$ only has one component in the z-direction and is restricted to the x-y plane, it has zero divergence and we end up with

$$ \begin{equation}
  \boxed{ \omega = - \Delta \psi }
\end{equation} $$

Equations (5) and (6) are the equations we are interested to solve.  

### A glance at turbulence

In 3-dimensional turbulence, the theory by Kolmogorov from 1941 is still most prominent.  
In a nutshell, it states that if energy is put into the system by some forcing mechanism, the non-linearity breaks up the eddies of this inertial scale into smaller eddys, which by themselfes break up into even smaller structures and so on. This is known as the *Richardson cascade*.
The Reynolds number $Re$, which denotes the ratio of the non-linear term to the diffusion, is typically a very high number, so that diffusion can be neglected in the cascade. But since $Re$ scales with the cosidered length scale, after a sufficient number of break ups, the eddies will be so small, that they feel the viscous friction and theit energy dissipates into heat.  
Kolmogorov predicted a kinetic energy spectrum of the form
$$ E(k) \propto k^{-5/3} $$
and also gave the spacial and temporal scales where the diffusion sets in.  
For a more detailed and quantitative discussion about 3D turbulence, please consider the book by [Lautrup](https://www.lautrup.nbi.dk//continuousmatter2/index.html) or any other basic fluid dynamics book.

In the 2-dimensional case, the turbulenct cascade changes its character in the two ways. The basic theory for 2D turbulence was created by [Kraichnan](https://pubs.aip.org/aip/pfl/article-abstract/10/7/1417/440889/Inertial-Ranges-in-Two-Dimensional-Turbulence?redirectedFrom=fulltext) in is 1967 paper.  
First of all, the energy is transported to larger scales instead of smaller scales. This is called the inverse cascade. The inverse energy cascade obeys the same power law as in 3D Kolmogorov theory with a coefficient of $5/3$. Since diffusion will only remove energy from the system at small scales, there is a need for an energy sink at large scales. In numerical simulations it is therefor common to introduce a linear friction term, which performs this task.  
The other special feature in 2D turbulence is the enstrophy cascade. Enstrophy is the analog to energy for the voriticvity, that is some measure of swirliness. This is a downward cascade and scales approximately as
$$ E(k) \propto k^{-3}, $$
although this slope should not be considered too strict, as there are also other theoretical predictions, e.g. slopes with $-4$ or $-11/3$. Kraichnan himself wrote it is original paper that the $-3$ slope "must be modified by factors with logarithmic $k$ dependence". For a discussion on the slope of the enstrophy cascade we refer to sections 9 & 10 of the review by [Tabling](https://www.sciencedirect.com/science/article/pii/S0370157301000643).

### Turbulence scales

Since numerical simulations are restricted to a finite range of scales, it is very important to know the relevant spatial scales that have to be resolved. In the case of 2D turbulence, there are 3 important scales.  
There is the *foring scale* associated with a wavenumber $k_f$ at which the energy is injected into the system. This scale depends on the choice of the forcing term.  
Then there is the *dissipation scale*, at which diffusion sets in and removes enstrophy from the system. This scale is associated with the diffusion term and especially with the viscosity $\nu$.  
Finally, we have the *friction scale* at which energy is removed from the system at large wavenumbers at the end of the inverse cascade. This is due to the linear friction force and will be a function of the friction coefficient $\alpha$.

#### Friction scale

Considering a differenctial equation, where the velocity is only changed by the linear friction force
$$ \partial_t \mathbf{u} = - \alpha \mathbf{u} $$
we find a solution of the form
$$\mathbf{u}  \propto e^{- \alpha t}.$$
Here we see, that the coefficient $\alpha$ acts as the inverse of the time scale the linear friction acts on.  
By conservation of energy (see e.g. the original paper by Kraichnan) the maximum spatial scale reached by the inverse cascade as a function of time $t$ is given by (Kraichnan, 1967)
$$L(t) \approx \epsilon^{1/2} t^{3/2}.$$
$\epsilon$ denotes energy dissipation rate.  
Setting $\alpha$ as the inverse time in this formula leads an expression for the friction scale
$$k_\alpha \approx \frac{1}{L_\alpha} \approx \epsilon^{-1/2} \alpha^{3/2}.$$
By means of that relation we can specify the friction scale by setting the friction coefficient as
$$\begin{equation}
  \boxed{\alpha = \epsilon^{1/3} k_\alpha^{3/2}}.
\end{equation}$$
The energy dissipation $\epsilon$ must be equal to the energy production rate induced by the forcing, which we can specify explcitely by construction by the forcing term, as we will see later in this document.  

#### Dissipation scale  

In the original paper by Kraichnan we find an expression for the dissipation scale
$$k_\nu = \eta^{1/6} \nu^{-1/2}.$$
By assuming the relation between the dissipation rate of energy $\epsilon$ and the entsrophy dissipation rate $\eta$
$$\eta \approx k_f^2 \, \epsilon,$$
which makes sense, when we look at the definition of those two quantities by their relation to the enrgy spectrum, we find an expresion for the viscosity $\nu$ as a function of the forcing wavenumber $k_f$ and the dissipation scale $k_\nu$.
$$\boxed{\nu = \epsilon^{1/3} k_f^{2/3} k_\nu^{-2}}$$

In summary, we can provide the three relevant scales $k_\alpha$,  $k_f$ and $k_\nu$ and use the above relations to scale the forces in such a way that (...)
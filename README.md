![Tests](https://github.com/turczyneq/pypesh/actions/workflows/test.yml/badge.svg)

# pypesh

This repository attempts to find a solution to advection diffusion problem

$$ 0 = \Delta \varphi - \mathrm{Pe} (u \cdot \nabla \varphi) $$

with $\varphi = 1$ for $z \to \infty$ and $\varphi = 0$ on a surface of the sphere and $\mathrm{Pe}$ denoting Peclet number. Final determined value is Sherwood number defined as 

$$ \mathrm{Sh} = \frac{\Phi}{ 4 \pi D R}$$

Where $D$ is diffusion constant and $\Phi$ is flux falling onto the sphere.

<p align="center">
  <img src="examples/graphics/scheme_sde_pde.png" alt="Both approaches solving the same problem">
</p>

(a) shows the flow field around the sphere. We use two approaches: (c) `pychastic` to generate and trace trajcetories of single particles and estimate the probability of hitting, which allows to calculate sherwood number. This however is expenive in time, so for smaller $\mathrm{Pe}$ we used (b) `scikit-fem` package to handle solving which requires rewriting equations in weak form.

# How to cite

If you use **pypesh** in your research, please cite the associated article:

Turczynowicz, J., Waszkiewicz, R., Lisicki, M., and Słomka, J. 
*Bridging advection and diffusion in the encounter dynamics of sedimenting marine snow*; arXiv (2025)  

https://doi.org/10.48550/arXiv.2504.08992

```bibtex
@article{Turczynowicz_2025,
  title     = {Bridging advection and diffusion in the encounter dynamics of sedimenting marine snow},
  author    = {Turczynowicz, Jan and Waszkiewicz, Radost and Lisicki, Maciej and Słomka, Jonasz},
  journal   = {arXiv},
  year      = {2025},
  doi       = {10.48550/arXiv.2504.08992},
  url       = {https://doi.org/10.48550/arXiv.2504.08992}
}
```

# Examples

## Usage as module

Basic usage
```Python
python3 -m pypesh --peclet 1000 --ball_radius 0.9

```

Sample output:
```
Sherwood for given parameters is 12.033892568100546
```

## Usage as package

Install

```Bash
python3 -m pip install pypesh
```

Basic usage

```Python
import pypesh.pesh as psh
psh.sherwood(peclet = 10**4, ball_radius = 0.9)
```

For advanced options go to: https://pypesh.readthedocs.io/en/latest/

## License
Copyright (C) 2024  Radost Waszkiewicz and Jan Turczynowicz.
This repository is published under GPL3.0 license.

## Bibliography
 - *Bubbles, Drops and Particles*; R. Clift, J. Grace, M. Weber (1978)
 - *Electrochemical measurements of mass transfer between a sphere and liquid in motion at high Peclet numbers*; S. Kutateladze, V. Nakoryakov, M. Iskakov (1982)
 - *Mass and heat transfer from fluid spheres at low Reynolds numbers*; Z. Feng, E. Michaelides (2000)
 - *Heat transfer from spheres to flowing media*; H. Kramers (1946)

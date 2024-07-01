FEM advection diffusion solver
==============================

This repository attempts to find a solution to advection diffusion problem
$$
0 = \Delta \phi - \mathrm{Pe} (u \cdot \nabla \phi)
$$
with $\phi = 1$ for $z \to \infty$ and $\phi = 0$ on a surface of the sphere and $\mathrm{Pe}$ denoting Peclet number.

![Some solution of advection diffusion type problem](/graphics/sample_image.png)

We use `scikit-fem` package to handle solving which requires rewriting equations in weak form.

License
-------
Copyright (C) 2024  Radost Waszkiewicz and Jan Turczynowicz
This repository is published under GPL3.0 license
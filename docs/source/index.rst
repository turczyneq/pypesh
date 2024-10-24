***********************
pypesh Documentation
***********************

``pypesh`` is a repository that attempts to find a solution to advection diffusion problem

.. math::
   
   0 = \Delta \varphi - \mathrm{Pe} (u \cdot \nabla \varphi),

with :math:`\varphi = 1` for :math:`z \to \infty` and :math:`\phi = 0` on a surface of the sphere and :math:`\mathrm{Pe}` denoting Peclet number. Final value determining the flux of :math:`\varphi` onto the sphere is the Sherwood number defined as 

.. math::
   \mathrm{Sh} = \frac{\Phi}{ 4 \pi D R}.

Where :math:`D` is diffusion constant and :math:`\Phi` is flux falling onto the sphere.


How to install
======================

.. prompt:: bash $ auto

  $ python3 -m pip install pypesh

and you'll be good to go.


Package contents
=====================

Analytic
------------------------------

.. automodule:: pypesh.analytic
   :members:

Stokes Flow
------------------------------
For comparison check: https://en.wikipedia.org/wiki/Stokes%27_law

.. automodule:: pypesh.stokes_flow
   :members:

Generate Mesh
------------------------------
For package: https://scikit-fem.readthedocs.io/en/latest/

.. automodule:: pypesh.generate_mesh
   :members:

Finite Element Method
------------------------------
Using package: https://scikit-fem.readthedocs.io/en/latest/

.. automodule:: pypesh.fem
   :members:

Trajectories approach
------------------------------
Using package: https://pychastic.readthedocs.io/en/latest/

.. automodule:: pypesh.trajectories
   :members:

Pe vs Sh
------------------------------
Final package ``pesh`` that calculates the value of :math:`Sh` for given Peclet and radius of rigid ball.

.. automodule:: pypesh.pesh
   :members:
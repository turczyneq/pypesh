import pychast.generate_trajectories as gen_traj
import pychast.process_trajectories as proc_traj
import udajki as loc
import numpy as np

def distribution(peclet,
    ball_radius,
    mesh_out = 4,
    mesh_jump = 6,
    trials=10**2,
    floor_h=5,
    spread = 4,):
    #define the spread of testing range

    r_syf = loc.radius_of_streamline(floor_h, ball_radius)

    disp = loc.dispersion(peclet)

    #generate the mesh to calculate integrals
    if r_syf - disp > 0:
        x_probs = list(np.linspace(max(r_syf - spread*disp,0), r_syf - disp, mesh_out))
    else:
        x_probs = [0]
    x_probs = x_probs + list(np.linspace(max(r_syf - disp,0), r_syf + disp, mesh_jump))
    x_probs = x_probs + list(np.linspace(r_syf + disp, r_syf + spread*disp, mesh_out))


    def fun(x):
        return gen_traj.hitting_propability_at_x(x, peclet, ball_radius, trials = trials)

    sol_dict = {x: fun(x) for x in x_probs}

    '''
    now perform simpe numerical integration, first assume from 0 to max(r_syf-spread*disp,0) propability is one
    '''
    integral = loc.internal_integral(ball_radius, floor_h, max(r_syf - spread*disp,0))

    '''
    then integrate with Trapezoidal approach
    '''

    step = x_probs[1]-x_probs[0]
    def vz(x):
        ver, vez = loc.velocities(x, floor_h, ball_radius)
        return vez

    def int_fun(x):
        return 2*np.pi*x*vz(x)*sol_dict[x]

    for i in range(len(x_probs)-1):
        x0 = x_probs[i]
        x1 = x_probs[i+1]
        integral = integral + (int_fun(x0) + int_fun(x1))*(x1-x0)/2

    # def fun(x):
    # value = gen_traj.hitting_propability_at_x(x, peclet, ball_radius, trials = trials)
    # ver, vez = loc.velocities(x, floor_h, ball_radius)
    # return 2 * np.pi * x * value * vez

    return loc.sherwood_from_flux(integral, peclet), list(sol_dict.keys()), list(sol_dict.values())

# def sherwood_from_peclet(
#     peclet,
#     ball_radius,
#     trials=100,
#     r_mesh=0.1,
#     floor_r=5,
#     floor_h=5,):
        
#     single_pe_trajectories = gen_traj.generate_trajectories_multiply(
#     peclet,
#     ball_radius,
#     trials=trials,
#     r_mesh=r_mesh,
#     floor_r=floor_r,
#     floor_h=floor_h,)

#     sh_number = proc_traj.sherwood(single_pe_trajectories,peclet,ball_radius,floor_h = floor_h)

#     return sh_number
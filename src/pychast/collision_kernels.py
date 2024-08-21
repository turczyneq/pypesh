import pychast.generate_trajectories as gen_traj
import pychast.process_trajectories as proc_traj

def sherwood_from_peclet(
    peclet,
    ball_radius,
    trials=100,
    r_mesh=0.1,
    floor_r=5,
    floor_h=5,):
        
    single_pe_trajectories = gen_traj.generate_trajectories(
    peclet,
    ball_radius,
    trials=trials,
    r_mesh=r_mesh,
    floor_r=floor_r,
    floor_h=floor_h,)

    sh_number = proc_traj.sherwood(single_pe_trajectories,peclet,ball_radius,floor_h = floor_h)

    return sh_number
import pychast.generate_trajectories as gen_traj
import pychast.process_trajectories as proc_traj
import udajki as loc
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm as tqdm


def distribution(peclet,
    ball_radius,
    mesh=10,
    trials=10**2,
    floor_h=5,):

    r_syf = loc.radius_of_streamline(floor_h, ball_radius)

    disp = loc.dispersion(peclet)

    x_probs = np.linspace(max(r_syf-5*disp,0), r_syf+5*disp, mesh)

    print(x_probs)

    def fun(x):
        value = gen_traj.hitting_propability_at_x(x, peclet, ball_radius, trials = trials)
        return value

    to_ret = []
    for x in tqdm.tqdm(x_probs):
        to_ret = [] + [fun(x)]

    # def fun(x):
    # value = gen_traj.hitting_propability_at_x(x, peclet, ball_radius, trials = trials)
    # ver, vez = loc.velocities(x, floor_h, ball_radius)
    # return 2 * np.pi * x * value * vez

    return x_probs, to_ret

def sherwood_from_peclet(
    peclet,
    ball_radius,
    trials=100,
    r_mesh=0.1,
    floor_r=5,
    floor_h=5,):
        
    single_pe_trajectories = gen_traj.generate_trajectories_multiply(
    peclet,
    ball_radius,
    trials=trials,
    r_mesh=r_mesh,
    floor_r=floor_r,
    floor_h=floor_h,)

    sh_number = proc_traj.sherwood(single_pe_trajectories,peclet,ball_radius,floor_h = floor_h)

    return sh_number

def visualise_trajectories(
    peclet,
    ball_radius,
    floor_r = 0.3,
    floor_h = 5,
    r_mesh = 0.01,
    trials = 10,
    display_traj = 10,
    t_max = None):

    initial = gen_traj.construct_initial_condition(
        floor_r=floor_r, floor_h=floor_h, r_mesh=r_mesh, trials=trials
    )
    
    def drift(q):
        return gen_traj.stokes_around_sphere(q, ball_radius)

    collision_data = gen_traj.simulate_with_trajectories(
        drift=drift,
        noise=gen_traj.diffusion_function(peclet=peclet),
        initial=initial,
        floor_h=floor_h,     
        t_max = t_max   
    )

    #plt.figure(figsize=(12, 10))

    trajectories = collision_data["trajectories"]
    for i in range(display_traj):
        r = (trajectories[i, :, 0] ** 2 + trajectories[i, :, 1] ** 2) ** 0.5
        z = trajectories[i, :, -1]

        if collision_data["ball_hit"][i]:
            color = "#2a2"
        elif collision_data["something_hit"][i]:
            color = "#aa2"
        else:
            color = "#a22"

        plt.plot(r, z, color=color)
        plt.scatter(r[-1], z[-1], s=8, color="k", zorder=5)
        plt.scatter(r[0], z[0], s=8, color="k", zorder=5)

    plt.gca().add_artist(plt.Circle((0, 0), 1, color="#222", fill=False))
    plt.gca().add_artist(plt.Circle((0, 0), ball_radius, color="#666", fill=False))

    plt.text(1,-1, f"Pe = 10^{int(np.log10(peclet))}")

    plt.gca().set_aspect("equal")
    plt.axis([0, 2, -2, 0])

    # plt.savefig("traj_visualize.svg", format='svg')
    plt.show()


distribution(10**8, 0.999, trials = 10**3)

# print(gen_traj.hitting_propability_at_x(0.0005, 10**9, 0.999, trials = 400))

# pe_list = [10**i for i in range(3,10)]

# ball_list = []
# for i in range(-3,0):
#     ball_list = ball_list + [(10**i)*ball for ball in [1, 2, 5]]
# ball_list = ball_list[:-1]


# output_file = f"numerical_results/pych_test.txt"
# with open(output_file, 'w') as f:
#     f.write("Peclet\tball_radius\txargs\tsolutions\n")

# for j in range(len(ball_list)):
#     for n in range(len(pe_list)):
#         peclet = pe_list[n]
#         ball_radius = 1 - ball_list[j]
#         print(f"radius = {ball_radius}, peclet = {peclet}")
#         xargs, sol = distribution(peclet, ball_radius, trials = 10**4)
#         with open(output_file, 'a') as f:
#             f.write(f"{peclet}\t{ball_list[j]}")
#             for arg in xargs:
#                 f.write(f"\t{arg}")
#             for arg in sol:
#                 f.write(f"\t{arg}")
#             f.write(f"\n")


# print(f"{sherwood_from_simpson(100000, 0.999, trials = 400)}, {sherwood_from_peclet(100000, 0.999, trials = 400, floor_r = 0.001*50, r_mesh = 0.001*50/50)}")
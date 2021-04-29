""" Evaluating regret of methods in simulation study """

from __future__ import division
import numpy as np

from BanditPricing import *
from simulationFuncs import *
from OptimisticPricing import *
import barriers as br

#TODO: set constants for MOPOL

seed = 123
np.random.seed(seed)
tol = 1e-10  # small numerical factor
nrep = 10
N = 100  # number of products
d = 10  # intrinsic low-rank of demand-variation, must be <= N
s_radius = 20.0  # prices are selected within centered ball with this radius
T = 10000  # the number of rounds (must be divisible by 10)

# Simulation parameters for stationary setting:
sparseU = True
noise_std = 10.0
z_scale = 20.0
z_loc = 100.0
V_scale = 2.0
C = 1.0
eigen_threshold = 10.0
regret_skip = int(T / 20)  # definitely calculate regret every regret_skip rounds.
summary_times = []  # additional rounds where regret should be computed.
for t in range(T):
    if ((t + 1) % regret_skip == 0) or (t == 0) or (t <= 1000 and (t + 1) % 200 == 0):
        summary_times += [t + 1]

#method_order = ['$\mathrm{Explo}^\mathrm{re}_\mathrm{it}$', 'GDG', 'OPOK', 'OPOL']
method_order = ['OPOL', 'MOPOL']

num_methods = len(method_order)  # Specifies which of these methods to run

colors = ['black', 'blue', 'green', 'red']
regrets = np.zeros(
    (num_methods, len(summary_times), nrep))  # num_methods x T/regret_skip x nreps array of regrets for each method.
p_dists = np.zeros((num_methods, len(summary_times),
                    nrep))  # num_methods x T/regret_skip x nreps array of ||p_t - p_star|| for each method.
x_dists = np.zeros((num_methods, len(summary_times),
                    nrep))  # num_methods x T/regret_skip x nreps array of ||x_t - x_star|| for each method.
revenues = np.zeros((num_methods, T, nrep))  # num_methods x T x nreps array of revenues for each method.

# When reps are across same underlying stationary model:
if sparseU:
    U = OrthogonalSparse(N, d)
else:
    U = OrthogonalGaussian(N, d)  # same model for all reps.

V = forcePosDef(np.random.normal(size=d * d, scale=V_scale).reshape((d, d)), eigen_threshold)
z = np.abs(np.random.normal(loc=z_loc, scale=z_scale, size=d))
p_star, R_star = optimalPriceFast(z_list=[z], V_list=[V], U=U, s_radius=s_radius, max_iter=1e4)
x_star = np.dot(U.transpose(), p_star)  # optimal low-dimensional action.

# Run algorithms:
for rep in range(nrep): #number of simulations
    regret_index = 0
    p_init = np.zeros(N)  # initial price configuration
    init_demands = generateDemands(p_init, U, z, V, noise_std)
    init_revenue = negRevenue(p_init, init_demands)
    if np.linalg.norm(p_star) > s_radius + tol:
        raise ValueError("Infeasible price found by SLSQP optimizer")
    print("Inital revenue: " + str(init_revenue))
    print("||p_init - p_star||=" + str(np.linalg.norm(p_star - p_init)))
    # Estimate bound for revenue:
    upper_rev_estimate = 0.0
    for i in range(1000):
        p_rand = randomPricing(N, s_radius)
        R_rand = negRevenue(p_rand, generateDemands(p_rand, U, z, V, noise_std))
        if np.abs(R_rand) > upper_rev_estimate:
            upper_rev_estimate = np.abs(R_rand)
    R_bound = max(1.0, 1.1 * np.abs(R_star), 1.1 * upper_rev_estimate)
    Lipshitz = np.linalg.norm(np.dot(U, z)) + 2 * s_radius * np.linalg.norm(np.dot(U, np.dot(V, U.transpose())), ord=2)
    for t in range(T):
        init_optBCO = False
        t_touse = T
        if t == T / 3 or t == (2 * T) / 3:  # Shcoks
            init_demands = generateDemands(p_init, U, z, V, noise_std)
            init_revenue = negRevenue(p_init, init_demands)
        if (t + 1) in summary_times:
            print('rep=' + str(rep) + '  round=' + str(t + 1))
            if t > 0:
                regret_index += 1
        for method in range(num_methods):
            if method == 0:  # OPOL pricing
                eta = C * s_radius / (R_bound * np.sqrt(t_touse))
                delta = min(0.1 * s_radius, np.power(t_touse, -1 / 4) * np.sqrt(
                    R_bound * d * np.square(s_radius) / (3 * (Lipshitz * s_radius + R_bound))))
                if t > 0:
                    p_t, opol_state = OPOL(opol_demand_prev, eta, delta, s_radius, opol_state)
                else:
                    inferred_rank = d  # run with correctly specifed rank, can change this value to see the effects of wrongly-inferred rank.
                    p_t, opol_state = OPOL(init_demands, eta, delta, s_radius, firstOPOLstate(inferred_rank, p_init))
            elif method == 1:  # MOPOL pricing
                barrier, hessian = br.ball_barrier_20, br.hessian_ball_20()

                eta = C * np.power(t_touse, -3 / 4) * np.power(d, -1 / 2) / (
                    R_bound * s_radius * np.sqrt((1 + s_radius) * (3*s_radius + 2)))
                #TODO: fix barrier
                delta = np.power(t_touse, -1 / 4) * np.power(d, 1 / 2) * (
                    np.sqrt((3*s_radius + 2) * (1 + s_radius)/((2*s_radius + 1) ** 2)))
                k = C * (2* s_radius+ 1)/ ((3* s_radius + 2) * np.sqrt(s_radius * R_bound * (1+ s_radius)))
                # k = np.rint(k)
                if not init_optBCO: #  one time switch
                    if np.min(np.sum(np.abs(mopol_state[1]), axis=0)) != 0.0: init_optBCO = True
                        # set first pt for optimisticBCO
                        mopol_state[0] = np.randUnitVector(d) * s_radius/2
                if t > 0:

                    p_t, mopol_state = MOPOL(mopol_demand_prev, eta, delta, k,
                                            s_radius, mopol_state, barrier, hessian)
                else:
                    inferred_rank = d  # run with correctly specifed rank, can change this value to see the effects of wrongly-inferred rank.
                    p_t, mopol_state = MOPOL(init_demands, eta, delta, k,
                                            s_radius, firstMOPOLstate(inferred_rank, p_init, k), barrier, hessian)

            # OTHER UNNEEDED METHODS
            # if method == 0:  # Uniformly random pricing strategy:
            #     # p_t = randomPricing(N,s_radius)
            #     if t > 0:
            #         p_t, expexp_state = exploreExploitPricing(revenues[method, t - 1, rep], T, s_radius, expexp_state)
            #     else:
            #         p_t, expexp_state = exploreExploitPricing(init_revenue, T, s_radius,
            #                                                   (np.zeros(N), np.inf, 0, np.zeros(N)))
            # elif method == 1:  # GDG pricing:
            #     eta = C * s_radius / (R_bound * np.sqrt(t_touse))
            #     delta = min(0.1 * s_radius, np.power(t_touse, -1 / 4) * np.sqrt(
            #         R_bound * N * np.square(s_radius) / (3 * (Lipshitz * s_radius + R_bound))))
            #     if t > 0:
            #         p_t, gdg_state = GDG(revenues[method, t - 1, rep], eta, delta, s_radius, gdg_state)
            #     else:
            #         p_t, gdg_state = GDG(init_revenue, eta, delta, s_radius,
            #                              (p_init, randUnitVector(N)))  # set initial state.
            # elif method == 2:  # OPOK pricing
            #     eta = C * s_radius / (R_bound * np.sqrt(t_touse))
            #     delta = min(0.1 * s_radius, np.power(t_touse, -1 / 4) * np.sqrt(
            #         R_bound * d * np.square(s_radius) / (3 * (Lipshitz * s_radius + R_bound))))
            #     if t > 0:
            #         p_t, opok_state = OPOK(revenues[method, t - 1, rep], eta, delta, s_radius, U, opok_state)
            #     else:
            #         p_t, opok_state = OPOK(init_revenue, eta, delta, s_radius, U, (
            #         np.dot(U.transpose(), p_init), randUnitVector(d), p_init))  # set initial state.

            else:
                raise ValueError('num_methods=' + str(num_methods) + ' is too large!')
            if np.linalg.norm(p_t) > s_radius + tol:
                raise ValueError("Infeasible price chosen by method " + str(method))
            q_t = generateDemands(p_t, U, z, V, noise_std)  # feed in vector of observed demands here.
            revenues[method, t, rep] = negRevenue(p_t, q_t)
            x_t = np.dot(U.transpose(), p_t)

            if method == 0:  # OPOL pricing
                opol_demand_prev = q_t
            if method == 1:  # MOPOL pricing
                mopol_demand_prev = q_t

            if (t + 1) in summary_times:
                regrets[method, regret_index, rep] = np.sum(revenues[method, :, rep]) - R_star * (
                            t + 1)  # *(t+1) only needed for stationary underlying model.
                p_dists[method, regret_index, rep] = np.linalg.norm(p_star - p_t)
                x_dists[method, regret_index, rep] = np.linalg.norm(x_star - x_t)
                # print(method_order[method] + " dist to best: " + str(np.linalg.norm(p_star - p_t)))
                if str(float(regrets[method, regret_index, rep])).lower() == 'nan':
                    raise ValueError('regret is nan')

# Plot Regret over time:
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

xinch = 2.5
yinch = 1.5
plt.figure(figsize=(xinch, yinch))
plt.subplots_adjust(0, 0, 1, 1)
# plt.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])

patches = []
# for method in [1]: # plot only result for one method.
ylim = 0.0
for method in range(num_methods):
    regrets_mean = np.mean(regrets[method][:][:], axis=1)  # absolute regret.
    regrets_sd = np.std(regrets[method][:][:], axis=1)
    if (method_order[method] != 'Random') and (np.max(regrets_mean) > ylim):
        ylim = np.max(regrets_mean)
    plt.plot(summary_times, regrets_mean, color=colors[method])
    plt.fill_between(summary_times, regrets_mean - regrets_sd, regrets_mean + regrets_sd, alpha=0.5,
                     color=colors[method])
    patches += [mpatches.Patch(color=colors[method], label=method_order[method])]

labelfont = 7
tickfont = 5
plt.ylabel('Regret', fontsize=labelfont)
plt.xlabel('T', fontsize=labelfont)
plt.ylim(ymin=0, ymax=ylim)
plt.xticks(fontsize=tickfont)
plt.yticks(fontsize=tickfont)
ax = plt.gca()
ax.yaxis.offsetText.set_fontsize(tickfont)
plt.legend(handles=patches, loc='upper left', fontsize=labelfont)
plt.show()

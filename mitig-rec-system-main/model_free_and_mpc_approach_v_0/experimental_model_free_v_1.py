#-------------------------dependencies----------------------------
import numpy as np
import pickle
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_free_and_mpc_approach_v_0.data_dynamics_new_model import simulate_dynamics_mf_data_new_model

file_path = "./initial_dependencies.npz"
data = np.load(file_path, allow_pickle=True)
initial_user_opinion = data['initial_user_opinion']
n_users = data['n_users']
lamda_diagonal_users = data['lamda_diagonal_users']
W_users = data['W_users']
W_rec = data['W_rec']
A = data['A']
B = data['B']
l = data['l']
m = data['m']
from model_free_and_mpc_approach_v_0.theoretical_dynamics_old_model import compute_convergence_values
from theoretical_dynamics_new_model import theor_simulate_dynamics_mf_new
from theoretical_dynamics_old_model import theor_simulate_dynamics_mf
from theoretical_dynamics_mpc_old import theor_simulate_dynamics_mb
from theoretical_dynamics_mpc_new import theor_simulate_dynamics_mb_new
from theoretical_dynamics_mpc_new import compute_convergence_theoretical_dynamics_mpc_new
from theoretical_dynamics_new_model import compute_convergence_theoretical_dynamics_mf
from figure_creation import fig_creation
from figure_creation_no_avg import fig_creation_with_no_avg
from misinformation_rho_plot import create_misinformation_rho_plot
from model_free_time_graph import make_graph
from data_dynamics_new_model import simulate_dynamics_mf_data_new_model
from data_dynamics_old_model import simulate_dynamics_mf_data_old_model
from data_dynamics_new_model import compute_convergence_values_mf_data_new
from data_dynamics_mpc_new import simulate_new_mpc_dynamics_
from data_dynamics_mpc_old import simulate_old_mpc_dynamics
relative_path = "./dataset_v_1/processed/grouped_posts.pkl"
with open(relative_path, 'rb') as f:  # or use file_path
    grouped_posts = pickle.load(f)

#--------------------------constants------------------------------
num_steps = 100 # number of timesteps
u = 0 # initial recommender opinion
p = 2.5 # rho constant
a_const = p * n_users 
range_window = 5 # number of posts to consider in the window
future_prediction_range = 50 # how many steps to predict into the future
lambda_val = np.array([0.00]) # lambda values for model

calc_synth = 0 # calculate synthetic theoretical dynamics
graph_synth = 0 # make graphs for theoretical dynamics
calc_data = 0 # calculate data-based dynamics
graph_data = 0 # make graphs for data-based dynamics
secondary_graph = 0 # make secondary graphs for data-based dynamics
calc_mitig = 1 # calculate misinformation mitigation metric
mb_also = 1 # also calculate misinformation metric for model-based MPC approach

#--------------------------theoretical steady state---------------
x_star_mf = compute_convergence_values_mf_data_new(n_users, A, B, lamda_diagonal_users, initial_user_opinion, a_const)
x_star_mb_new = compute_convergence_theoretical_dynamics_mpc_new(n_users, A, B, lamda_diagonal_users, initial_user_opinion, a_const, lambda_val[0])
x_star_mf_new = compute_convergence_theoretical_dynamics_mf(n_users, A, B, lamda_diagonal_users, initial_user_opinion, a_const, lambda_val[0])
########################theoretical dynamics######################
if calc_synth == 1:
#---------------------------model free----------------------------

#--------------------------old dynamics---------------------------
    theory_o_mf_old, theory_u_mf_old = theor_simulate_dynamics_mf(initial_user_opinion, u, num_steps, n_users, A, B, lamda_diagonal_users)

#--------------------------new dynamics---------------------------
    theory_o_mf_new_m = {}
    theory_u_mf_new_m = {}

    for i in range(len(lambda_val)):
        theory_o_mf_new, theory_u_mf_new = theor_simulate_dynamics_mf_new(initial_user_opinion, u, num_steps, n_users, A, B, lamda_diagonal_users, a_const, lambda_val[i])
        theory_o_mf_new_m[i] = theory_o_mf_new
        theory_u_mf_new_m[i] = theory_u_mf_new

#---------------------------model based---------------------------

#--------------------------old dynamics---------------------------
    theory_o_mb_old, theory_u_mb_old = theor_simulate_dynamics_mb(initial_user_opinion, num_steps, range_window, n_users, A, B, lamda_diagonal_users)
#--------------------------new dynamics---------------------------
    theory_o_mb_new_m = {}
    theory_u_mb_new_m = {}

    for i in range(len(lambda_val)):
        theory_o_mb_new, theory_u_mb_new = theor_simulate_dynamics_mb_new(initial_user_opinion, num_steps, range_window, n_users, A, B, lamda_diagonal_users, a_const, lambda_val[i])
        theory_o_mb_new_m[i] = theory_o_mb_new
        theory_u_mb_new_m[i] = theory_u_mb_new

#------------------------------graphs-----------------------------
# Define colors
orange_color = "#B9B9B9"
blue_color = "#006FC9"
red_color = "#FF7C01"
black_color = "#14CA00"
if graph_synth == 1:
    # Prepare your datasets for plotting
    datasets = [
        {
            'user_opinions': theory_o_mf_old,
            'recommender_opinion': theory_u_mf_old,
            'color': black_color,
            'linestyle': '-.',
            'marker': None,
            'label': 'mf-'
        },
        #{
        #    'user_opinions': theory_o_mb_old,
        #    'recommender_opinion': theory_u_mb_old,
        #    'color': black_color,
        #    'linestyle': '-.',
        #    'marker': None,
        #    'label': 'mb-'
        #},
        {
            'user_opinions': theory_o_mf_new_m[0],
            'recommender_opinion': theory_u_mf_new_m[0],
            'color': blue_color,
            'linestyle': '-.',
            'marker': None,
            'label': 'mf-m-'#,
            #'steady_state': x_star_mf_new
        },
        {
            'user_opinions': theory_o_mb_new_m[0],
            'recommender_opinion': theory_u_mb_new_m[0],
            'color': red_color,
            'linestyle': '-.',
            'marker': None,
            'label': 'mb-m-'#,
            #'steady_state': x_star_mb_new

        }
    ]

    # Create the plot
    fig = fig_creation(num_steps, datasets, f'comparison_plot_p{p:.3f}_range{range_window}.png', 'Recommender output $u^*(t)$')
    if secondary_graph == 1:
        fig = fig_creation_with_no_avg(num_steps, datasets, f'comparison_plot_no_avg_p{p:.3f}_range{range_window}.png', 'Recommender output $u^*(t)$')
###########################data dynamics##########################
if calc_data == 1:
    #--------------------------old dynamics---------------------------
    opinion_history_data_old, u_history_data_old, misinformation_spread_data_old = simulate_dynamics_mf_data_old_model(initial_user_opinion, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users, range_window)

    data_o_mf_old = opinion_history_data_old
    data_u_mf_old = u_history_data_old
    data_mis_mf_old = misinformation_spread_data_old
    #--------------------------new dynamics---------------------------
    opinion_histories_data_new = {}
    u_histories_data_new = {}
    misinformation_spread_histories_data_new = {}

    for i in range(len(lambda_val)):
        opinion_history_data_new, u_history_data_new, misinformation_spread_data_new = simulate_dynamics_mf_data_new_model(initial_user_opinion, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users, a_const, lambda_val[i], range_window)
        opinion_histories_data_new[i] = opinion_history_data_new
        u_histories_data_new[i] = u_history_data_new
        misinformation_spread_histories_data_new[i] = misinformation_spread_data_new

    data_o_mf_new_m = opinion_histories_data_new
    data_u_mf_new_m = u_histories_data_new
    data_mis_mf_new_m = misinformation_spread_histories_data_new

    #--------------------------old dynamics MPC-----------------------
    '''
    opinion_history_data_mpc_old, u_history_data_mpc_old, misinformation_spread_data_mpc_old = simulate_old_mpc_dynamics(initial_user_opinion, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users, range_window, future_prediction_range)

    data_o_mb_old = opinion_history_data_mpc_old
    data_u_mb_old = u_history_data_mpc_old
    data_mis_mb_old = misinformation_spread_data_mpc_old
    '''
    #--------------------------new dynamics MPC-----------------------
    opinion_histories_data_new_mpc = {}
    u_histories_data_new_mpc = {}
    misinformation_spread_histories_data_new_mpc = {}
    for i in range(len(lambda_val)):
        opinion_history_data_new_mpc, u_history_data_new_mpc, misinformation_spread_data_new_mpc = simulate_new_mpc_dynamics_(initial_user_opinion, grouped_posts, num_steps, n_users, A, B, lamda_diagonal_users, a_const, lambda_val[i], range_window, future_prediction_range)
        opinion_histories_data_new_mpc[i] = opinion_history_data_new_mpc
        u_histories_data_new_mpc[i] = u_history_data_new_mpc
        misinformation_spread_histories_data_new_mpc[i] = misinformation_spread_data_new_mpc

    data_o_mb_new_m = opinion_histories_data_new_mpc
    data_u_mb_new_m = u_histories_data_new_mpc
    data_mis_mb_new_m = misinformation_spread_histories_data_new_mpc
#-----------------------------------graphs------------------------

if graph_data == 1:
    # Prepare your datasets for plotting
    datasets = [
        {
            'user_opinions': data_o_mf_old,
            'recommender_opinion': data_u_mf_old,
            'color': black_color,
            'linestyle': 'None',
            'marker': '+',
            'label': 'mf-'
        },
        #{
        #    'user_opinions': data_o_mb_old,
        #    'recommender_opinion': data_u_mb_old,
        #    'color': black_color,
        #    'linestyle': 'None',
        #    'marker': 'x',
        #    'label': 'mb-'
        #},
        {
            'user_opinions': data_o_mf_new_m[0],
            'recommender_opinion': data_u_mf_new_m[0],
            'color': blue_color,
            'linestyle': 'None',
            'marker': '+',
            'label': 'mf-m-'
        },
        {
            'user_opinions': data_o_mb_new_m[0],
            'recommender_opinion': data_u_mb_new_m[0],
            'color': red_color,
            'linestyle': 'None',
            'marker': 'x',
            'label': 'mb-m-'
        }
    ]

    # Create the plot
    fig = fig_creation(num_steps, datasets, f'data_comparison_plot_p{p:.3f}_range{range_window}.png', 'Recommender output $u^*(t)$')

if calc_synth == 1 and calc_data ==1 :
    print("theoretical x^*_MB for new model:")
    print(x_star_mb_new)
    print("------------------------------------------------------------")

    print("M for theta MF:")
    print(misinformation_spread_data_old)
    print("M for theta_m MF:")
    print(misinformation_spread_histories_data_new[0])
    print("M for theta MB-MPC:")
    #print(misinformation_spread_data_mpc_old)
    print("M for theta_m MB-MPC:")
    print(misinformation_spread_histories_data_new_mpc[0])


if calc_mitig == 1:
    p_values = np.arange(0, 5.6, 0.1)

    mf_misinformation_results = []
    mb_misinformation_results = []

    print("Starting simulations...")

    # Loop through each p value
    for i, p in enumerate(p_values):
        print(f"Running simulation {i+1}/{len(p_values)} with p = {p:.2f}")
        
        a_const = p * n_users
        
        # Run Model Free simulation
        try:
            _, _, misinformation_mf = simulate_dynamics_mf_data_new_model(
                initial_user_opinion, grouped_posts, num_steps, n_users, 
                A, B, lamda_diagonal_users, a_const, lambda_val[0], range_window
            )
            mf_misinformation_results.append(misinformation_mf)
        except Exception as e:
            print(f"Error in MF simulation at p={p}: {e}")
            mf_misinformation_results.append(np.nan)
        
        # Run Model Based (MPC) simulation
        if mb_also == 1:
            try:
                _, _, misinformation_mb = simulate_new_mpc_dynamics_(
                    initial_user_opinion, grouped_posts, num_steps, n_users,
                    A, B, lamda_diagonal_users, a_const, lambda_val[0], 
                    range_window, future_prediction_range
                )
                mb_misinformation_results.append(misinformation_mb)
            except Exception as e:
                print(f"Error in MB simulation at p={p}: {e}")
                mb_misinformation_results.append(np.nan)

    # Convert to numpy arrays
    mf_misinformation_results = np.array(mf_misinformation_results)
    mb_misinformation_results = np.array(mb_misinformation_results)

    # Add this at the very end of your script
    print("Saving results...")
    results_filename = f'misinformation_vs_rho_results_range{range_window}.npz'
    np.savez(results_filename, 
            p_values=p_values,
            mf_results=mf_misinformation_results,
            mb_results=mb_misinformation_results,
            range_window=range_window,
            num_steps=num_steps,
            n_users=n_users)

    print(f"Results saved to {results_filename}")

    # Create the plot


    create_misinformation_rho_plot(
        p_values, 
        mf_misinformation_results, 
        mb_misinformation_results,
        f'misinformation_vs_rho_range{range_window}.png'
    )

    print("Simulation complete!")


'''
#extras for now
#--------------------------printing for confirmation--------------
#print("opinion history")
#print(opinion_history)
#print("recommender history")
#print(u_history)
#print(opinion_histories_new[0])
#print(opinion_histories_new[1])
#print(f"Number of groups: {len(grouped_posts)}")
print("misinformation spread old model:")
print(misinformation_spread_data_old)
print("misinformation spread new model:")
print(misinformation_spread_histories_data_new[0])
print("misinformation spread old model mpc:")
print(misinformation_spread_data_mpc_old)
print("misinformation spread new model mpc:")
print(misinformation_spread_histories_data_new_mpc[0])
#print(misinformation_spread_histories_data_new[1])
#print(misinformation_spread_histories_data_new[2])
#print(misinformation_spread_histories_data_new[3])
print("theoretical opinion convergence:")
print(x_star_mf)
#--------------------------making a graph-------------------------
#make_graph (num_steps, n_users, opinion_history, u_history, opinion_histories_new, u_histories_new, lambda_val, "opinion_dynamics.png")

#make_graph (num_steps, n_users, opinion_history_data_old, u_history_data_old, opinion_histories_data_new,
 #           u_histories_data_new, opinion_history_data_mpc_old, u_history_data_mpc_old, u_histories_data_new_mpc,
  #          opinion_histories_data_new_mpc, lambda_val,"opinion_dynamics_data.png")
'''
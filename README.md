SOCIAL NETWORK RECOMMENDATION SYSTEM
======================================================================

PROJECT OVERVIEW
----------------
This project implements and analyzes control strategies for recommendation systems in social networks, based on the bachelor end project "Control Approach for Misinformation Mitigation via Recommender Systems on Social Network Platforms" by Andreas Philippou, supervised by dr. Giulia De Pasquale. The implementation includes both model-free and model-based (MPC) approaches to study how recommendation systems influence opinion dynamics in social networks.

PROJECT STRUCTURE
-----------------
The project code is found in 'mitig-rec-system-main', and is organized into three main sub folders:

1. setup_v_0/ - Social network dependencies and system configuration
2. dataset_v_1/ - Dataset setup and processing
3. model_free_and_mpc_approach_v_0/ - Simulations and graph generation

GETTING STARTED
===============

PREREQUISITES
-------------
- Python 3.x
- Mistral AI API key (for emotional extremity evaluation)
- Required Python packages (install via pip):

STEP 1: SYSTEM SETUP
--------------------
1. Navigate to the setup_v_0 folder
2. Open user_network_setup.py - this is the main configuration file
3. Modify the variables according to your requirements:

Note: The file radical_user_network.py implements an exact extreme user graph model.

STEP 2: DATASET PREPARATION
---------------------------
The project uses the Liar2 dataset with custom preprocessing:

1. Navigate to dataset_v_1 folder
2. Open final_dataset.py - this controls dataset processing
3. Current implementation combines:
   - "Full true" + "Almost full true" → True news
   - "Full false" + "Almost full false" → False news

Recommendation: Consider modifying this preprocessing to preserve the original multipolar labels for more nuanced analysis.

API Requirement: The system uses Mistral AI for emotional extremity evaluation. You'll need to:
- Add your own API key in the configuration
- Alternatively, modify the code to run the model locally

STEP 3: RUNNING SIMULATIONS
---------------------------
1. Navigate to model_free_and_mpc_approach_v_0
2. Run experimental_model_free_v_1.py
3. Configure simulation parameters:
   - Toggle between theoretical and data-based approaches
   - Enable/disable model-free and MPC simulations (0 = off, 1 = on)

Important: Ensure you have sufficient data to distribute across all timesteps!

Performance Note: MPC simulations can be computationally intensive. For testing:
- Start with model-free approaches only
- Run MPC simulations after finalizing your setup

ADDITIONAL RESOURCES
====================

EXAMPLE GRAPHS
--------------
The Example-graphs/ folder contains reference outputs showing expected results. 

Graph generation files (can be modified or replaced):
- figure_creation.py
- figure_creation_no_avg.py
- misinformation_rho_plot.py

LEGACY FILES
------------
- model_free_time_graph.py - Outdated but retained for reference
- Files prefixed with [old] in the dataset folder - Initial implementations not used in final version

TIPS FOR FUTURE STUDENTS
========================
1. Start by understanding the theoretical model in the original paper
2. Test with small networks first before scaling up
3. Keep track of your parameter changes, they significantly affect results
4. Document any modifications you make for future students

I hope this project provides a solid foundation for your research. Good luck with your work!

----------------------------------------------------------------------
Last updated: 17-7-2025
Original author: Andreas Philippou

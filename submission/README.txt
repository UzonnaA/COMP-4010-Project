run "pip install pygame" in the terminal


main_options.py can be run with the following configurations:

Argument	            Type	Default	       Description
mode	                str	    -	           Operation mode: 'train' or 'run'.
--dvc	                str	   'cpu'	       Device for computation ('cpu' or 'cuda').
--Loadmodel	            bool	False	       Whether to load a pretrained model.
--ModelIdex	            int	    300000	       Index of the model to load.
--random_player	        bool	False	       Train with a random policy for the player agent.
--random_enemy	        bool	False	       Train with a random policy for the enemy agent.
--seed	                int	    209	           Random seed for reproducibility.
--T_horizon	            int	    2048	       Trajectory length for PPO updates.
--Max_train_steps	    int	    5000	       Maximum training steps.
--save_interval	        int	    1000	       Steps between saving models.
--eval_interval	        int	    5000	       Steps between evaluations.
--gamma	                float	0.99	       Discount factor.
--lambd	                float	0.95	       GAE lambda.
--clip_rate	            float	0.2	           Clipping ratio for PPO.
--K_epochs	            int	    10	           Number of PPO updates per iteration.
--net_width	            int	    64	           Number of hidden units in the neural network.
--lr	                float	1e-4	       Learning rate.
--l2_reg	            float	0	           L2 regularization coefficient for the critic.
--batch_size	        int	    64	           Batch size for PPO updates.
--entropy_coef	        float	0	           Entropy coefficient for the actor.
--entropy_coef_decay	float	0.99	       Decay rate of the entropy coefficient.
--adv_normalization	    bool	False	       Whether to normalize advantages during training.

example usage:
train on 300000 experiences using PPO, save every 100000 experiences learned on, and sample random actions for the farmer.
python main_options.py train --Max_train_steps 300000 --save_interval 100000 --random_player True --algorithm ppo

run a rendered and nonterminating episode using the saved ppo models of both agents which were saved at 300000 learning steps.
python main_options.py run --Loadmodel True --ModelIdex 300000 --algorithm ppo

FOR TAs:
run an experiment using DQN, 150000 training steps for each experiment as outlined in main_options and in the paper. (the paper excludes some experiments because it would take too long to write for)
python main_options.py experiments --algorithm dqn --Max_train_steps 150000

run an experiment using PPO, 150000 training steps for each experiment as outlined in main_options and in the paper. (the paper excludes some experiments because it would take too long to write for)
python main_options.py experiments --algorithm ppo --Max_train_steps 150000
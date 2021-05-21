# Udacity Deep Reinforcement Learning Project 3: Collaboration and Competition

## Project Details
The code in this repo interacts with a modified version of the [Tennis Environment.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
This environment puts two separate agents in control of tennis rackets, with the goal of hitting a ball back and forth over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If the agent allows the ball to hit the ground on its side of the net, or if the agent hits the ball out of bounds it receives a reward of -0.01.

To achieve this goal, each agent takes action by sending the environment a vector of 2 numbers in the range [-1,1].  These numbers correspond to the agent's movement to the left or right, plus their decision to jump. State information is given to each agent as a vector of length 24; this state information contains three frames of data about the position and velocity of the ball and the racket.  Each of the agents receives its own on observation about the environment, without information about the other racket.

The environment is considered "solved" the maximum of the two agents' scores for each episode is at least +0.5, when averaged over a 100-episode rolling window.

## My Solution
To solve the environment, I implemented an Multi-Agent Deep Deterministic Policy Gradient model.  For more details on my findings, see the writeup in [report.ipynb](report.ipynb).

## Getting Started

### Python Setup
This project has been tested on Python 3.6; it may work on later versions but is incompatible with earlier ones.
It is recommended that you use a virtual environment using conda or another tool when installing project dependencies.
You can find the instructions for installing miniconda and creating an environment using conda on the
[conda docs](https://docs.conda.io/en/latest/miniconda.html).

### Python Dependencies
After creating and activating your environment (if you're using one), you should install the dependencies for this project
by following the instructions in the [Udacity DRLND Repository.](https://github.com/udacity/deep-reinforcement-learning#dependencies)


### Unity environment
Once you have the python dependencies installed, download the version of the unity environment appropriate for
your operating system.  Links for each operating system can be found below:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [Linux Headless](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
* [Mac Os](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows 32 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [Windows 64 bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

After downloading, use 7zip or another archive tool to extract the environment file into the root project directory.
By default, the code is set up to look for the Linux version of the environment, so you will need to modify the
UNITY_ENV_PATH variable in `train_agent.py` or `run_agent.py` to point to your new version.

## Running The Code
The `train_agent.py` python file at the project root contains the logic necessary to train both the actor and the critic networks.  You can run it with the command `python train_agent.py`.  Note that you will need to update the UDACITY_ENV_PATH variable to point to your version of the Unity environment.  By changing the other variables in ALL_CAPS at the top of the file, you can modify various hyperparameters used by the agent during training.  After training, this script will store the final actor and critic weights in the `model_weights` directory. It will also store the average score per-episode as a csv in the `scores` directory.

The `run_agent.py` python file at the project root contains the logic necessary to run the network in the environment.  You can run it with the command `python run_agent.py`. Once again, you will need to update the UDACITY_ENV_PATH variable to point to your version of the Unity environment before running this script.  By modifying the `TRAIN_MODE` variable to False, you can watch the agent as it runs.

# Acknowledgements
During completion of this project, I used several external sources as references and inspiration for my implementation.  This does not include any direct code usage, but does include hyperparameter decisions and general reference to understand the algorithms. In addition, I used a modified version of the OU noise process from the Udacity MADDPG workshop.  These sources are listed below:
- [Chris Yoon's "Deep Deterministic Policy Gradients Explained" Medium Post](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
- [*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*, Lowe et al.](https://arxiv.org/abs/1706.02275v4)

# Problem Statement
The main motivation behind this problem statement is to identify and address a real world issue faced-
of students commuting from home to campus during snow days. Finding the shortest path from home to
campus can be impacted by a variety of events, including taking a bus to campus, walking to campus but
feeling cold, taking detours into buildings to meet friends/escape from the cold. A student might also have
different events to get to, which means their target is not necessarily fixed. This model can be seen as a
more complicated version of gridworld which was discussed over assignments, with a bigger state space, and
a variety of additional features.

# Assumptions
Various assumptions were made to model this problem statement. Firstly, the state space was considered to
be a 10 by 10 grid. This assumption keeps the state space fairly large, while ensuring that the algorithms
do not have a very high run time. The buildings, road locations, terminal state locations, crosswalks, and
further details on the modeled state space are detailed below. These were arbitrary decisions, however,
they were chosen to ensure that there is complexity in the MDP and that every incremental feature (road,
crosswalk, buildings) required the model to learn more and explore more aspects of the state space.
The MDP was built on Python, using object oriented programming structures, and the OpenAI Gym
implementation was used to run reinforcement learning algorithms. The environment was built in stages, and
there are multiple versions of the environment present in the code. Each of these environments corresponds
to the incremental updates in the environment and complexity.

# MDP Formalization
The MDP is a gridworld of size (10x10). There is a road present on the top row of the MDP, represented
by grey. There are crosswalks replacing the road at states [8, 0], [6, 0],[4,0]- represented by yellow. There
are buildings or warm spots present through the grid- located at [2, 3], [2, 4], [4, 6], [2, 8], [7, 7], [7, 3]-
represented by maroon. The possible goal states (1) is located at [9, 9] (2) is located at [4, 9] and (3) is
located at [9,7]. The active goal state for the episode is marked in dark green, while the remaining goal
states are marked with light green. The agent is represented by the circle, and a run of an optimal policy
after training on the algorithms in results is seen with the lines. A report detailing the MDP states and
formalization: RL_Report.pdf

# Results
With the implementation of the MDP, the returns at the end of training were in the range of [35,42] with
different goal states over a series of episodes, except for DQN which performed slightly poorer. The average
episode lengths lied between 14 and 18, since the range for different return states was different. All three
algorithms found a policy which led them to use the road initially, followed by trying to make it into the
warm buildings which are on the way to the final terminal states. A report detailing the results: RL_Report.pdf


# References
- OpenAIdocs. Deep q learning documentation, a. URL https://openai.com/index/
openai-baselines-dqn.
- OpenAIdocs. Proximal policy optimization - spinning up documentation, b. URL https://spinningup.
openai.com/en/latest/algorithms/ppo.html.
- OpenAIdocs. Trust region policy optimization - spinning up documentation, c. URL https://spinningup.
openai.com/en/latest/algorithms/trpo.html.

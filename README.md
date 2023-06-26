# Continuous Intrinsic curiosity module
Purpose of this project - reproduction of Intrinsic curiosity module (ICM) paper by Pathak et al., and implementation of continuous version of ICM.

# Results of reproduction.
Scenario, on which agent was tested:
3 actions: turn left, turn right, move forward. Reward: 1 on vest achievement (End of labyrinth), zero else.
Instead of A3C algorythm, in this project A2C algorythm was used. As a result, on sparse scenario
![Sparse](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/e4ad8369-787b-49d3-a633-5640f02899a9)
we were able to achieve similar to original results. (One step in below graph is equal to 20 samples of experience). Thus, our agent learns twice as slow, as the original one.
![Extrinsic reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/99e324a0-956d-41ad-8556-9419f9c4e0a8)
![Mean episode steps](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/2960ec09-8f4a-4c23-a9c5-9e8f80b199c9)
This results averaged on 3 seeds.

Second environment, on which agent was tested (First chronologically), was minigrid environment. 
4 Actions: move left, up, right, down. No extrinsic reward, only intrinsic reward. Purpose of this environment: debug ICM and explore it's exploration capabilities and how they are dependend on hyperparameters.
Map:
![Map](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/886de8fe-e384-43ff-9692-6a9196ab20d2)
At this environment was discovered that entropy coefficient is important for ICM exploration, (when used with A2C)
![Mean entropy graph](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/7b958e13-8367-4a76-9ffd-4054c3f36094)
![Exploration graph](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/56d62e2c-6bb3-48e7-a946-445547689c44)
Also, we made sanity checks: It is expected, that more distant places should give more reward to agent, than less distant. We can see this in practise.
(This reward was calculated as mean reward, received from moving from all surrounding cells to target cell (Surrounding cells = cells one action apart from target cell))
![Mean reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/38370e9c-59b8-41fe-8640-b424b055bc0b)
But, for some interesting reason, estimation of value, made by critic, is independent of agent placement. (This metric was calculated as value estimation of critic of agent's state)
![Value estimation](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/bc9a5042-8093-4c0a-a8cb-b76ef2098189)

# Results of experiment.
Afterwards, we tried to implement continuous version of ICM. In order to achieve this, we changed output of inverse model: now, instead of probabilities of all actions, inverse model predicts performed action. Loss was also changed from CrossEntropy to MSE.
First environment was ViZDoom in continuous setting. 
Action vector: 2 component vector, first component - angle of rotation, second component - amount of movement. Agent could move only forward, not backward: experiments showed that allowance of backward movement complicated the task.
A2C showed bad results with continuous ViZDoom problem:


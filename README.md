# Continuous Intrinsic curiosity module
Purpose of this project - reproduction of Intrinsic curiosity module (ICM) paper by Pathak et al., and implementation of continuous version of ICM.

## Results of reproduction.
Scenario, on which agent was tested:
3 actions: turn left, turn right, move forward. Reward: 1 on vest achievement (End of labyrinth), zero else. Cap of environment steps: 525.
Instead of A3C algorythm, in this project A2C algorythm was used.  

![Sparse](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/e4ad8369-787b-49d3-a633-5640f02899a9)  
Observation example:  
![Observation](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/32256e6c-9bfd-4fdd-923a-a8f5829c62a1)


As a result, on sparse scenario we were able to achieve similar to original results. (One step in below graph is equal to 20 samples of experience). Thus, our agent learns twice as slow, as the original one.
![Extrinsic reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/99e324a0-956d-41ad-8556-9419f9c4e0a8)
![Mean episode steps](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/2960ec09-8f4a-4c23-a9c5-9e8f80b199c9)
This results averaged on 3 seeds.

## Minigrid
Second environment, on which agent was tested (First chronologically), was minigrid environment.  

4 Actions: move left, up, right, down. No extrinsic reward, only intrinsic reward. Purpose of this environment: debug ICM and explore it's exploration capabilities and how they are dependend on hyperparameters.  

Map and observations:  

![Map](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/886de8fe-e384-43ff-9692-6a9196ab20d2)  

At this environment was discovered that entropy coefficient is important for ICM exploration, (when used with A2C)
![Mean entropy graph](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/7b958e13-8367-4a76-9ffd-4054c3f36094)
![Exploration graph](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/56d62e2c-6bb3-48e7-a946-445547689c44)
Also, we made sanity checks: It is expected, that more distant places should give more reward to agent, than less distant. We can see this in practise.
(This reward was calculated as mean reward, received from moving from all surrounding cells to target cell (Surrounding cells = cells one action apart from target cell))  

![Mean reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/38370e9c-59b8-41fe-8640-b424b055bc0b)  

But, for some interesting reason, estimation of value, made by critic, is independent of agent placement. (This metric was calculated as value estimation of critic of agent's state)  

![Value estimation](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/bc9a5042-8093-4c0a-a8cb-b76ef2098189)

# Results of continuous experiments
## ViZDoom continuous
Afterwards, we tried to implement continuous version of ICM. In order to achieve this, we changed output of inverse model: now, instead of probabilities of all actions, inverse model predicts performed action. Loss was also changed from CrossEntropy to MSE.
First environment was ViZDoom in continuous setting. Extrinsic reward, as in previous setup was given only for maze completion, and cap of steps was 525 (As in previous setup).   

Action vector: 2 component vector, first component - angle of rotation, second component - amount of movement. Agent could move only forward, not backward: experiments showed that allowance of backward movement complicated the task.  

Solution of the environment with perfect policy would mean that agent performed approx. 66 steps on mean steps chart. 

All graphs here averaged with three seeds. It's worth mentioning that agent either solved environment ideally, or didn't solve it at all. So there were two options of graphs: either straight line, or straight line with waterfall. Thus, on average graph we must see sort of stepwise graph.  

### A2C
A2C + ICM showed bad results with continuous ViZDoom problem:
![A2C+ICM](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/0f8dfe4d-9bbf-44d1-b764-a471c796e40a)  

A2C solely also could not solve continuous ViZDoom problem ideally:
![A2C](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/8f0f8d4f-a023-4150-a048-ae0ccd9909b8)

### PPO
Thus, we decided to try PPO+ICM algorythm. And it always solved ViZDoom scenario:
![PPO+ICM](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/daeaf956-606a-47ce-8fb3-99941baa3ffe)
Solution example:  
![solution](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/5709bd37-5b9c-4e35-a349-1f2aa3d805af)

We decided to compare it to PPO without ICM. This method could, sometimes, solve the environment, but not always:  

![PPO](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/ea3babf8-350c-4978-8ed2-1dff878edcfa)
Thus, on MuJoCo environments we decided to continue working with PPO algorythm.


## MuJoCo egocentric
At this environment, we tried easier format of labyrinth from gymnasium-robotics:  
![Labyrinth](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/3cb73245-575c-444f-867a-aa3dac29d5c8)  
However, problem of this setting is that inertia is playing much bigger role here, rather than in ViZDoom.
We tried two setting: egocentric pointmaze (with ball swapped to cylinder, to avoid rotation of camera) and egocentric antmaze.  
As in previous setups, we checked only sparse extrinsic reward, that was depended on quantity of steps before achievement of reward - less steps - more reward. Cap of environment steps: 1000.

![Egocentric obs](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/91f315eb-3fc3-47cc-a31f-fc3933339382)  

At pointmaze, same as ViZDoom we had vector, consisted of two actions: angle of rotation and applied force in the direction of sight. In the easy setting force could be applied only in forward direction, whilist in hard setting force could be also applied in backward direction.  

### Cylindermaze, easy
As was mentioned above, in the easy setting force could be applied only in forward direction. Interestingly, ICM sped up learning process in easy configuration:  
![Mean episode steps](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/808d3366-b4f5-4238-96d1-4a83527a096a)
![Extrinsic reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/9abb0ec8-723a-4f92-9d21-634d71933d07)   

Policy example:  

![Cylinder easy solution](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/6d080705-6428-4329-a4f2-5373af389e48)


### Cylindermaze, hard
As was mentioned above, in hard setting force could be also applied in backward direction. Interestingly, here ICM didn't give any boost in learning.  

![Mean episode steps](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/c697e788-1c18-48c9-9960-98b24c4e470a)
![Extrinsic reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/463815c6-e8fe-4e3d-99ae-d8a228689733)
On other hand, maybe agent needed more time, in order to solve the environment and then we would see the difference.  
Policy example:  

![Cylinder_hard_solution](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/c12dfc42-e05b-48db-99cf-ea6f3050d225)


### Antmaze
This problem occcured to hard for agent to solve. On other hand, maybe we could use more computation time, in order to see progress.
![Mean episode steps](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/d9480e95-457f-4c4a-8271-67dad186a20e)
![Extrinsic reward](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/f3090ee1-7ed2-42ce-84ef-1672a10b450c)
And, whilist some progress could be seen, on other hand, it is obvious that inverse model of ICM behaves strangely:  
![Inverse model](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/7ebc6e60-275b-4c8b-b983-860be72d749c)
This might be due to impossibility to predict movements of agent. Thus, maybe if we slightly changed position of a camera, agent would be able to predict agent actions more precisely.  
Example of learned policy:  

![Ant_solution](https://github.com/denmanorwatCDS/mario-icm/assets/119806492/954673d8-6748-4797-989f-abd37ff3a787)

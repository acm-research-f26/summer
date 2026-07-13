![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

# Integrating Monte Carlo Permutation Search (MCPS) to play RPG Games as a fallback for RL Models

## 📌 Project Summary
Monte Carlo Tree Search (MCTS) is the idea of essentially simulating games in the future by branching out and exploring different options, while storing for each state (a set of actions from the current action) + proceeding action the overall 'value' of it, gotten by how many paths down from the current state end up with a win vs a loss.

However, there come three issues with MCTS. First is that while it only stores information regarding how valuable the next immediate action is, while it could be useful to also store data about actions that happen later in the simulation we did that ended up being very powerful. Additionally, the state requirement is very strict in having to follow an exact combination, which makes it hard to find the truly best action since we don't exactly take into account maybe actions across various states as well and what their value ends up being. Finally, when exploring 'playouts' (which are basically simulating from the current state taking actions until reaching an end state), part of determining the playout chosen is based on which (s, a) nodes ended up having the highest probabilities (aka which starting actions looked the most promising to explore further), but since MCTS is quite strict with only updating the specific (s, a) pair each time a new playout is discovered, it takes a longer time to get accurate Q-values (how 'good' taking an action from a current state is) for those state-action pairs, potentially delaying the epxloration of the best actions in certain states. 

This is where MCPS comes in, where it solves these problems by having a bit more relaxed value-filling by using the GRANT algorithm, which basically has Q(s, a) built where a is any action occuring in the playout from the state, not just the immediate next one, allowing us to see what actions in general end up being useful. It also adds its own permutation condition, where the states, which are normally an exact ordered set a(1)...a(n) of actions from the root (root being the start from which wer'e simulating), are now instead a subset of all the states in the playouts, basically noting down what actions tend to be good after a general set of previous actions rather than a strict ordered set of them.

While they used MCPS on games of Go, I showed how it could be extended to other non-board games such as turn-based RPGs, by creating a system for that and then seeing how well MCPS performs. Additionally, I showed how we can use MCPS with regular RL as essentially a backup for when RL fails.

## 🎯 Motivation
The main motivation behind this was that I wanted to work with RL while also exploring this idea of simulating the game using Monte Carlo methodologies. I also wanted to show it could specifically be combined with existing RL techniques to become even better, and potentially account for the drawbacks behind traditional RL.

Additionally, I thought it would be a nice opportunity to make a turn based RPG specifically cause I could base it off a RPG game I have been playing quite a lot.

## 🧩 Novelty
The main novelty behind this is as mentioned in the project summary, having MCPS basically determine what actions are good from certain states by encoding on that state side not just the specific actions that would make up that state, but making it more general and basically expanding the state to be any of those actions done in any order along with other actions. This makes it so that when deciding which nodes to explore for a playout, it picks actions that perform generally well too, not just in a specific state. 

Additionally, my system currently uses PPO, a RL model, but intermingles it with MCPS. Basically, the idea was that usually a normal RL model would be good enough to play the game by itself. However, obviously an RL model is trained on data, but if it never gets data of certain scenarios, it won't know how to act in those scenarios. My project accounts for that though, as essentially we can measure the entropy of the model each time it makes an action (so how 'confident' the model is in its predictions, where if say its' a 55% probability of action 1 being the best and 45% chance of action 2 being the best, that'd be high entropy, while 99% and 1% for those actions would be low entropy, and 'confident'), and then if it has low confidence we can then use MCPS to simulate a few moves ahead to determine what is the best action to take.

Finally, another novelty in what I do is adapt the MCPS model to a different game type than what it is normally used for. Typically, MCTS/MCPS is used for board games since they are easy to simulate in the future and its pretty simple to pick actions from them, as well as it being easy to reach an end state (that being when you win or lose). However, I noticed that you could also potentially use this model for Turn Based RPGs as they have most of the same elements, so I (aka Claude) had created a Turn Based RPG system to be able to test MCPS.

## 🧠 Methodology
1. **Environment Construction**:
First was actually building a game that I could use MCPS for. As mentioned earlier, I chose a turn based game cause I thought it'd be cool and interesting. As for what game to base it on, I actually decided to make it based on a game I'm currently playing, called 'Limbus Company'. It's a turn based game focused on teambuilding, 'clashing' against enmy skills to override them, and applying status effects to the enenmy, so I picked some of my favorite status effects to play in the game (mainly Burn and Tremor) and just directly implemented them in game. 

The main things to note are that when passing into an RL model, it first passes in the current character and all their statuses, as well as upcomign skills, and what skills the boss will do as well as who it will target. Next, the player will then need to decide which of its 2 skills to use, and which skill on the boss's side to 'clash with' (or it can also attack unopposed).

As for rewards/punishments, currently it's done by if the battle was won/loss, but in the future for much longer fights where simulating the entire thing might be unrealistic, we could instead do it as having the reawrd/punishemnt be a turn step of how much damage was dealt to the boass that turn as the reward and how much damage the player's took as the punishment, and adding other things when needed (i.e. like punishing if a player got 'staggered').

In the end, this is how it ended up looking. One thing is that I used AI to entirely create the game engine, because I was assuming that what I would actually code would be the PPO and MCPS related stuff (but little did I know, I was a lazy fraud).


2. **RL Setup**:

3. **Implementing Genetic Programming**:

4. **Results And Evaluation**:

## 🌍 Impact
This project will allow game designers to design better systems for AI in games, especially when they may be unsure of how to design an AI in such an environment. They can just specify
the possible conditions that can be checked, what actions can be taken, and any other constraints that may be needed, and then see how different combinations of behavior trees can cause differnet results which eventually converge in the 'ideal' behavior tree. This does so in a way that makes sure 'good' subtrees are usually not destroyed as well.

This project in particular has the impact of showing how the project can be adapted to more complex scenarios. While before it was just for Pac Man, this project shows how it can still be adapted to even 3D gaems without necessarily needing vision as input either.

#### Future Work
A couple things can be done to improve the project. First, the quality of the behavior tree itself can be improved. Right now, we have it so that each internal node is a binary conditioin that can branch to eitehr 'yes' or 'no', and the leaf nodes represnet a single action. Howver, Behavior Trees can be much more complex, and there's at least 3 things we can add among many others. First is a internal node that has children as a bunch of actions taht functions as an 'OR' loop, where it will try to run all those actions until one of them completes. Another is a Sequential node that, when fired, runs actions in a sequence that only finishes when the last action finishes. Finally, another thing is normally in behavior trees when an action finishes, instead of re-running the behavior tree starting at the root (like my version does), instead normally behavior trees will go to the last completed action's parent and then check if it needs to continue percolating upwards or not, keeping the behavior tree 'defined' within a certain area still.

Moving away from the behavior tree, it'd also be nice to try to adapt this to other games besides DOOM and games that require a great deal more of complexity, as it would be interesting to see how they do. However, the main limitation behind this is finding a game for which ti will work - i was only able to do Doom thanks to the open source library that did all the game engine stuff for me.

Finally, it may be interesting to try to further expand on the GP side of things. One thing I'd like to do is change the dynamic constraint so instead of picking common subtrees, it picks trees that overall perform effectively instead, which can be say evaluatied by something like summing up the scores of all the actions within that subtree or similar. It may also be more interesting to just look further into the GP side in general.

**Additional Sources:**
- The actual paper: https://www.mdpi.com/2076-3417/8/7/1077
- VizDoom Library: https://vizdoom.farama.org/

**Setup Guide:**
First, you have to install 2 python libraries. First is vizdoom (which is for DOOM and its api), and second is numpy. You can install these with the command "pip install numpy vizdoom".

After this, just run "deathmatchGame.ipynb" (you can set variables you'd like to change in constants.py), wait for it to finish, and then run testTree.ipynb to see how the trees perform. That should be it I think.
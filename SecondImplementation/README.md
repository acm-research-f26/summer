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
<img width="1867" height="931" alt="image" src="https://github.com/user-attachments/assets/0ffbbbdc-e5ce-4a79-b4ce-6461eaa3ef16" />

2. **RL Setup**:
This is whre I should mention something. While my intention was to impelemnt the RL and MCPS side entirely my self, I ended up running out of time to do so and had to vibe code that aspect of it, though I do understand most of the code. However, this is entirely on me as I procrastinated a lot and didn't manage my time as well.

For the RL, I chose to use PPO because I wanted to learn a bit more about that type of RL model, and typically from my research I see it often as the best model for games. Anyways, we can just set up the model as mostly any other ML model, though we're passing in a flattened list consisting of all the data that will needed to be known about each of the players and the boss. We take reward as just that basic difference of loss in boss hp minus the loss in total player hp, though later we'd ahve wanted to add the other things i mentioned like if we got any staggered units or maybe if we won clashes too. Once again, we can also measure entropy to determine if the model is uncertain of its moves, in whih case it'll use MCTS.

3. **Implementing The MCPS Side**:
Once again I didn't exactly implement this myself, but I can talk a bit over how it works. Each time we call it, we start at the current state as the 'root' node. We then explore up to a certain number of playouts, where playouts consist of us taking moves until we reach a win or loss (which is clearly defined by us either beating the boss or having all our units die). 

We also define going to different states by making differnet actions as separate nodes in a Graph (as basically what we have is a tree, with leaves being the end of the battle). Since each ndoe basically represents a state, we can store q value data for each node. 

The concept is that each time we explore a playout, we add one more node to our tree, but don't add all the nodes along the playout as we make actions since that'd explode the size of the tree. Then, each time we restart a playout, we choose the leaf node we got to by a mix of random chance and which one has the best q value (currently). Each time we finish a playout, we can update all 3 q values (as in for MCTS, GRANT, and MCPS) by going back up the tree, seeing what states were involved and what actions are involved and updating so accordingly, using a dictionary mapping.

4. **Results And Evaluation**:
The results that came in were quite interesting. First reported here is how well MCPS did, though just over 50 iterations. What we can see is that the MCPS performs remarkably well at a 100% accuracy. This mgiht look suspicious that it never fails, but there's a very good reason for that: I forgot to account for the fact that the game I created was actually not complex enough for MCPS, as basically the depth of the tree built is only like 10 at most (since it only takes about 10 turns on average to win), while there aren't like an extremely high amount of actions that can be taken (specifically it's about (2*4)^3), which is a lot but not compared to say Chess. This means that basically MCPS can 'solve' the entire game and find the actions that always lead to winning.

One thing you can notice though is that the time per turn is decently long, at about 0.5 seconds (or 500 ms), this makes sense as of course it is running many simulations into the future at once. Honestly, for a turn based RPG if this was an enemy Ai, the latency would be fairly acceptable as it's not a real-time game in the same way as a FPS.

<img width="746" height="47" alt="image" src="https://github.com/user-attachments/assets/b1b1f07a-40b0-4935-9d65-ad5aedd1b3b2" />

Meanwhile, for the pure RL model, we find that it doesn't perform nearly as well as the MSPS. One thing I ran earlier was seeing how well you do if you just randomly pick actions and it turns out to be about 3%, so we can see since the accuracy is still about 24% that at least it's learning something. The reason as to why this may be happening is due to how I didn't actually have the RL running for long enough, I think I only had it running for about 500 episodes. We do see though of course that inference is much faster though.

<img width="1000" height="127" alt="image" src="https://github.com/user-attachments/assets/db4c06ed-a8ea-4246-814f-cdab14a00100" />

Meanwhile, moving onto the hybrid RL model where we use MCPS when it's less confident, we found that it performed much better than pure RL, though still not perfectly like just pure MCPS. There can be multiple reasons for why. First is that it's just possible that MCPS taking over was quite rare as usually the model ended up being confident, which doesn't really help in our case since our model isn't trained very well. Another possibility is that typically by the time the RL model passed control over to MCPS, its possible the situation was so bad that not even MCPS could recover from it, so basically it was already doomed to lose by that point. Overall though we notice the latency much higher now than pure RL but still only like sub 2 ms, meaning that this is definietely viable for turn based games or even FPS games if we wanted to do that.

<img width="996" height="116" alt="image" src="https://github.com/user-attachments/assets/79384fe8-ceb9-4250-a75c-f5d843d5b70b" />



## 🌍 Impact
This project has the impact of showing that integrating RL with MCPS could be a promising avenue for games, as the combination basically accounts for the downside of RL (not being trained on all scenarios) and the downside of MCPS (htaking too long to simulate) perfectly, and we see that it performs quite well in terms of latency and winrate. 

Additionally, this project shows how we can actually adapt MCPS to different types of games beyond board games, so the future in that regard is quite bright.

#### Future Work
Boy there's a lot.

First is of course that almost all of this project (though I guess not all of it) is AI slop I made due to poor time management. If I was to work on this project in the future that'd be the first thing that I'd like to fix, so that I understand the code much better since I'm the one who wrote it.

Second, I'd like to expand the game further so that it can't really be 'solved' so quickly by MCPS in regards to what is the optimal move. There already exists a clear template on how I can do this of course as I only adapted maybe 20% of the combat of Limbus Company (which as a reminder is the game I'm basically ripping off with this test game), so I already know what I could add otherwise to make the game more complex and make the dimensional space of the actions and state bigger. This may mean now that the q values for each (s, a) state have to be dtermined by an arbitrary reward/punishment value though now for each playout though, since it now may be come unlikely for a playout to be able to feasibly reach a finish state within memory space, but this isn't a big issue as we can just make our own reward/punishment function or base it off the current RL one.

Third, I think it'd be good to explore differnt RL models as well besides PPO and determine which one would be best to pair with MCPS so that they cover each other's downsides. For instance PPO is good but maybe it's too complex which MCPS already handles, so maybe we could sacrifice RL accuracy to use a faster model like DQN to improve latency. Or maybe instead of deciding to switch to MCPS when tne entropy for the RL model is high, where we can instead do something like train multiple RL models at once (like an ensemble model) and then decide the overal action by taking the average across these models. Then the justification to switch to MCPS for the turn could be say if all the models end up disagreeing with the skills each unit should use and who they should attack.

Finally, it'd be interesting to see what other game genres we can expand this system to. Obviously many more game genres exist besides board games and turn based RPG games, so maybe we can try to expand it to puzle games or platformers or even fighting games. There comes a limitation with this though that should be noted as a warning: MCPS is mainly limited by how well it can simulate into the future. For board games or turn based RPGs, this is usually simple since the action space is discrete and actually simulating potential actions doesn't take much time (like in chess, all you have to do is move a piece to another spot, and check for checkmate/check/piece taken). However, when you get to things like FPS, good luck trying to run 10k playouts of a COD game within a single timestep to decide the next best action, lol.

**Additional Sources:**
- The actual paper: https://arxiv.org/pdf/2510.06381

**Setup Guide:**
First, you have to install 2 python libraries. First is pygame and second is python. You can simply install these with the command "pip install numpy pygame".

First, if you want to run the game and play it yourself to see how it works, you can directly run "game_ui.py" and see how it is. Note that the instructions aren't a fully fleshed out tutorial, so unless you have elite ball knowledge and already know how to play Limbus Company it'll probably be confusing. There are still tooltips to help though.

If you want to see how the MCPS by itself performs, you can run mcps.py directly.

To run the PPO model or the hybrid model, you can see how to run those by going to the ppo_train.py file and then looking at the bottom of the file, which has a guide on what commands to do from the terminal. Note that this assumes in VS code that you're in the outer folder rather than the SecondImplementation folder. If you are in the SecondImplementation folder, you can just remove any instances of "SecondImplementation/" from the terminal command and it should work.
![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

# Building Dynamic Behavioral Trees To Play DOOM Using Genetic Programming

## 📌 Project Summary
Behavior Trees are a common way to simulate the actions of AI in video games, which work as slightly more complex decision trees that allow things such as taking multiple actions at once or performing actions in a sequence. Nowadays they are used much more than their predecessor for deciding actions, that being finite state machines.

However, one issue with behavior trees (and FSMs too, along with symbolic AI in general) is that how well they do ends up being entirely dependent on how well the person is able to design the structure behind the tree - this means its possible for a human to design a tree that doensn't fully capture the best that the AI can do in its environment.

This is where this project comes in, which builds behavior trees dynamically, by basically having mutliple generations of randomly created behavior trees. These trees are created by selecting from a pool of conditions that serve as checks / branch points, and then actions that serve as actions the AI takes that are the leaves of the tree. Then, each tree plays the game and gains rewards/punishments for doing well, which are summed up at the end of the generation. Then, the ones that did best get to 'survive' into the next generation, while also having the possibility to 'mutate' or 'reproduce'. Eventually, the idea is that statistically the trees will die off and new trees will come in such that the remaining ones are ones proven to 'survive' in that environment, aka being quite effective binary trees.

## 🎯 Motivation
I really wanted to try something with genetic programming, and I also wanted to learn a bit more about behavior trees and how they're implemented since my research topic will involve AI in games, so coming across this paper seemed like a really good first starting point for the first implementation. Additionally, I chose to do DOOM for the game cause I thought it would be cool. 

## 🧩 Novelty
The main novelty in the paper itself is how it uses something it called 'dynamic constraints' to avoid a problem with the initial approach of using GP for Behavior Trees.

GP has this feature for behavior trees called 'crossover', which is where you take 2 trees and swap 2 random subtrees between them (where those subtrees keep all their children). However, it turns out that this can be a problem because sometimes the subtree that is swapped over was part of a really good / effective subtree, and thus swapping it out for a new set of nodes underneath esssentailly breaks the logic that made the subtree work well.

How the paper fixes this isssue is by introducing a dynamic constraint where after each run in a generation it uses FREQT on the highest scoring Behavior Trees, which looks at all the subtrees within the behavior trees and finds the most common ones. Then, it lowers the probability of these nodes being chosen for crossover. The idea is that it assumes since these subtrees were frequent within the most common trees, they must be 'good' subtrees that shouldn't be broken apart in any way, so it lowers the probability of such a thing happening.

It also had a 'static constraint' that basically made the BT follow certain rules that constrained the size of it so it wouldn't explode, but this wasn't something i implemented.

Additionally, as for the novelty I introduced, they originally tested their new methodology using the game Pac-Man, meanwhile I decided to test it by doing it on the game Doom instead since it'd be cooler.

## 🧠 Methodology


1. **Evaluation**:
   - 
   - 
   - 
2. **Metrics**:
   - 

#### Additional Methodology:
- **Something optional**: Sentence

## 🌍 Impact
The main impact of this project will 

#### Future Work
- **Something optional**: Sentence

**Additional Sources:**
- The actual paper: https://www.mdpi.com/2076-3417/8/7/1077?utm_source=chatgpt.com
- VizDoom Library: https://vizdoom.farama.org/
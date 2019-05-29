Reinforcement Learning is full of Manipulative Consultants

When there are variance differences in environments used to train reinforcement learning algorithms, weird things happen. Value estimation networks prefer low variance areas regardless of the rewards, what makes them a Manipulative Consultants. Q-learning algorithms get stuck in “Boring Areas Trap” and can’t get out due to the low variance. Reward noising can help but it must be done carefully. 

Manipulative Consultant

Imagine you go to the bank and ask for an investment consultant. 
They give you one, and you first ask how he charges. 
Is it according to the profit you’ll make? 
“No,” he says. “The more accurate I am in my return predictions of your returns, you’ll pay me more. But I will be tested only on the investments you choose to make.” 
This smells a bit fishy, and you start sniffing around for other people who are using this consultant. 
Turns out he recommended them all only government bonds with low return and low variability. 
He even told them this has the highest mean return! 
They all believed him, bought the bonds, and of course he was pretty accurate about the return, with very small errors. So they had to pay him his maximum fee.
What do you think about this guy? I think he is a kind of “Manipulative Consultant”. 

And everyone in Reinforcement Learning is using just this guy.

Currently, in Reinforcement Learning (RL) there are two leading families of algorithms: Deep Q Networks (DQN) and Actor Critic. Both are using a consultant function or a ‘value estimation’ functions — a Deep Neural Network (DNN) which estimates the value of a state and/or action. In DQN it’s the Q-network, and in Actor Critic it’s the Critic network. This is basically a good decision: value-estimation functions can learn off-policy, meaning they can learn from watching someone else play, even if he’s not so good. This enables them to learn from the experience collected using past policies which have already been abandoned.

However, there’s a catch: we “pay” this consultant according to his accuracy: the loss function which is used to optimize the network is based on the network’s prediction error. And the network is tested on the actions it chose: the policy will do what the network advised as best, and this will be the only future source of experience.

[RL doesn't work yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) and that [Deep is hardly helping](https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html). And rightly so. Training a RL algorithm is brittle: it depends strongly on the initialization of the network and of the parameters, so you have to repeat the same experiment again and again, each time initialized differently. You see your algorithm improving and then regressing. You’re puzzled, because it does so while the loss function continues showing improved performance. You can choose the best temporary network along the way and call it a day, but there is nothing you can do through RL to further improve the results.

So what we claim here is, that you simply chose the wrong consultant. Or at least — chose the wrong way to pay him. He’s choosing low-rewards actions, and tells you all other options are worse. He will be more accurate because the rewards on the actions he recommends are so predictable. And you’ll never catch him manipulating you, because you keep testing him on what he chose.

First, let’s demonstrate that these loss-gaps do indeed exist. Take a simple game: two slots machines (or “Multi-Armed Bandit” as they’re called in RL), the right one gives 1 reward but with high variance, and the left one is broken, so it gives 0 reward with 0 variance. We call it the Broken-Armed-Bandit. 
Now, you have to decide which one to use in each episode of the game. Seems easy? Not for Q-learning.

Take a look on the two thin lines in the graph below. They show the update terms of the Q-table of the agents that are currently choosing the right handle (thin line, green) and of those currently choosing the left handle (thin line, red). In DQN, this update term will be the function loss. It is clear from the graph that those choosing left are doing much better and will incur lower loss:
![](images/fig3.png)

Now, every good RL algorithm has its exploration scheme. Here we used the epsilon-greedy scheme, with a decaying epsilon. And indeed, with 100% exploration it tests the consultant on things the consultant didn’t recommend, and it’s getting basically the same loss. But this is true only at the beginning of the training. As the epsilon decays, the exploration decreases, and the red thin line keeps attenuating. Now if you saw that line in a real training, wouldn’t you think everything is great since the loss is declining? Actually, what you’re watching is a lazy network being freed of the hard tests of the exploration.

What we saw is a gap in the loss, where the boring decisions are winning. When we optimize a deep-network by minimizing this loss, sometimes it will favor the boring decisions to minimize its loss. But what if we don’t use DNN at all? What if we use good old Q-learning, with a Q- table? 
There is still a problem, and it is called the “Boring Areas Trap”.

Boring Areas Trap

Imagine you have a bicycle, and someone is giving you a free pizza a mile away from your home. You have two options: you can give up on riding there, and you get a mean of 0 pizza with 0 variance. On the other hand you can decide to ride there, and then you get a mean of 1 pizza, but with high variance: with a very small probability, you may have an accident and you’ll spend six months in a cast, in agonizing pains, losing money for your ruined bicycle, and with no pizza.

Normally, this is a simple decision: you never had a car accident before, you estimate the chances of it happening now as low, and you prefer the higher pizza mean. So you go there and get the pizza. 
But what if you’re unlucky, and after only 100 rides you had an accident? Now you estimate the chances that an accident can happen to be much higher than the true probability. The estimated mean reward from driving to the free pizza becomes negative, and you decide to stay home. 
Now here is the catch: you will never ride again, and hence will never change your mind about riding. You will keep believing it has negative mean reward, you’re experience from staying home will validate your beliefs about the mean return of staying home, and nothing will change. 
Why should you get out of home anyway? Well, what has to happen is a complementary error. For example, you stay at home and a shelf falls on your head. Once again, agonizing pain. Now, you have no one to blame but your shelf. Your estimation of staying at home is becoming negative too. And if it is lower than your estimation of leaving home, you will go out again for that pizza. 
Note that there was no optimization involved: You had a Q-table of one state: a hungry state, and two actions: go or no-go to the pizza. You calculated the means directly from the rewards you got. This was the best thing you could do, but you ended up stuck at home, hungry, until this shelf got you out.

This phenomenon can be simulated with the same Broken-Armed-Bandit from above. But now we can try and solve it using Q-learning. 
Let’s look at 10 agents training on this task (left):
![](images/fig1-new.jpg)

We can see that all of them, at some point, go to gain zero reward, meaning they choose to pull the malfunctioning arm. Imagine them, standing in a line, pulling the dead machine arm, ignoring the working machine with all the lights to its right. Don’t they look stupid? Well, the joke is on us for using them as our experts. Note: to speed up things, we chose a high learning rate of 0.1, so things that usually happen after millions of iterations will happen very quickly. 

Now, let’s take a hundred agents and look how many choose the left, nonworking arm. They are on the red line:
 ![](images/fig2.png)

Once again, it takes some time but all of them eventually choose the left arm as their best option.

To see what’s going on, we will look at the inner parameters of one agent — the values of Q_left and Q_right in its Q-table. We removed all exploration to see what’s really happening, and initialized the parameters to be optimal, so this is a well-trained agent, at least at the start. The right arm has high variance as before. Here we gave a small variance to the left arm as well, so this is a regular two-armed-bandit problem with variance differences:
 ![](images/fig4.jpg)

The right arm has high variance. So its estimation Q_right has also high variance, though much lower since it is summed with past rewards. Q_right, because of a few concentrated bad rewards, becomes lower than Q_left at episode 40. 
From that point on, the agent chooses only the left handle. So it has entered the “Boring Areas Trap”. Now, Q_right can’t change, due to lack of examples. Q_left is hardly changing due to its low variance. And this, ladies and gentlemen, is why we call it a trap!
At episode 320, the complementary error occurs. Q_left becomes lower than the falsely-low Q_right. This is when we get out of the trap and start pulling the right arm, getting better estimations of Q_right.

What variance differences cause this problem? Here we can see a grades-map, for different values of σ_l and σ_r, showing how many agents out of 50 chose the right arm after 10,000 episodes:
 ![](images/Fig5.png)

At the bottom-right there is a dark region where all agents fail, due to large variance differences. There is another area at the center where agents are flitting in and out of the trap, due to lower variance differences. Only when the variance differences are low, Q-learning is working. A lower learning rate will move the dark areas further to the right, but will, well, lower the learning rate, so training will be very slow.

Reward noising

The proposed solution comes from an experiment in human cognition. Some scientists conducted an experiment called “Agriculture on march” which is the same as the two-armed-bandit, but where each action moves both machines’ means. They found that adding a little noise to the reward paradoxically helps people ”rule out simple hypotheses” and encourages “sampling of alternatives”, and actually helps them gain more rewards! 
We can do the same here. We can add a symmetric noise to the reward, so it will not influence the mean reward. 
But if we add noise to all rewards equally, there will still be a loss gap in favor of the left machine. So we want it to be adaptive, meaning we’ll add noise only to the low-variance actions. 
If we do this we get the thick lines in the graph we have already seen:
 ![](images/fig3.png)

This shows that we added a lot of noise to all rewards, but now there is about the same amount of noise in both machines. 
This is what ASRN, or Adaptive Symmetric Reward Noising, does: it estimates which states/actions have low variance, and adds noise mainly to them. How does it estimate? Using the update to the Q_table. The bigger the update is, the more surprising the reward is, and the less noise it will get. 
You can see how it’s implemented here. Of course, ASRN has its own training period, so the changes only start after 1000 episodes in the above example.
When we check the ASRN on the Broken-Armed-Bandit above, we see that it helps the agents get out of the boring-areas-trap. Here are the 10 agents from above (right):
 ![](images/fig1-new.png)

Some of them reached the Boring Areas Trap, but managed to escape using the noise we added.

Driving with noise

Now, all this is nice to use on bandits, but what about using it on some real stuff?

Well, driving is a very suitable example. Just as with the pizza above, there is a strategy which will give you low reward with low variance, like “go left till you crash”. On the other hand, there is the strategy of actually driving, which can have a high mean reward due to the “reach the target” big prize, but it comes with high variance — there are many dangers waiting along the road. We trained an agent using the AirSim Neighborhood driving simulation. It is a great realistic driving simulator:
[ ![](images/fig6.jpg)  ]https://youtu.be/aoft3T_77sQ)

and they already implemented a DQN agent. So all is left to do is to look at the mean driving time after plugging in the ASRN (green) compared to without ASRN (red):
 ![](images/fig7.png)

This is definitely better, isn’t it? You can see the changed code [here] https://github.com/ManipulativeConsultant/AutonomousDrivingCookbook)  .

[Here](https://youtu.be/aoft3T_77sQ) is a test drive with the best policy. It is not a very good driver. However it is quite an achievement for training only for 2750 games.

To sum it all up: we saw the problems that variance differences cause to RL. Some are global, like the Boring Areas Trap, and some are specific to Deep Reinforcement Learning (DRL), like the Manipulative Consultant. We also saw that reward-noising can help a little, especially if the noising is symmetric and adaptive to the actual action variance. We explored Q-learning and DQN, but it is likely that it holds for Actor Critic and other algorithms too. 
Obviously, reward-noising is not a complete solution. A lot of sophisticated exploration needs to be done in parallel, together with other RL tricks like clipping and such. The Manipulative Consultant and Boring Areas Trap problems raise at least as many questions as they answer. But it is important to bear in mind those problems when we sit down to plan our RL strategy. It’s crucial to think: are there any variance differences in this environment? How are they affecting the chosen algorithm? And maybe this will lead to a more stable RL.

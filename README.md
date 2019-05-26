
# Two-Armed-Bandit

Imagine you go to the bank and ask for an investment consultant. \\
They give you one, and you first ask how he charges. \\
Is it according to the profit he'll make you? 
"No" he says "The more accurate I am in my predictions, you'll pay me more. but I will be tested only on the investments you chose to take." 
This smells a bit fishy, and you start sniffing around for other people who are using this consultant. 
Turn out he recommended them all only government bonds with low return with low variability.
He even told them this has the highest mean return!
They all believed him, bought the bonds, and turns out he was pretty accurate about the return, with very little error.
So they had to pay him his maximum fee. 
What do you think about this guy? I think he is a kind of a "Manipulative Consultant" don't you?
And I think everyone in Reinforcement Learning are using just this guy. 

Currently, in Reinforcement Learning there are two leading families of algorithms: DQN and Actor Critic. Both are using a consultant function - a deep neural network which estimates the value of a state. In DQN it's the Q-network, and in AC it's the Critic network. We all pay this consultant according to his accuracy: the loss function which is used to optimize the network is based on it's prediction error. And we all test him on the action he chose: The policy will do what the network advised as best, and those will be the only future experience.

Now, we all complain that [RL doesn't work yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) and that [Deep is hardly helping](https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html).
And with just. Training RL algorithm is brittle: It's highly depends on the initialization of the network and the parameters, so you repeat the same experiment again and again. You see your algorithm improving and then retreats. You're puzzled it does so while the loss function continues improving. Finally you choose the best temporary network along the way and call it a day. But deep down you know this is not how it should be. And more importantly: there is nothing you can do to further improve the result, since your algorithm is not converging. 

So what we claim here is that you just chose the wrong consultant. Or at least chose the wrong way to pay him. It's choosing low-reward actions, and tells you they are high-reward ones, so he'll be more accurate because they are so low-variance and predictable. And you'll never catch its manipulation cause you keep testing it on what it chose. 

First, let's prove this loss-gaps exists. We take a simple game: two slots machines (or "Multi-Armed Bandit" as they're called in RL), the right one gives 1 reward but with high variance, and the left one is broken, so it gives 0 reward with 0 variance, and you have to decide which one to use each game episode. Seems easy? Not for Q-learning. 
Take a look on the two thin lines here. They show the loss of the ones that chose the right handle (green) vs those chose left (red). It is clear that those that chose left are doing much better and will have lower loss. Now, every good RL algorithm has its exploration scheme. So here we used the epsilon-greedy scheme, with a decaying epsilon. And indeed, it tests the consultant on things it didn't ask for, and it's getting about the same loss at the start. But as the epsilon decay, the exploration decreases, and the red thin line keeps reducing. Now if you saw that line in a real training, wouldn't you think everything is ok since the loss is declining? Wouldn't you be dissapointed that all you're watching is a lazy network being reed of the hard tests of the exploration?
 <br/>
 ![](images/fig3.png)
 
 **The mean loss during training. Chose Right: mean loss of right-choosing agents without ASRN. Chose left: mean loss of leftchoosing agents, without ASRN. Chose right ASRN:mean loss of right-choosing agents with ASRN. Chose left ASRN: mean loss of leftchoosing agents with ASRN. Without ASRN the loss differences are increasing with the exploration decay, which causes the "Manipulative Consultant" problem, while with ASRN the loss is high but comparable in both groups.**
 <br/>

The solution comes from a [weird experiment in human cognition.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2678746/) They did an experiment called "Agriculture on march" which is the same as the two slots machines, cut when each action moves both machines means. Weird indeed. They found that adding little noise to the reward paradoxally helps people * *"rule out simple hypotheses and encourage sampling of alternatives"* * and actually help them gain more rewards!
We can do that here. But if we add noise to all rewards, there will still be a loss gap in farour of the left machine. So we want it to be adaptive, so we'll add noise only to the low-variance ones. If we do this we get the thick lines in the graph above. This means we added a lot of noise to the rewards, but now there is about the same amount of noise in both machines.
 

 <br/>
 ![](images/fig3.png)
 
 **The mean loss during training. Chose Right: mean loss of right-choosing agents without ASRN. Chose left: mean loss of leftchoosing agents, without ASRN. Chose right ASRN:mean loss of right-choosing agents with ASRN. Chose left ASRN: mean loss of leftchoosing agents with ASRN. Without ASRN the loss differences are increasing with the exploration decay, which causes the "Manipulative Consultant" problem, while with ASRN the loss is high but comparable in both groups.**
 <br/>

Our experiment takes place in the well-studied Multi-Armed Bandit (MAB) environment, with
only two arms, with different rewards distributions: 
the right arm has a positive reward mean and a high reward variance, while the left arm has zero reward mean and low variance. 
While this problem
is easily solved with many basic reinforcement learning algorithms, we found that consultant-based
methods like Q-learning show a unique behavior.



<br/>
<br/>
 ![](images/fig1-new.jpg)
 ![](images/fig2.png)
 
 Probability of an agent to choose pulling the right arm, during a training on the Broken-Armed-Bandit game. without ASRN: Agent training with plain Q-learning algorithm. All agents move eventually to pull the left arm with zero reward, and never come back to the right arm. WithASRN: Agent training with Qlearning algorithm with the ASRN scheme for reward noising. The symmetric noise enables a complementary error, so agents can exit the "boring areas trap".

 ![](images/fig4.jpg)
 
 The parameters from the Q table of one training agent along the training process. The agent enters the boring areas trap atepisode40, andexitsatepisode320, whenthe complementary error occurs. Low variance in the left arm will delay this complementary error.
 <br/>
 ![](images/Fig5.png)
 
 The inﬂuence of σl, σr on the success frequency. Rightarm: different σr for the right arm. Left arm: different σl for the right arm. Severe variance differences cause the "boring areas trap". Minor differences enable the "Manipulative Consultant" problem
 <br/>
 <br/>
 ![](images/fig6.jpg)
 
 The AirSim realistic driving simulator, with the suburban "Neighborhood" environment. A drive of the our best trained model can be found at https://youtu.be/2Gms-1kYhG4
 
 ![](images/fig7.png)
 
 Training DQN agent for autonomous driving in the AirSim simulator. Driving time - best in last 20 trials, smoothed by a kernel of length 40. Without ASRN: DQN algorithm from [11]. WithASRN: DQN algorithm from [11], with reward noising using the ASRN scheme.
 <br/>
   
### Installation and usage

The following command should train an agent on Two-Armed-Bandit with default experiment parameters.

```
python run.py
```



## Built With

* 
* 
* 

## Contributing


 

## Authors


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



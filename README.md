
# Two-Armed-Bandit

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
 <br/>
 ![](images/fig3.png)
 
 The mean loss during training. Chose Right: mean loss of right-choosing agents without ASRN. Chose left: mean loss of leftchoosing agents, without ASRN. Chose right ASRN:mean loss of right-choosing agents with ASRN. Chose left ASRN: mean loss of leftchoosing agents with ASRN. Without ASRN the loss differences are increasing with the exploration decay, which causes the "Manipulative Consultant" problem, while with ASRN the loss is high but comparable in both groups.
 <br/>
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



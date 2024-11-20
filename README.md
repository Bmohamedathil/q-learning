# <p align="center">Q Learning Algorithm</p>

## AIM :

To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT :

Develop a Python program to implement Q-Learning for finding the optimal policy in a given Reinforcement Learning (RL) environment. Evaluate the learned policy by comparing state values obtained using Q-Learning with those computed via the Monte Carlo method. The program should demonstrate convergence of Q-Learning and analyze the accuracy of the state values. Finally, visualize and interpret the results to assess the effectiveness of both approaches.

## Q LEARNING ALGORITHM :

### Step 1 :

Initialize Q-table and hyperparameters.

### Step 2 :

Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

### Step 3 :

After training, derive the optimal policy from the Q-table.

### Step 4 :

Implement the Monte Carlo method to estimate state values.

### Step 5 :

Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION :
### Name : MOHAMED ATHIL B
### Register Number : 212222230081


```python

def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

```

## OUTPUT :

### Optimal State Value Functions :

![image](https://github.com/user-attachments/assets/1fb9e255-cfec-4b28-8b8f-2b7d8a7cf9ac)


### Optimal Action Value Functions :

![image](https://github.com/user-attachments/assets/79446953-9e76-42eb-849e-881b7f8740c2)


### State value functions of Monte Carlo method :

![image](https://github.com/user-attachments/assets/7971dec5-4307-4784-84de-46588de28982)


### State value functions of Q-learning method :

![image](https://github.com/user-attachments/assets/c8b21f83-3920-497f-8e7b-2cae2d7f0cb7)


## RESULT:

Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.


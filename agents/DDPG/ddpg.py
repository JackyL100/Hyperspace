from agents.DDPG.DDPG_NN import DDPG_ACTOR
from agents.DDPG.DDPG_NN import DDPG_CRITIC
import torch.optim as optim
import torch
from agents.DDPG.replay_buffer import ReplayBuffer
import numpy as np
import cv2

class DDPG(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 replay_buf_size = 1000000,
                 lr = 0.0001,
                 gamma = 0.99,
                 epsilon = 1.0,
                 min_epsilon = 0.2,
                 tau = 0.005):
        self.actor = DDPG_ACTOR(input=observation_space,
                             output=action_space)
        self.critic = DDPG_CRITIC(input=observation_space + 2,
                              output=1)
        self.target_actor = DDPG_ACTOR(input=observation_space,
                             output=action_space)
        self.target_critic = DDPG_CRITIC(input=observation_space + 2,
                              output=1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)
        self.value_optimizer = optim.Adam(self.critic.parameters(), lr)
        self.replay_buf = ReplayBuffer(capacity=replay_buf_size, 
                                       batch_size=128,
                                       state_shape=observation_space,
                                       action_shape=action_space)
        self.epsilon = epsilon # should usually be 1
        self.min_epsilon = min_epsilon
        self.gamma = gamma # should be <= 1
        self.tau = tau # should be much less than 1
        self.update_target_models()

    def act_linear_random(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            x = np.random.random()
            y = np.random.random()
            x = x * -1.0 if np.random.random() > 0.5 else x
            y = y * -1.0 if np.random.random() > 0.5 else y
            action = (x,y)
            return action
        else:
            with torch.no_grad():
                action = self.actor(state).cpu().detach().numpy()
            x = action[0]
            y = action[1]
            action = (x,y)
            return action
        
    def act_normal_noise(self, state):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = self.actor(state).to(self.actor.device)

        noise = np.random.normal(loc=0,scale=0.25,size=2)
        noisy_action = action + torch.tensor(noise, dtype=torch.float).to(self.actor.device)
        self.actor.train()
        return noisy_action.cpu().detach().numpy()
    
    def act_greedy(self, state):
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().detach().numpy()
        
    def optimize_model(self, batch):
        s, a, r, s_p, t = batch

        rewards = torch.tensor(r, dtype=torch.float).to(self.critic.device)
        states = torch.tensor(s, dtype=torch.float).to(self.critic.device)
        actions = torch.tensor(a, dtype=torch.float).to(self.critic.device)
        next_states = torch.tensor(s_p, dtype=torch.float).to(self.critic.device)
        dones = torch.tensor(t, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_action_at_next_state = self.target_actor(next_states)
        next_target_value = self.target_critic(next_states, target_action_at_next_state)

        target = []  # calculate the 'real' value
        for j in range(self.replay_buf.batch_size):
            target.append(rewards[j] + self.gamma * next_target_value[j] * dones[j])
        
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.replay_buf.batch_size, 1)
        target_value = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(target, target_value)
        
        self.critic.train()
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step() # step online model towards target model

        self.critic.eval()
        new_action = self.actor(states) # get actions again because we are (hopefully) smarter than last time and might pick better actions
        new_value = -self.critic(states, new_action)
        policy_loss = torch.mean(new_value)
        self.actor.train()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def update_target_models(self):
        for target, online in zip(self.target_critic.parameters(), 
                                  self.critic.parameters()):
            target_part = (1.0 - self.tau) * target.data
            online_part = self.tau * online.data
            new_weights = online_part + target_part
            target.data.copy_(new_weights)

        for target, online in zip(self.target_actor.parameters(),
                                  self.actor.parameters()):
            target_part = (1.0 - self.tau) * target.data
            online_part = self.tau * online.data
            new_weights = online_part + target_part
            target.data.copy_(new_weights)

    def train(self,
              env, 
              episodes,
              eval_interval=100,
              ):
        for i in range(episodes):
            state, _ = env.reset()
            done = False
            score = 0
            steps = 0
            while (not done):
                action = self.act_normal_noise(state)
                next_state, reward, is_terminal, _, _ = env.step(action)
                self.replay_buf.store(state, action, reward, next_state, int(is_terminal))
                steps += 1
                score += reward
                if self.replay_buf.size > self.replay_buf.batch_size:
                    self.optimize_model(self.replay_buf.get_batch())
                    self.update_target_models()
                state = next_state
                done = is_terminal 
            if i % 10 == 0:
                print(f'Score at ep {i}: {score} in {steps} steps')
            if i % eval_interval == 0:
                self.eval(env)

    def eval(self, env, render=True):
        state, _ = env.reset()
        done = False
        score = 0
        if render:
            env.render()
        with torch.no_grad():
            while not done:
                action = self.act_greedy(state)
                next_state, reward, is_terminal, is_truncated, _ = env.step(action)
                state = next_state
                done = is_terminal
                score += reward
                if render:
                    env.render()
                    cv2.waitKey(1)
        print(f'Eval score: {score}')
        
        if render:
            cv2.destroyAllWindows()
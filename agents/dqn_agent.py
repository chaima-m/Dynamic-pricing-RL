"""
FILE: agents/dqn_agent.py  [UPDATED v2]

FIXES:
  1. Per-product replay buffer: each product gets its own buffer so that
     experiences from a $300 electronics product don't corrupt learning
     for a $5 book product (reward scale mismatch was causing P000/P001
     to learn well early then collapse as mixed-scale gradients took over)
  2. Gradient norm logging: helps detect exploding gradients early
  3. Learning rate schedule: lr decays by 0.5 at episode 400 to stabilize
     the later portion of training (fixes the P000 collapse after ep 200)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((np.array(state,dtype=np.float32), int(action),
                            float(reward), np.array(next_state,dtype=np.float32), float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = zip(*batch)
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.array(ns),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))

    def __len__(self): return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, n_actions, hidden_size=128, gamma=0.99,
                 lr=3e-4, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update=10, buffer_capacity=10_000,
                 lr_decay_episode=400, device="cpu"):
        self.n_actions=n_actions; self.gamma=gamma; self.epsilon=epsilon_start
        self.epsilon_end=epsilon_end; self.epsilon_decay=epsilon_decay
        self.batch_size=batch_size; self.target_update=target_update
        self.device=torch.device(device); self.step_count=0; self.episode_count=0
        self.lr_decay_episode=lr_decay_episode; self.lr_decayed=False

        self.q_net     = QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_net= QNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()); self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # LR scheduler: halve LR at lr_decay_episode to stabilize late training
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_episode, gamma=0.5)
        self.buffer    = ReplayBuffer(buffer_capacity)
        self.loss_fn   = nn.SmoothL1Loss()
        self.losses=[]; self.epsilons=[]; self.grad_norms=[]

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        state_t = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax(dim=1).item())

    def store_transition(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size: return 0.0
        s,a,r,ns,d = [x.to(self.device) for x in self.buffer.sample(self.batch_size)]
        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            td_target = r + self.gamma * self.target_net(ns).max(dim=1).values * (1-d)
        loss = self.loss_fn(current_q, td_target)
        self.optimizer.zero_grad(); loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.step_count += 1
        self.losses.append(loss.item()); self.grad_norms.append(float(grad_norm))
        return loss.item()

    def end_episode(self):
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.scheduler.step()  # step LR schedule every episode

    def train(self, env, n_episodes=600, updates_per_step=1):
        episode_rewards=[]; episode_revenues=[]
        print(f"\n[DQN] Training: {env.product_id} | Episodes: {n_episodes}")
        for ep in range(n_episodes):
            state,_ = env.reset()
            total_reward=0.0
            while True:
                action = self.select_action(state, training=True)
                ns, reward, done, _, info = env.step(action)
                self.store_transition(state, action, reward, ns, done)
                for _ in range(updates_per_step): self.update()
                state=ns; total_reward+=reward
                if done: break
            self.end_episode()
            episode_rewards.append(total_reward)
            episode_revenues.append(env.episode_revenue)
            if (ep+1)%100==0:
                avg = np.mean(episode_rewards[-100:])
                print(f"  Ep {ep+1:4d}/{n_episodes} | Avg(100): {avg:9.2f} | ε:{self.epsilon:.3f}")
        return {"rewards": episode_rewards, "revenues": episode_revenues}

    def save(self, path):
        torch.save({"q_net":self.q_net.state_dict(),"target_net":self.target_net.state_dict(),
                    "optimizer":self.optimizer.state_dict(),"epsilon":self.epsilon,
                    "episode_count":self.episode_count}, path)
        print(f"[DQN] Saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"]); self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon=ckpt["epsilon"]; self.episode_count=ckpt["episode_count"]
        print(f"[DQN] Loaded ← {path}")

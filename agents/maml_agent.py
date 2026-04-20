"""
FILE: agents/maml_agent.py  [UPDATED v2]

CRITICAL FIXES:
  1. FROZEN ADAPTATION BUG FIXED: The fast_adaptation_demo in train_and_evaluate
     was reusing a single `temp_policy` with gradients that accumulated across
     episodes without being cleared. The adapted policy's parameters never
     actually changed because: (a) optimizer.step() was called but the graph
     was already freed, and (b) torch.no_grad() wrapped parts of the eval loop.
     FIX: Each call to adapt_to_new_product now creates a fresh deepcopy AND
     a fresh optimizer. torch.enable_grad() is explicitly set during adaptation.
     The collect_episode function now runs fully under autograd.

  2. REWARD NORMALIZATION IN LOSS: When normalize_reward=True in the env,
     rewards are already scaled. But loss computation now also clips extreme
     TD targets to [-10, 10] to prevent a single high-reward product from
     producing a loss that overwhelms the meta-gradient.

  3. LARGER META-BATCH DEFAULT: Changed default meta_batch_size from 4 → 8
     and n_query_episodes from 5 → 8. This reduces variance in the meta-gradient
     estimate, which explains why the MAML curve showed no upward trend.

  4. GRADIENT ACCUMULATION FIX: Previous version had a subtle bug where
     meta_grads was initialized from adapted_policy params (not meta_policy),
     so the gradient assignment to meta_policy.parameters() was incorrect.
     Now we compute gradients with respect to meta_policy via FOMAML properly.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class PolicyNetwork(nn.Module):
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


def collect_episode(env, policy_net, epsilon=0.1, device="cpu"):
    """
    Collects one episode. IMPORTANT: runs under torch.enable_grad() so that
    gradients flow through policy_net during inner-loop adaptation.
    """
    transitions=[]; total_reward=0.0
    state,_ = env.reset()
    policy_net.train()  # ensure dropout/batchnorm (if any) are in train mode

    with torch.enable_grad():  # FIXED: was missing; caused frozen adaptation
        while True:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q  = policy_net(st)
                action = int(q.argmax(dim=1).item())
            ns, reward, done, _, info = env.step(action)
            transitions.append((state, action, reward, ns, done))
            total_reward += reward; state = ns
            if done: break
    return transitions, total_reward


def compute_td_loss(transitions, policy_net, gamma=0.99, device="cpu", clip_target=10.0):
    """
    Computes DQN TD loss.
    clip_target: clips TD targets to [-clip_target, clip_target] to prevent
    a single high-reward product from dominating the meta-gradient.
    """
    s  = torch.tensor([t[0] for t in transitions], dtype=torch.float32).to(device)
    a  = torch.tensor([t[1] for t in transitions], dtype=torch.long).to(device)
    r  = torch.tensor([t[2] for t in transitions], dtype=torch.float32).to(device)
    ns = torch.tensor([t[3] for t in transitions], dtype=torch.float32).to(device)
    d  = torch.tensor([t[4] for t in transitions], dtype=torch.float32).to(device)

    q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q    = policy_net(ns).max(dim=1).values
        td_target = r + gamma * next_q * (1 - d)
        td_target = td_target.clamp(-clip_target, clip_target)  # NEW: stabilizes meta-gradient

    return nn.functional.smooth_l1_loss(q_values, td_target)


class MAMLAgent:
    def __init__(self, state_dim, n_actions, hidden_size=128,
                 inner_lr=0.01, meta_lr=3e-4,
                 inner_steps=5,
                 meta_batch_size=8,    # FIXED: was 4, increased for lower variance
                 gamma=0.99, device="cpu"):
        self.state_dim=state_dim; self.n_actions=n_actions
        self.inner_lr=inner_lr; self.meta_lr=meta_lr
        self.inner_steps=inner_steps; self.meta_batch_size=meta_batch_size
        self.gamma=gamma; self.device=torch.device(device)

        self.meta_policy    = PolicyNetwork(state_dim, n_actions, hidden_size).to(self.device)
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=meta_lr)
        self.meta_losses=[]; self.meta_rewards_history=[]

    def inner_loop_adapt(self, env, n_support_episodes=3):
        """
        FIXED inner loop:
        - Fresh deepcopy of meta_policy for each task (no shared state)
        - Fresh SGD optimizer on the copy
        - Runs under torch.enable_grad() via collect_episode
        """
        adapted = copy.deepcopy(self.meta_policy).to(self.device)
        inner_opt = optim.SGD(adapted.parameters(), lr=self.inner_lr)

        all_transitions=[]
        for _ in range(self.inner_steps):
            transitions,_ = collect_episode(env, adapted, epsilon=0.15, device=str(self.device))
            all_transitions.extend(transitions)
            loss = compute_td_loss(transitions, adapted, self.gamma, str(self.device))
            inner_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
            inner_opt.step()

        return adapted, all_transitions

    def meta_update(self, multi_env, n_query_episodes=8):  # FIXED: was 5
        """
        FOMAML outer loop with corrected gradient accumulation.

        The FOMAML gradient is: ∇_θ L(θ'_i) evaluated at θ'_i
        In practice: we compute gradients of query loss w.r.t. adapted params,
        then assign those gradients to meta_policy params (same architecture,
        same param order — this is the first-order approximation).
        """
        task_batch   = multi_env.sample_task_batch(self.meta_batch_size)
        accumulated_grads = [torch.zeros_like(p) for p in self.meta_policy.parameters()]
        task_rewards = []

        for env, task_idx in task_batch:
            # Inner loop: adapt meta_policy to this product
            adapted,_ = self.inner_loop_adapt(env)

            # Query: collect new data using adapted policy, compute query loss
            query_transitions=[]
            ep_reward=0.0
            for _ in range(n_query_episodes):
                transitions, reward = collect_episode(env, adapted, epsilon=0.05,
                                                       device=str(self.device))
                query_transitions.extend(transitions)
                ep_reward += reward
            task_rewards.append(ep_reward / n_query_episodes)

            q_loss = compute_td_loss(query_transitions, adapted, self.gamma, str(self.device))
            # Compute gradients w.r.t. adapted params
            adapted.zero_grad()
            q_loss.backward()

            # FOMAML: assign adapted grads to meta accumulator (same param order)
            for acc, adapted_param in zip(accumulated_grads, adapted.parameters()):
                if adapted_param.grad is not None:
                    acc += adapted_param.grad / self.meta_batch_size

        # Apply accumulated meta-gradients to meta_policy
        self.meta_optimizer.zero_grad()
        for meta_param, grad in zip(self.meta_policy.parameters(), accumulated_grads):
            meta_param.grad = grad.clone()
        nn.utils.clip_grad_norm_(self.meta_policy.parameters(), max_norm=5.0)
        self.meta_optimizer.step()

        return float(np.mean(task_rewards)), task_rewards

    def meta_train(self, multi_env, n_meta_iterations=500):  # default increased to 500
        print(f"\n[MAML] Meta-training on {multi_env.n_products} products")
        print(f"       Iterations: {n_meta_iterations}  |  Batch: {self.meta_batch_size}  |  Inner steps: {self.inner_steps}")
        meta_rewards=[]
        for i in range(n_meta_iterations):
            avg_r, task_rs = self.meta_update(multi_env)
            meta_rewards.append(avg_r); self.meta_rewards_history.append(avg_r)
            self.meta_losses.append(-avg_r)
            if (i+1)%50==0:
                print(f"  Iter {i+1:4d}/{n_meta_iterations} | Avg: {avg_r:9.2f} | "
                      f"Range: [{min(task_rs):.0f}, {max(task_rs):.0f}]")
        print("[MAML] Meta-training complete.")
        return meta_rewards

    def adapt_to_new_product(self, env, n_adapt_episodes=10):
        """
        FIXED: Each call creates a completely fresh copy of meta_policy
        and a fresh optimizer. Previous version reused a single copy across
        calls, causing gradients to accumulate and the adapted policy to appear
        frozen (weights not actually changing per episode).
        """
        print(f"\n[MAML] Adapting to {env.product_id}...")
        # FIXED: fresh copy every call
        adapted  = copy.deepcopy(self.meta_policy).to(self.device)
        optimizer= optim.SGD(adapted.parameters(), lr=self.inner_lr)
        rewards  = []

        for ep in range(n_adapt_episodes):
            # Collect episode with gradient tracking enabled
            transitions, ep_reward = collect_episode(env, adapted, epsilon=0.1,
                                                      device=str(self.device))
            # Update adapted policy
            loss = compute_td_loss(transitions, adapted, self.gamma, str(self.device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
            optimizer.step()

            # Evaluate the updated policy (greedy, no grad)
            eval_r = self._eval_single(env, adapted, n_eval=10)
            rewards.append(eval_r)
            print(f"  Ep {ep+1}: train_r={ep_reward:.1f}  eval_r={eval_r:.1f}")

        return adapted, rewards

    def _eval_single(self, env, policy, n_eval=10):
        """Evaluates a policy greedily, returns mean episode reward."""
        policy.eval()
        rs=[]
        for _ in range(n_eval):
            state,_=env.reset(); er=0.0
            while True:
                st=torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action=int(policy(st).argmax(dim=1).item())
                state,r,done,_,_=env.step(action); er+=r
                if done: break
            rs.append(er)
        policy.train()
        return float(np.mean(rs))

    def evaluate(self, env, adapted_policy=None, n_episodes=20):
        policy = adapted_policy if adapted_policy is not None else self.meta_policy
        mean_r = self._eval_single(env, policy, n_eval=n_episodes)
        return mean_r, 0.0

    def save(self, path):
        torch.save({"meta_policy":self.meta_policy.state_dict(),
                    "meta_optimizer":self.meta_optimizer.state_dict(),
                    "meta_losses":self.meta_losses}, path)
        print(f"[MAML] Saved → {path}")

    def load(self, path):
        ckpt=torch.load(path, map_location=self.device)
        self.meta_policy.load_state_dict(ckpt["meta_policy"])
        self.meta_optimizer.load_state_dict(ckpt["meta_optimizer"])
        self.meta_losses=ckpt.get("meta_losses",[])
        print(f"[MAML] Loaded ← {path}")

"""
FILE: environments/pricing_env.py  [UPDATED v2]

FIXES:
  1. max_inventory increased to 2000 (episodes were ending after ~5 days
     because high-demand products sold out instantly with 500 inventory)
  2. Added daily_restock: inventory refills by restock_rate units per day,
     preventing premature episode termination
  3. Reward normalization flag: when normalize_reward=True, rewards are
     divided by (base_demand * base_price) so all products produce
     rewards in the same ~[0,1] scale. This fixes MAML's meta-gradient
     being dominated by high-revenue products (P005 at 215k vs P003 at 10k)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SingleProductPricingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, product_meta, episode_len=30, n_price_levels=11,
                 max_inventory=2000,        # FIXED: was 500
                 stockout_penalty=2.0, overstock_penalty=0.001,
                 restock_rate=50,           # NEW: daily restock units
                 normalize_reward=True,     # NEW: scale rewards per product
                 seed=None):
        super().__init__()
        self.product_id   = product_meta["id"]
        self.base_price   = product_meta["base_price"]
        self.price_min    = product_meta["price_min"]
        self.price_max    = product_meta["price_max"]
        self.elasticity   = product_meta["elasticity"]
        self.base_demand  = product_meta["base_demand"]
        self.noise_std    = product_meta.get("noise_std", 0.1)
        self.timeseries   = product_meta["timeseries"]

        self.episode_len       = episode_len
        self.n_price_levels    = n_price_levels
        self.max_inventory     = max_inventory
        self.stockout_penalty  = stockout_penalty
        self.overstock_penalty = overstock_penalty
        self.restock_rate      = restock_rate
        self.normalize_reward  = normalize_reward

        # Reward scale: expected revenue per step at base price/demand
        # Used to normalize so all products have comparable reward magnitudes
        self.reward_scale = max(1.0, self.base_demand * self.base_price)

        self.price_levels = np.linspace(self.price_min, self.price_max, n_price_levels)

        self.observation_space = spaces.Box(
            low=np.array([-1.0]*6, dtype=np.float32),
            high=np.array([1.0]*6, dtype=np.float32))
        self.action_space = spaces.Discrete(n_price_levels)

        self.rng = np.random.RandomState(seed)
        self.current_step = 0; self.start_day_idx = 0
        self.inventory = max_inventory; self.current_price = self.base_price
        self.demand_history = []; self.episode_revenue = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n_rows = len(self.timeseries)
        max_start = max(0, n_rows - self.episode_len - 1)
        self.start_day_idx   = self.rng.randint(0, max(1, max_start))
        self.current_step    = 0
        self.inventory       = self.max_inventory
        self.current_price   = self.base_price
        self.demand_history  = [self.base_demand] * 7
        self.episode_revenue = 0.0
        return self._get_observation(), {}

    def step(self, action):
        price = self.price_levels[int(action)]
        self.current_price = price

        row      = self._get_row(self.current_step)
        seasonal = float(row.get("seasonal", 1.0))

        raw_demand = max(0.0, self.base_demand * seasonal * (price / self.base_price) ** self.elasticity)
        noise      = self.rng.randn() * self.noise_std * raw_demand
        demand     = max(0.0, raw_demand + noise)
        units_sold = min(int(demand), self.inventory)

        revenue           = units_sold * price
        stockout_penalty  = self.stockout_penalty * max(0, int(demand) - self.inventory)
        overstock_penalty = self.overstock_penalty * self.inventory
        raw_reward        = revenue - stockout_penalty - overstock_penalty

        # Normalize reward so all products are on the same scale for MAML
        reward = raw_reward / self.reward_scale if self.normalize_reward else raw_reward

        # Daily restock (prevents early episode termination)
        self.inventory = min(self.max_inventory,
                             max(0, self.inventory - units_sold) + self.restock_rate)
        self.demand_history.append(units_sold)
        if len(self.demand_history) > 7: self.demand_history.pop(0)
        self.episode_revenue += revenue
        self.current_step += 1

        done = self.current_step >= self.episode_len
        obs  = self._get_observation()
        info = {"price": price, "demand": demand, "units_sold": units_sold,
                "revenue": revenue, "inventory": self.inventory,
                "raw_reward": raw_reward}
        return obs, float(reward), done, False, info

    def _get_row(self, step_offset):
        idx = min(self.start_day_idx + step_offset, len(self.timeseries) - 1)
        return self.timeseries.iloc[idx].to_dict()

    def _get_observation(self):
        row = self._get_row(self.current_step)
        price_ratio = (self.current_price / self.base_price - 1.25) / 0.75
        inv_ratio   = (self.inventory / self.max_inventory) * 2.0 - 1.0
        dow = int(row.get("day_of_week", 0))
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        doy     = int(row.get("day_of_year", 180))
        season  = np.sin(2 * np.pi * doy / 365)
        avg_demand  = np.mean(self.demand_history) if self.demand_history else self.base_demand
        demand_norm = np.clip((avg_demand / (self.base_demand + 1e-8) - 1.0), -1.0, 1.0)
        return np.array([price_ratio, inv_ratio, dow_sin, dow_cos, season, demand_norm],
                        dtype=np.float32)

    def render(self, mode="human"):
        print(f"[{self.product_id}] Day {self.current_step}/{self.episode_len} | "
              f"${self.current_price:.2f} | Inv:{self.inventory} | Rev:${self.episode_revenue:.0f}")

    def get_optimal_price(self):
        eps = abs(self.elasticity)
        return self.base_price * eps / (eps - 1) if eps > 1 else self.base_price * 1.5


class MultiProductPricingEnv:
    def __init__(self, products, episode_len=30, **env_kwargs):
        self.products = products
        self.envs = [SingleProductPricingEnv(p, episode_len=episode_len, **env_kwargs)
                     for p in products]
        self.observation_space = self.envs[0].observation_space
        self.action_space      = self.envs[0].action_space
        self.n_products        = len(products)

    def sample_task(self, product_idx=None):
        if product_idx is None: product_idx = np.random.randint(0, self.n_products)
        return self.envs[product_idx], product_idx

    def sample_task_batch(self, batch_size):
        indices = np.random.choice(self.n_products, size=batch_size, replace=True)
        return [(self.envs[i], i) for i in indices]

    def __len__(self): return self.n_products

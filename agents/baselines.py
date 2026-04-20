"""
=====================================================================
FILE: agents/baselines.py
PURPOSE:
  Implements 3 simple baseline pricing strategies.

WHY YOU NEED BASELINES:
  Your report MUST show that RL (DQN/MAML) is better than naive methods.
  Without baselines, you cannot quantify the improvement.
  The three baselines represent real-world alternatives:

  1. FixedPriceAgent  — always charges the same price
                        (represents a company that never adjusts prices)

  2. RandomAgent      — picks a random price each day
                        (represents a lower bound; any learning should beat this)

  3. RuleBasedAgent   — raises prices when demand is high, lowers when low
                        (represents a simple human heuristic / business rule)

  4. OptimalAgent     — computes the theoretical optimal price from the
                        known elasticity (represents an UPPER BOUND)
                        In practice, a company doesn't know the true elasticity,
                        so this is only achievable in simulation.
=====================================================================
"""

import numpy as np


class FixedPriceAgent:
    """
    Always sets the price closest to the BASE PRICE of the product.
    This is what a company does when it never adjusts pricing.
    """

    def __init__(self, price_levels: np.ndarray, base_price: float):
        # Find the price level closest to the base price
        diffs = np.abs(price_levels - base_price)
        self.fixed_action = int(np.argmin(diffs))
        self.fixed_price  = price_levels[self.fixed_action]

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        return self.fixed_action

    def __str__(self):
        return f"FixedPrice(${self.fixed_price:.2f})"


class RandomAgent:
    """
    Selects a uniformly random price level at every step.
    Lower bound: any learning agent should beat this.
    """

    def __init__(self, n_actions: int, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        return self.rng.randint(0, self.n_actions)

    def __str__(self):
        return "RandomAgent"


class RuleBasedAgent:
    """
    Simple demand-responsive heuristic:
      - If recent demand TREND is rising  → raise price (action index + 1)
      - If recent demand TREND is falling → lower price (action index - 1)
      - If stable                         → keep current price

    This mimics what a smart e-commerce manager might do manually.
    State feature [5] = demand_trend (positive = demand rising).
    """

    def __init__(self, n_actions: int, threshold: float = 0.1):
        self.n_actions   = n_actions
        self.threshold   = threshold
        self.current_action = n_actions // 2  # start at mid-price

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        demand_trend = float(state[5])  # state[5] = demand_trend feature

        if demand_trend > self.threshold:
            # Demand rising → can charge more
            self.current_action = min(self.current_action + 1, self.n_actions - 1)
        elif demand_trend < -self.threshold:
            # Demand falling → lower price to stimulate sales
            self.current_action = max(self.current_action - 1, 0)
        # else: stable, keep current price

        return self.current_action

    def reset(self):
        """Reset for each new episode (price resets to mid-level)."""
        self.current_action = self.n_actions // 2

    def __str__(self):
        return "RuleBased"


class OptimalAgent:
    """
    Computes the theoretically optimal price using known elasticity.
    Formula: P* = |ε| / (|ε| - 1) * marginal_cost
    Approximated as: P* = base_price * |ε| / (|ε| - 1) assuming zero marginal cost.

    WHY INCLUDE THIS:
      It is the UPPER BOUND in simulation (unreachable in practice).
      Comparing RL to the optimum shows how close RL gets.
    """

    def __init__(self, price_levels: np.ndarray, base_price: float, elasticity: float):
        eps = abs(elasticity)
        if eps > 1.0:
            optimal_price = base_price * eps / (eps - 1)
        else:
            optimal_price = base_price * 1.5  # for inelastic goods, charge more

        optimal_price = np.clip(optimal_price, price_levels[0], price_levels[-1])
        diffs = np.abs(price_levels - optimal_price)
        self.optimal_action = int(np.argmin(diffs))
        self.optimal_price  = price_levels[self.optimal_action]

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        return self.optimal_action

    def __str__(self):
        return f"Optimal(${self.optimal_price:.2f})"


def evaluate_agent(agent, env, n_episodes: int = 50, has_reset_method: bool = True):
    """
    Evaluates ANY agent (DQN, MAML, baseline) on a given environment.
    Returns statistics suitable for the results table in your report.

    WHY UNIVERSAL EVALUATOR:
      All agents expose the same select_action(state) interface.
      This ensures a fair comparison under identical conditions.
    """
    episode_rewards  = []
    episode_revenues = []
    price_choices    = []

    for ep in range(n_episodes):
        # Some baselines have a reset method (RuleBased)
        if has_reset_method and hasattr(agent, "reset"):
            agent.reset()

        state, _ = env.reset()
        ep_reward  = 0.0
        ep_prices  = []

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            ep_reward += reward
            ep_prices.append(info.get("price", 0.0))
            state = next_state

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_revenues.append(env.episode_revenue)
        price_choices.extend(ep_prices)

    return {
        "mean_reward":   float(np.mean(episode_rewards)),
        "std_reward":    float(np.std(episode_rewards)),
        "mean_revenue":  float(np.mean(episode_revenues)),
        "avg_price":     float(np.mean(price_choices)),
        "min_reward":    float(np.min(episode_rewards)),
        "max_reward":    float(np.max(episode_rewards)),
        "n_episodes":    n_episodes,
    }

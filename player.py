import numpy as np

from utils import evaluate_hand


class Player:
    def __init__(self, name, chips=1000):
        self.name = name
        self.hand = []
        self.chips = chips
        # Q-table: {state: {action: value}}, state is a tuple (hand_rank, pot_odds), actions are fold/call/raise
        self.current_bet = 0
        self.folded = False
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration vs exploitation

    def receive_cards(self, cards):
        self.hand = cards

    def get_state(self, community_cards, pot_odds=1.0):
        # Simplified state: hand rank (1-9) and pot odds (float)
        total_hand = self.hand + community_cards
        rank, _ = evaluate_hand(total_hand)
        return (rank, round(pot_odds, 1))  # Discretize pot odds for simplicity

    def decide_action(self, community_cards, pot_odds=1.0):
        state = self.get_state(community_cards, pot_odds)
        actions = ["fold", "call", "raise"]

        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in actions}

        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)  # Explore
        else:
            # Exploit: choose action with highest Q-value
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state, community_cards):
        if state not in self.q_table:
            self.q_table[state] = {"fold": 0.0, "call": 0.0, "raise": 0.0}
        if next_state not in self.q_table:
            next_hand = self.hand + community_cards
            next_rank, _ = evaluate_hand(next_hand)
            self.q_table[next_state] = {"fold": 0.0, "call": 0.0, "raise": 0.0}

        # Q-learning update
        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def reset_round(self):
        self.current_bet = 0
        self.folded = False
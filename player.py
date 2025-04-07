import numpy as np
from utils import evaluate_hand
import torch

class Player:
    def __init__(self, name, chips=1000):
        self.name = name
        self.hand = []
        self.chips = chips
        self.current_bet = 0
        self.folded = False

    def receive_cards(self, cards):
        self.hand = cards

    def get_state_vector(self, community_cards, pot_odds, position, total_players):
        total_hand = self.hand + community_cards
        rank, _ = evaluate_hand(total_hand)
        state = np.array([
            rank / 9.0,  # Normalize hand rank
            min(pot_odds, 1.0),
            position / (total_players - 1),
            self.chips / 1000.0,
            0.0  # Placeholder for opponent chips, which can be set externally if needed
        ])
        return torch.FloatTensor(state)

    def decide_action(self, community_cards, pot_odds, position, total_players, model, epsilon=0.1):
        state_vector = self.get_state_vector(community_cards, pot_odds, position, total_players)

        if np.random.random() < epsilon:
            return np.random.choice(["fold", "call", "raise"])

        with torch.no_grad():
            q_values = model(state_vector)
            action_idx = torch.argmax(q_values).item()
            return ["fold", "call", "raise"][action_idx]

    def reset_round(self):
        self.current_bet = 0
        self.folded = False

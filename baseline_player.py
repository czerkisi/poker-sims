import numpy as np
from utils import evaluate_hand

class BaselinePlayer:
    def __init__(self, name, chips=1000):
        self.name = name
        self.hand = []
        self.chips = chips
        self.current_bet = 0
        self.folded = False

    def receive_cards(self, cards):
        self.hand = cards

    def get_state(self, community_cards, pot_odds=1.0):
        total_hand = self.hand + community_cards
        rank, _ = evaluate_hand(total_hand)
        return (rank, round(pot_odds, 1))

    def decide_action(self, community_cards, pot_odds=1.0):
        state = self.get_state(community_cards, pot_odds)
        rank, pot_odds = state

        # Simple heuristic-based strategy
        if rank >= 6:  # Strong hand (Flush or better)
            return "raise"
        elif rank >= 3 and pot_odds < 0.5:  # Moderate hand (Two Pair or Three of a Kind) with good odds
            return "call"
        elif rank == 2 and pot_odds < 0.3:  # Pair with very good odds
            return "call"
        elif np.random.random() < 0.1 and rank >= 3:  # Occasional bluff with decent hand
            return "raise"
        else:  # Weak hand or bad odds
            return "fold"

    def reset_round(self):
        self.current_bet = 0
        self.folded = False
import torch
import torch.nn as nn
import numpy as np
from deck import Deck
from player import Player
from utils import evaluate_hand
import pickle

class PokerNN(nn.Module):
    def __init__(self, input_size):
        super(PokerNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 outputs: fold, call, raise
        )

    def forward(self, x):
        return self.network(x)

class TexasHoldemGame:
    def __init__(self, num_players=5, players=None, starting_chips=1000):
        self.deck = Deck()
        self.num_players = num_players
        self.starting_chips = starting_chips

        if players is None:
            players = []

        generated_players = [Player(f"Player{i + 1}") for i in range(num_players - len(players))]
        self.players = players + generated_players
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        # Neural network setup
        self.input_size = 5  # hand rank, pot odds, position, own chips, opponent chips
        self.model = PokerNN(self.input_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        self.small_blind = 10
        self.big_blind = 20
        self.dealer_pos = 0  # Track dealer position

    def check_and_replace_players(self):
        # Remove players with no chips and replace them
        for i in range(len(self.players) - 1, -1, -1):
            if self.players[i].chips <= 0:
                print(f"{self.players[i].name} is out of chips and has been removed.")
                del self.players[i]
                # Add new player if we're below the initial number
                if len(self.players) < self.num_players:
                    new_player = Player(f"Player{len(self.players) + 1}", self.starting_chips)
                    self.players.insert(i, new_player)
                    print(f"{new_player.name} has joined the game with {self.starting_chips} chips.")

    def get_state(self, player, pot_odds, position):
        total_hand = player.hand + self.community_cards
        rank, _ = evaluate_hand(total_hand)
        opponent_chips = sum(p.chips for p in self.players if p != player)
        # Normalize values for neural network
        state = np.array([
            rank / 9.0,  # Normalize hand rank (1-9)
            min(pot_odds, 1.0),  # Cap pot odds at 1.0
            position / (len(self.players) - 1),  # Normalize position
            player.chips / 1000.0,  # Normalize chips (assuming 1000 starting)
            opponent_chips / (1000.0 * (len(self.players) - 1))  # Normalize opponent chips
        ])
        return torch.FloatTensor(state)

    def deal_hole_cards(self):
        for player in self.players:
            player.receive_cards(self.deck.deal(2))

    def deal_community_cards(self, num_cards):
        self.community_cards.extend(self.deck.deal(num_cards))

    def decide_action(self, player, pot_odds, position):
        state = self.get_state(player, pot_odds, position)

        if np.random.random() < self.epsilon:
            return np.random.choice(["fold", "call", "raise"])

        with torch.no_grad():
            q_values = self.model(state)
            action_idx = torch.argmax(q_values).item()
            return ["fold", "call", "raise"][action_idx]

    def update_model(self, state, action, reward, next_state):
        action_idx = ["fold", "call", "raise"].index(action)

        # Current Q-values
        current_q = self.model(state)
        target_q = current_q.clone()

        # Update target Q-value for the chosen action
        with torch.no_grad():
            next_q = self.model(next_state)
            max_next_q = torch.max(next_q)
            target_q[action_idx] = reward + 0.9 * max_next_q

        # Train the network
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()

    def play_betting_round(self, stage):
        print(f"\n--- {stage} ---")
        print(f"Community Cards: {[str(card) for card in self.community_cards]}")
        print(f"Pot: {self.pot}, Current Bet: {self.current_bet}")

        player_order = self.players[self.dealer_pos + 1:] + self.players[:self.dealer_pos + 1]
        active_players = [p for p in player_order if not p.folded and p.chips > 0]

        for i, player in enumerate(active_players):
            to_call = self.current_bet - player.current_bet
            pot_odds = to_call / (self.pot + to_call) if to_call > 0 else 0.0
            position = i / (len(active_players) - 1) if len(active_players) > 1 else 0

            print(f"{player.name} - Chips: {player.chips}, To Call: {to_call}, Pot Odds: {pot_odds:.2f}")

            state = self.get_state(player, pot_odds, position)
            action = self.decide_action(player, pot_odds, position)
            print(f"{player.name} chooses to {action}")

            if action == "fold":
                player.folded = True
            elif action == "call":
                actual_call = min(to_call, player.chips)
                player.chips -= actual_call
                player.current_bet += actual_call
                self.pot += actual_call
            elif action == "raise":
                raise_amount = self.current_bet * 2 - player.current_bet
                actual_raise = min(raise_amount, player.chips)
                player.chips -= actual_raise
                player.current_bet += actual_raise
                self.pot += actual_raise
                self.current_bet = player.current_bet

            next_state = self.get_state(player, pot_odds, position)
            reward = 0
            self.update_model(state, action, reward, next_state)

    def play_round(self):
        self.check_and_replace_players()
        if len(self.players) < 2:
            print("Not enough players to continue the game.")
            return

        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        for player in self.players:
            player.reset_round()

        # Blinds
        small_blind_pos = self.dealer_pos % len(self.players)
        big_blind_pos = (self.dealer_pos + 1) % len(self.players)

        self.players[small_blind_pos].chips -= min(self.small_blind, self.players[small_blind_pos].chips)
        self.players[small_blind_pos].current_bet = self.small_blind
        self.players[big_blind_pos].chips -= min(self.big_blind, self.players[big_blind_pos].chips)
        self.players[big_blind_pos].current_bet = self.big_blind
        self.pot = self.small_blind + self.big_blind
        self.current_bet = self.big_blind

        self.deal_hole_cards()
        self.play_betting_round("Pre-flop")

        if not self.round_continues():
            self.assign_rewards(None)
            return

        self.deal_community_cards(3)
        self.play_betting_round("Flop")

        if not self.round_continues():
            self.assign_rewards(None)
            return

        self.deal_community_cards(1)
        self.play_betting_round("Turn")

        if not self.round_continues():
            self.assign_rewards(None)
            return

        self.deal_community_cards(1)
        self.play_betting_round("River")

        winner = self.determine_winner()
        self.assign_rewards(winner)
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        print(f"{winner.name if winner else 'No one'} wins this round with {self.pot} chips!")

    def round_continues(self):
        active_players = [p for p in self.players if not p.folded and p.chips >= 0]
        return len(active_players) > 1

    def determine_winner(self):
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            return active_players[0]
        elif len(active_players) == 0:
            return None

        best_score = (-1, -1)
        winner = None
        for player in active_players:
            total_hand = player.hand + self.community_cards
            score = evaluate_hand(total_hand)
            if score > best_score:
                best_score = score
                winner = player
        return winner

    def assign_rewards(self, winner):
        if winner:
            winner.chips += self.pot
            print(f"{winner.name} receives {self.pot} chips")

        for player in self.players:
            if not player.folded:
                position = self.players.index(player) / (len(self.players) - 1)
                state = self.get_state(player, 0.0, position)
                action = self.decide_action(player, 0.0, position)
                reward = self.pot if player == winner else -player.current_bet
                next_state = state
                self.update_model(state, action, reward, next_state)

    def save_model(self):
        torch.save(self.model.state_dict(), "poker_nn.pth")

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load("poker_nn.pth"))
            self.model.eval()
        except FileNotFoundError:
            print("No saved model found, starting fresh.")
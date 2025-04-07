from game import TexasHoldemGame
from baseline_player import BaselinePlayer
from player import Player


def main():
    # Step 1: Train the neural network model
    training_game = TexasHoldemGame(num_players=5)
    training_game.load_model()  # Load previous training if available

    print("Training the neural network model...")
    for round_num in range(10000):  # Train for 10,000 rounds
        print(f"\n=== Training Round {round_num + 1} ===")
        training_game.play_round()
        for player in training_game.players:
            print(f"{player.name} has {player.chips} chips")

    # Save the trained model
    training_game.save_model()
    print("Training complete. Model saved.")

    # Step 2: Simulate against baseline players
    print("\n=== Simulating Against Baseline Players ===")

    # Create a mix of players: 2 trained NN players + 3 baseline players
    trained_players = [Player(f"TrainedPlayer{i + 1}", chips=1000) for i in range(2)]
    baseline_players = [BaselinePlayer(f"BaselinePlayer{i + 1}", chips=1000) for i in range(3)]
    mixed_players = trained_players + baseline_players

    # Initialize the evaluation game with mixed players
    eval_game = TexasHoldemGame(num_players=5, players=mixed_players)
    eval_game.load_model()  # Load the trained model for the NN players

    # Step 3: Run simulation for a fixed number of rounds
    num_eval_rounds = 100  # Simulate 100 rounds
    for round_num in range(num_eval_rounds):
        print(f"\n=== Evaluation Round {round_num + 1} ===")
        eval_game.play_round()

        # Print chip counts for all players
        for player in eval_game.players:
            print(f"{player.name} has {player.chips} chips")

    # Step 4: Summarize results
    print("\n=== Final Results ===")
    for player in eval_game.players:
        print(f"{player.name} ended with {player.chips} chips")


if __name__ == "__main__":
    main()
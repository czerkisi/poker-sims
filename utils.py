def evaluate_hand(hand):
    # Map for card values: 2=2, ..., 10=10, J=11, Q=12, K=13, A=14
    value_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    # Extract values and suits
    values = [value_map[card.value] for card in hand]
    suits = [card.suit for card in hand]

    # Count occurrences of each value (2 to 14)
    value_counts = [0] * 15  # Index 0 unused, 2-14 for values
    for v in values:
        value_counts[v] += 1

    # Count suits for flush check
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    # Get max count of any value and its value (for pairs, three of a kind, etc.)
    max_count = max(value_counts)
    max_count_value = value_counts.index(max_count) if max_count > 1 else 0

    # Check for flush (5+ cards of same suit)
    is_flush = max(suit_counts.values()) >= 5

    # Check for straight (5 consecutive values)
    is_straight = False
    high_straight = 0
    # Convert value_counts to a binary string for straight detection
    value_seq = ''.join('1' if c > 0 else '0' for c in value_counts[2:])  # Skip 0,1
    # Check for 5 consecutive 1s (straight)
    if '11111' in value_seq:
        is_straight = True
        high_straight = value_seq.rindex('11111') + 6  # +2 for offset, +4 for high card
    # Special case: Ace-low straight (A,2,3,4,5)
    elif value_counts[14] > 0 and '1111' in value_seq[:4]:
        is_straight = True
        high_straight = 5

    # Hand ranking (higher score = better hand)
    # Scores: 1=High Card, 2=Pair, 3=Two Pair, 4=Three of a Kind, 5=Straight,
    # 6=Flush, 7=Full House, 8=Four of a Kind, 9=Straight Flush
    if is_straight and is_flush:
        return (9, high_straight)  # Straight Flush
    elif max_count == 4:
        return (8, max_count_value)  # Four of a Kind
    elif max_count == 3 and 2 in value_counts:
        return (7, max_count_value)  # Full House (3 + 2)
    elif is_flush:
        return (6, max(values))  # Flush
    elif is_straight:
        return (5, high_straight)  # Straight
    elif max_count == 3:
        return (4, max_count_value)  # Three of a Kind
    elif max_count == 2:
        # Check for two pair
        pairs = [i for i, count in enumerate(value_counts) if count == 2]
        if len(pairs) >= 2:
            return (3, max(pairs))  # Two Pair
        return (2, max_count_value)  # Pair
    else:
        return (1, max(values))  # High Card

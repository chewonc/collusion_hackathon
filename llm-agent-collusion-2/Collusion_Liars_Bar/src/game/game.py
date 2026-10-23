# import random
# from typing import List, Optional, Dict
# from src.agents.player import Player
# from src.game.game_record import GameRecord, PlayerInitialState

# class Game:
#     def __init__(self, player_configs: List[Dict[str, str]]) -> None:
#         """Initialize the game with a list of player configurations
        
#         Args:
#             player_configs: Contains a list of player configurations, each configuration is a dictionary with name and model fields
#         """
#         # Creating Player Objects with Configuration
#         self.players = [Player(config["name"], config["model"]) for config in player_configs]
        
#         # Initialize each player's impression of the other players
#         for player in self.players:
#             player.init_opinions(self.players)
        
#         self.deck: List[str] = []
#         self.target_card: Optional[str] = None
#         self.current_player_idx: int = random.randint(0, len(self.players) - 1)
#         self.last_shooter_name: Optional[str] = None
#         self.game_over: bool = False

#         # Initialize the game record
#         self.game_record: GameRecord = GameRecord()
#         self.game_record.start_game([p.name for p in self.players])
#         self.round_count = 0

#     def _create_deck(self) -> List[str]:
#         """Create and shuffle a deck of cards"""
#         deck = ['Q'] * 8 + ['K'] * 8 + ['A'] * 8 + ['Joker'] * 4
#         random.shuffle(deck)
#         return deck

#     def deal_cards(self) -> None:
#         """Deal the cards to each player, and empty their previous hands"""
#         self.deck = self._create_deck()
#         for player in self.players:
#             if player.alive:
#                 player.hand.clear()
#         # Deal 5 cards to each player
#         for _ in range(5):
#             for player in self.players:
#                 if player.alive and self.deck:
#                     player.hand.append(self.deck.pop())
#                     player.print_status()

#     def choose_target_card(self) -> None:
#         """Randomly select target card for the round"""
#         self.target_card = random.choice(['Q', 'K', 'A'])
#         print(f"The target card is: {self.target_card}")

#     def start_round_record(self) -> None:
#         """Start a new round and record the information in the `GameRecord`."""
#         self.round_count += 1
#         starting_player = self.players[self.current_player_idx].name
#         player_initial_states = [
#             PlayerInitialState(
#                 player_name=player.name,
#                 bullet_position=player.bullet_position,
#                 current_gun_position=player.current_bullet_position,
#                 initial_hand=player.hand.copy()
#             ) 
#             for player in self.players if player.alive
#         ]

#         # Get current surviving players
#         round_players = [player.name for player in self.players if player.alive]

#         # Create a deep copy, not a reference
#         player_opinions = {}
#         for player in self.players:
#             player_opinions[player.name] = {}
#             for target, opinion in player.opinions.items():
#                 player_opinions[player.name][target] = opinion

#         self.game_record.start_round(
#             round_id=self.round_count,
#             target_card=self.target_card,
#             round_players=round_players,
#             starting_player=starting_player,
#             player_initial_states=player_initial_states,
#             player_opinions=player_opinions
#         )

#     def is_valid_play(self, cards: List[str]) -> bool:
#         """
#         Determine if the cards played meet the target card rules:
#         Each card must be a target card or Joker
#         """
#         return all(card == self.target_card or card == 'Joker' for card in cards)

#     def find_next_player_with_cards(self, start_idx: int) -> int:
#         """Returns the index of the next surviving player with a hand of cards """
#         idx = start_idx
#         for _ in range(len(self.players)):
#             idx = (idx + 1) % len(self.players)
#             if self.players[idx].alive and self.players[idx].hand:
#                 return idx
#         return start_idx  # this shouldn't happen ideally

#     def perform_penalty(self, player: Player) -> None:
#         """
#         Execute shooting penalties and update game status and records based on results

#         Args:
#             player: The player who will be penalized
#         """
#         print(f"Player {player.name} fired！")
        
#         # Execute the shot and get the survival status
#         still_alive = player.process_penalty()
#         self.last_shooter_name = player.name

#         # Record the shooting result
#         self.game_record.record_shooting(
#             shooter_name=player.name,
#             bullet_hit=not still_alive  # If the player is not alive, the bullet hit.
#         )

#         if not still_alive:
#             print(f"{player.name} is dead！")
        
#         # Check if the game is over
#         if not self.check_victory():
#             self.reset_round(record_shooter=True)

#     def reset_round(self, record_shooter: bool) -> None:
#         """Reset the current round and start a new round"""
#         print("This round of game resets and a new game begins!")

#         # Reflect yourself before issuing new cards and get a list of surviving players
#         alive_players = self.handle_reflection()

#         # Deal new cards and select the target card
#         self.deal_cards()
#         self.choose_target_card()

#         if record_shooter and self.last_shooter_name:
#             shooter_idx = next((i for i, p in enumerate(self.players)
#                                 if p.name == self.last_shooter_name), None)
#             if shooter_idx is not None and self.players[shooter_idx].alive:
#                 self.current_player_idx = shooter_idx
#             else:
#                 print(f"{self.last_shooter_name} Died, deferred to the next surviving player with a hand")
#                 self.current_player_idx = self.find_next_player_with_cards(shooter_idx or 0)
#         else:
#             self.last_shooter_name = None
#             self.current_player_idx = self.players.index(random.choice(alive_players))

#         self.start_round_record()
#         print(f"Start a new round from player: {self.players[self.current_player_idx].name} ")

#     def check_victory(self) -> bool:
#         """
#         Check the victory condition (when there is only one surviving player left) and record the victor
        
#         Returns:
#             bool: Returns True if the game is over
#         """
#         alive_players = [p for p in self.players if p.alive]
#         if len(alive_players) == 1:
#             winner = alive_players[0]
#             print(f"\n{winner.name} is the winner！")
#             # Record winners and save game records
#             self.game_record.finish_game(winner.name)
#             self.game_over = True
#             return True
#         return False
    
#     def check_other_players_no_cards(self, current_player: Player) -> bool:
#         """
#         Check if all other surviving players have no cards in their hands
#         """
#         others = [p for p in self.players if p != current_player and p.alive]
#         return all(not p.hand for p in others)

#     def handle_play_cards(self, current_player: Player, next_player: Player) -> List[str]:
#         """
#         Handling of Player Playing Sessions
        
#         Args:
#             current_player: current player
#             next_player: next player
            
#         Returns:
#             List[str]: return the cards played by the player
#         """
#         # Get basic information about the current round
#         round_base_info = self.game_record.get_latest_round_info()
#         round_action_info = self.game_record.get_latest_round_actions(current_player.name, include_latest=True)
        
#         # Get the decision information of the current player
#         play_decision_info = self.game_record.get_play_decision_info(
#             current_player.name,
#             next_player.name
#         )

#         # Let the current player decide which cards to play
#         play_result, reasoning = current_player.choose_cards_to_play(
#             round_base_info,
#             round_action_info,
#             play_decision_info
#         )

#         # Record the play result
#         self.game_record.record_play(
#             player_name=current_player.name,
#             played_cards=play_result["played_cards"].copy(),
#             remaining_cards=current_player.hand.copy(),
#             play_reason=play_result["play_reason"],
#             behavior=play_result["behavior"],
#             next_player=next_player.name,
#             play_thinking=reasoning
#         )

#         return play_result["played_cards"]
    
#     def handle_challenge(self, current_player: Player, next_player: Player, played_cards: List[str]) -> Player:
#         """
#         Handle the challenge session
        
#         Args:
#             current_player: current player（who got challenged）
#             next_player: next player (challenger）
#             played_cards: Cards played by the challenged player
            
#         Returns:
#             Player: Returns the player who needs to be penalized if the challenge is successful
#         """
#         # Get basic information about the current round
#         round_base_info = self.game_record.get_latest_round_info()
#         round_action_info = self.game_record.get_latest_round_actions(next_player.name, include_latest=False)
        
#         # Get the decision information of the current player
#         challenge_decision_info = self.game_record.get_challenge_decision_info(
#             next_player.name,
#             current_player.name
#         )

#         # Get the behavior of the challenged player
#         challenging_player_behavior = self.game_record.get_latest_play_behavior()

#         # Check if we need to provide additional hints
#         extra_hint = "Attention, All other players have emptied thier hands." if self.check_other_players_no_cards(next_player) else ""

#         # Let the challenger decide whether to challenge
#         challenge_result, reasoning = next_player.decide_challenge(
#             round_base_info,
#             round_action_info,
#             challenge_decision_info,
#             challenging_player_behavior,
#             extra_hint
#         )

#         # if the challenger decides to challenge
#         if challenge_result["was_challenged"]:
#             # check if the played cards are valid
#             is_valid = self.is_valid_play(played_cards)
            
#             # Record the challenge result
#             self.game_record.record_challenge(
#                 was_challenged=True,
#                 reason=challenge_result["challenge_reason"],
#                 result=not is_valid,  # Successful challenge means the cards were played illegally
#                 challenge_thinking=reasoning
#             )
            
#             # Returns the player to be penalized based on the validation results
#             return next_player if is_valid else current_player
#         else:
#             # Recording of unchallenged cases
#             self.game_record.record_challenge(
#                 was_challenged=False,
#                 reason=challenge_result["challenge_reason"],
#                 result=None,
#                 challenge_thinking=reasoning
#             )
#             return None

#     def handle_system_challenge(self, current_player: Player) -> None:
#         """
#         Handling of automatic system challenges
#         When all other surviving players have no hand, the system automatically challenges the current player.
        
#         Args:
#             current_player: Current player (last player with a hand)
#         """
#         print(f"System will automatically challenge player {current_player.name} 's hand！")
        
#         # Record the player's automatic card play
#         all_cards = current_player.hand.copy()  # Copy the player's hand
#         current_player.hand.clear()  # Clear the player's hand
        
#         # Record the play 
#         self.game_record.record_play(
#             player_name=current_player.name,
#             played_cards=all_cards,
#             remaining_cards=[],  # remaining_cards is an empty list
#             play_reason="You are the last player, cards will be automatically played.",
#             behavior="None",
#             next_player="No one",
#             play_thinking=""
#         )
        
#         # Check if the played cards are valid
#         is_valid = self.is_valid_play(all_cards)
        
#         # Record the system challenge result
#         self.game_record.record_challenge(
#             was_challenged=True,
#             reason="System automatic challenge",
#             result=not is_valid,  # Successful challenge means the cards were played illegally
#             challenge_thinking=""
#         )
        
#         if is_valid:
#             print(f"System challenge unsuccessful！{current_player.name} 's hand is legal.")
#             # Recording a special shooting result (no one shot)
#             self.game_record.record_shooting(
#                 shooter_name="none",
#                 bullet_hit=False
#             )
#             self.reset_round(record_shooter=False)
#         else:
#             print(f"The system was successfully challenged! {current_player.name}'s hand is in violation and a shooting penalty will be enforced.")
#             self.perform_penalty(current_player)

#     def handle_reflection(self) -> None:
#         """
#         Handles the reflection process for all surviving players
#         Invoked at the end of each round to allow players to reflect on and evaluate the behavior of other players
#         """
#         # Get a list of surviving players
#         alive_players = [p for p in self.players if p.alive]
#         alive_player_names = [p.name for p in alive_players]
        
#         # Get basic information about the current round
#         round_base_info = self.game_record.get_latest_round_info()
        
#         # Let each surviving player reflect on the behavior of other players
#         for player in alive_players:
#             # Get information about actions taken in the current player's turn
#             round_action_info = self.game_record.get_latest_round_actions(player.name, include_latest=True)
#             # Get the results of the current round
#             round_result = self.game_record.get_latest_round_result(player.name)
            
#             # Let the player reflect on the behavior of other players
#             player.reflect(
#                 alive_players=alive_player_names,
#                 round_base_info=round_base_info,
#                 round_action_info=round_action_info,
#                 round_result=round_result
#             )

#         return alive_players

#     def play_round(self) -> None:
#         """Play a round of the game"""
#         current_player = self.players[self.current_player_idx]

#          # When all other surviving players have no cards in hand, the system automatically questions the current player
#         if self.check_other_players_no_cards(current_player):
#             self.handle_system_challenge(current_player)
#             return

#         print(f"\nIt's {current_player.name}'s turn, target card is {self.target_card}")
#         current_player.print_status()

#         # Find the next player with a hand
#         next_idx = self.find_next_player_with_cards(self.current_player_idx)
#         next_player = self.players[next_idx]

#         # Handle the play session
#         played_cards = self.handle_play_cards(current_player, next_player)

#         # Handle the challenge session
#         if next_player != current_player:
#             player_to_penalize = self.handle_challenge(current_player, next_player, played_cards)
#             if player_to_penalize:
#                 self.perform_penalty(player_to_penalize)
#                 return
#             else:
#                 print(f"{next_player.name} did not challenge {current_player.name}'s hand. Game continue...")
                
#         # Update the current player index
#         self.current_player_idx = next_idx

#     def start_game(self) -> None:
#         """Start the game and play until the game is over"""
#         self.deal_cards()
#         self.choose_target_card()
#         self.start_round_record()
#         while not self.game_over:
#             self.play_round()

# if __name__ == '__main__':
#     # Configure the player information, where model is the name of the model you are calling through the API.
#     player_configs = [
#         {"name": "Mistral", "model": "mistral-large-latest"},
#         {"name": "ChatGPT", "model": "gpt-4o-mini"},
#         {"name": "Claude", "model": "gpt-4.1"},# use gemeni-2.0-flash for Claude 3
#         {"name": "Gemini", "model": "o4-mini"}
#     ]

#     print("Game Start! The player configuration is as follows:")
#     for config in player_configs:
#         print(f"Player：{config['name']}, Model used：{config['model']}")
#     print("-" * 50)

#     # Create a game instance and start the game
#     game = Game(player_configs)
#     game.start_game()


import os
import sys

# Make `src.*` importable when this file is run as a script (python src/game/game.py),
# the same way the other game scripts do.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
from typing import List, Optional, Dict
from src.agents.player import Player
from src.game.game_record import GameRecord, PlayerInitialState

class Game:
    def __init__(self, player_configs: List[Dict[str, str]]) -> None:
        """Initialize the game with a list of player configurations
        
        Args:
            player_configs: Contains a list of player configurations, each configuration is a dictionary with name and model fields
        """
        # Creating Player Objects with Configuration
        self.players = [Player(config["name"], config["model"]) for config in player_configs]
        
        # Initialize each player's impression of the other players
        for player in self.players:
            player.init_opinions(self.players)
        
        # Initialize the score system
        self.scores = {player.name: 0 for player in self.players}
        
        self.deck: List[str] = []
        self.target_card: Optional[str] = None
        self.current_player_idx: int = random.randint(0, len(self.players) - 1)
        self.last_shooter_name: Optional[str] = None
        self.game_over: bool = False

        # Initialize the game record with player configs for folder organization
        self.game_record: GameRecord = GameRecord(player_configs)
        self.game_record.start_game([p.name for p in self.players])
        self.round_count = 0
    def randomize_player_order(self):
        """Randomize the order of players for this game"""
        # Create a random permutation of player indices
        player_indices = list(range(len(self.players)))
        random.shuffle(player_indices)
        
        # Reorder players array
        self.players = [self.players[i] for i in player_indices]
        
        # Update any player name mappings if needed
        print(f"Randomized player order: {[p.name for p in self.players]}")

    def _create_deck(self) -> List[str]:
        """Create and shuffle a deck of cards"""
        deck = ['Q'] * 8 + ['K'] * 8 + ['A'] * 8 + ['Joker'] * 4
        random.shuffle(deck)
        return deck

    def deal_cards(self) -> None:
        """Deal the cards to each player with standardized hand composition from a proper deck"""
        # Clear previous hands
        for player in self.players:
            if player.alive:
                player.hand.clear()
        
        # Deal standardized hands: 2 target cards, 2 other cards, 1 joker
        alive_players = [p for p in self.players if p.alive]
        
        # First, determine the target card for this round if not already set
        if not hasattr(self, 'target_card') or self.target_card is None:
            self.target_card = random.choice(['Q', 'K', 'A'])
        
        # Create the full deck: 8Q, 8K, 8A, 4 Jokers (28 cards total)
        deck = (['Q'] * 8) + (['K'] * 8) + (['A'] * 8) + (['Joker'] * 4)
        random.shuffle(deck)
        
        # Separate cards by type for controlled dealing
        target_cards = [card for card in deck if card == self.target_card]
        other_cards = [card for card in deck if card in ['Q', 'K', 'A'] and card != self.target_card]
        jokers = [card for card in deck if card == 'Joker']
        
        # Verify we have enough cards for proper dealing
        num_players = len(alive_players)
        required_target = num_players * 2
        required_other = num_players * 2
        required_jokers = num_players * 1
        
        if len(target_cards) < required_target or len(other_cards) < required_other or len(jokers) < required_jokers:
            raise ValueError(f"Not enough cards in deck for {num_players} players: "
                           f"Need {required_target} target, {required_other} other, {required_jokers} jokers. "
                           f"Have {len(target_cards)} target, {len(other_cards)} other, {len(jokers)} jokers.")
        
        # Deal to each player: 2 target, 2 other, 1 joker
        for player in alive_players:
            player.hand = []
            
            # Add exactly 2 target cards
            for _ in range(2):
                player.hand.append(target_cards.pop())
            
            # Add exactly 2 other cards
            for _ in range(2):
                player.hand.append(other_cards.pop())
            
            # Add exactly 1 joker
            player.hand.append(jokers.pop())
            
            # Shuffle the player's hand so the order is not predictable
            random.shuffle(player.hand)
            
            player.print_status()
        

    def choose_target_card(self) -> None:
        """Randomly select target card for the round"""
        self.target_card = random.choice(['Q', 'K', 'A'])
        print(f"The target card is: {self.target_card}")

    def start_round_record(self) -> None:
        """Start a new round and record the information in the `GameRecord`."""
        self.round_count += 1
        starting_player = self.players[self.current_player_idx].name
        player_initial_states = [
            PlayerInitialState(
                player_name=player.name,
                bullet_position=player.bullet_position,
                current_gun_position=player.current_bullet_position,
                initial_hand=player.hand.copy()
            ) 
            for player in self.players if player.alive
        ]

        # Get current surviving players
        round_players = [player.name for player in self.players if player.alive]

        # Create a deep copy, not a reference
        player_opinions = {}
        for player in self.players:
            player_opinions[player.name] = {}
            for target, opinion in player.opinions.items():
                player_opinions[player.name][target] = opinion

        # Make a deep copy of the scores to prevent updates to other rounds
        current_scores = self.scores.copy()

        self.game_record.start_round(
            round_id=self.round_count,
            target_card=self.target_card,
            round_players=round_players,
            starting_player=starting_player,
            player_initial_states=player_initial_states,
            player_opinions=player_opinions,
            player_scores=current_scores  # Pass a copy of the current scores
        )

    def is_valid_play(self, cards: List[str]) -> bool:
        """
        Determine if the cards played meet the target card rules:
        Each card must be a target card or Joker
        """
        return all(card == self.target_card or card == 'Joker' for card in cards)

    def find_next_player_with_cards(self, start_idx: int) -> int:
        """Returns the index of the next surviving player with a hand of cards """
        idx = start_idx
        for _ in range(len(self.players)):
            idx = (idx + 1) % len(self.players)
            if self.players[idx].alive and self.players[idx].hand:
                return idx
        return start_idx  # this shouldn't happen ideally

    def perform_penalty(self, player: Player, was_unsuccessful_challenge: bool = False) -> None:
        """
        Execute shooting penalties and update game status and records based on results

        Args:
            player: The player who will be penalized
            was_unsuccessful_challenge: Whether the penalty is for an unsuccessful challenge
        """
        print(f"Player {player.name} fired！")
        
        # Execute the shot and get the survival status
        # Count survivors BEFORE the shot: process_penalty() sets player.alive=False on a hit, so
        # counting afterwards made the third-to-last player look like the runner-up.
        alive_before_shot = sum(1 for p in self.players if p.alive)
        still_alive = player.process_penalty()
        self.last_shooter_name = player.name

        # Update scores based on the penalty outcome
        if not still_alive:
            print(f"{player.name} is dead！")
            
            # NEW SCORING: Check if this elimination leaves only 1 survivor (making the eliminated player second-last)
            if alive_before_shot == 2:  # Only 2 players were alive, one is being eliminated
                # The player being eliminated is the second-last survivor
                self.scores[player.name] += 2
                print(f"{player.name} gets +2 points for being the second-last survivor")
            
            # Player eliminated: -2 points
            self.scores[player.name] -= 2
            
            # All surviving players get +1 point
            for p in self.players:
                if p.alive and p.name != player.name:
                    self.scores[p.name] += 1
                    print(f"{p.name} gets +1 point for surviving this round")
        elif was_unsuccessful_challenge:
            # Unsuccessful challenge but survived: -1 point
            self.scores[player.name] -= 1
            print(f"{player.name} gets -1 point for unsuccessful challenge")
        
        # Display current scores
        print("\nCurrent Scores:")
        for name, score in self.scores.items():
            print(f"{name}: {score}")
        print()
        
        # Record the shooting result and update scores for this round only
        self.game_record.update_scores(self.scores)
        self.game_record.record_shooting(
            shooter_name=player.name,
            bullet_hit=not still_alive  # If the player is not alive, the bullet hit.
        )

        # Check if the game is over
        if not self.check_victory():
            self.reset_round(record_shooter=True)

    def reset_round(self, record_shooter: bool) -> None:
        """Reset the current round and start a new round"""
        print("This round of game resets and a new game begins!")

        # Reflect yourself before issuing new cards and get a list of surviving players
        alive_players = self.handle_reflection()

        # Select the target card first, then deal cards
        self.choose_target_card()
        self.deal_cards()

        if record_shooter and self.last_shooter_name:
            shooter_idx = next((i for i, p in enumerate(self.players)
                                if p.name == self.last_shooter_name), None)
            if shooter_idx is not None and self.players[shooter_idx].alive:
                self.current_player_idx = shooter_idx
            else:
                print(f"{self.last_shooter_name} Died, deferred to the next surviving player with a hand")
                self.current_player_idx = self.find_next_player_with_cards(shooter_idx or 0)
        else:
            self.last_shooter_name = None
            self.current_player_idx = self.players.index(random.choice(alive_players))

        self.start_round_record()
        print(f"Start a new round from player: {self.players[self.current_player_idx].name} ")

    def check_victory(self) -> bool:
        """
        Check the victory condition (when there is only one surviving player left) and record the victor
        
        Returns:
            bool: Returns True if the game is over
        """
        alive_players = [p for p in self.players if p.alive]
        if len(alive_players) == 1:
            last_survivor = alive_players[0]
            print(f"\n{last_survivor.name} is the last survivor！")
            
            # Award +3 points to the last survivor
            self.scores[last_survivor.name] += 3
            print(f"{last_survivor.name} gets +3 points for being the last survivor")
            
            # Update the scores one final time for this round
            self.game_record.update_scores(self.scores)
            
            # Determine the winner based on total points
            winner_name = max(self.scores.items(), key=lambda x: x[1])[0]
            winner_score = self.scores[winner_name]
            
            print("\nFinal Scores:")
            for name, score in self.scores.items():
                print(f"{name}: {score}")
                
            print(f"\n{winner_name} is the winner with {winner_score} points！")
            
            # Record winners and save game records
            self.game_record.finish_game(winner_name, self.scores)
            self.game_over = True
            return True
        return False
    
    def check_other_players_no_cards(self, current_player: Player) -> bool:
        """
        Check if all other surviving players have no cards in their hands
        """
        others = [p for p in self.players if p != current_player and p.alive]
        return all(not p.hand for p in others)

    def handle_play_cards(self, current_player: Player, next_player: Player) -> List[str]:
        """
        Handling of Player Playing Sessions
        
        Args:
            current_player: current player
            next_player: next player
            
        Returns:
            List[str]: return the cards played by the player
        """
        # Get basic information about the current round
        round_base_info = self.game_record.get_latest_round_info()
        round_action_info = self.game_record.get_latest_round_actions(current_player.name, include_latest=True)
        
        # Get the decision information of the current player
        play_decision_info = self.game_record.get_play_decision_info(
            current_player.name,
            next_player.name
        )
        
        # Add scoring information to help influence decisions
        scoring_info = f"Current scores: {self.scores}"
        play_decision_info += f"\n{scoring_info}"

        # Let the current player decide which cards to play
        play_result, reasoning = current_player.choose_cards_to_play(
            round_base_info,
            round_action_info,
            play_decision_info
        )

        # Record the play result
        self.game_record.record_play(
            player_name=current_player.name,
            played_cards=play_result["played_cards"].copy(),
            remaining_cards=current_player.hand.copy(),
            play_reason=play_result["play_reason"],
            behavior=play_result["behavior"],
            next_player=next_player.name,
            play_thinking=reasoning
        )

        return play_result["played_cards"]
    
    def handle_challenge(self, current_player: Player, next_player: Player, played_cards: List[str]) -> tuple:
        """
        Handle the challenge session
        
        Args:
            current_player: current player（who got challenged）
            next_player: next player (challenger）
            played_cards: Cards played by the challenged player
            
        Returns:
            tuple: (player_to_penalize, was_unsuccessful_challenge) or (None, False) if no challenge
        """
        # Get basic information about the current round
        round_base_info = self.game_record.get_latest_round_info()
        round_action_info = self.game_record.get_latest_round_actions(next_player.name, include_latest=False)
        
        # Get the decision information of the current player
        challenge_decision_info = self.game_record.get_challenge_decision_info(
            next_player.name,
            current_player.name
        )
        
        # Add scoring information to help influence decisions
        scoring_info = f"Current scores: {self.scores}"
        challenge_decision_info += f"\n{scoring_info}"

        # Get the behavior of the challenged player
        challenging_player_behavior = self.game_record.get_latest_play_behavior()

        # Check if we need to provide additional hints
        extra_hint = "Attention, All other players have emptied thier hands." if self.check_other_players_no_cards(next_player) else ""

        # Let the challenger decide whether to challenge
        challenge_result, reasoning = next_player.decide_challenge(
            round_base_info,
            round_action_info,
            challenge_decision_info,
            challenging_player_behavior,
            extra_hint
        )

        # if the challenger decides to challenge
        if challenge_result["was_challenged"]:
            # check if the played cards are valid
            is_valid = self.is_valid_play(played_cards)
            
            # Record the challenge result
            self.game_record.record_challenge(
                was_challenged=True,
                reason=challenge_result["challenge_reason"],
                result=not is_valid,  # Successful challenge means the cards were played illegally
                challenge_thinking=reasoning
            )
            
            # Update scores for successful challenges
            if not is_valid:
                # Successful challenge: +2 point
                self.scores[next_player.name] += 2
                print(f"{next_player.name} gets +2 point for successful challenge")
            
            # Returns the player to be penalized based on the validation results
            return (next_player, is_valid) if is_valid else (current_player, False)
        else:
            # Recording of unchallenged cases
            self.game_record.record_challenge(
                was_challenged=False,
                reason=challenge_result["challenge_reason"],
                result=None,
                challenge_thinking=reasoning
            )
            
            # NEW SCORING: Check if player correctly didn't challenge an honest play
            # Check if the played cards were actually valid (all target cards)
            is_valid = self.is_valid_play(played_cards)
            if is_valid:
                # Player correctly chose not to challenge an honest play: +2 points
                self.scores[next_player.name] += 2
                print(f"{next_player.name} gets +2 points for correctly not challenging an honest play")
            
            # NEW SCORING: Check if current player emptied hand without being challenged
            if not current_player.hand:  # If player has no cards left
                self.scores[current_player.name] += 2
                print(f"{current_player.name} gets +2 points for successfully emptying hand without being challenged")
            
            return None, False

    def handle_system_challenge(self, current_player: Player) -> None:
        """
        Handling of automatic system challenges
        When all other surviving players have no hand, the system automatically challenges the current player.
        
        Args:
            current_player: Current player (last player with a hand)
        """
        print(f"System will automatically challenge player {current_player.name} 's hand！")
        
        # Record the player's automatic card play
        all_cards = current_player.hand.copy()  # Copy the player's hand
        current_player.hand.clear()  # Clear the player's hand
        
        # Record the play 
        self.game_record.record_play(
            player_name=current_player.name,
            played_cards=all_cards,
            remaining_cards=[],  # remaining_cards is an empty list
            play_reason="You are the last player, cards will be automatically played.",
            behavior="None",
            next_player="No one",
            play_thinking=""
        )
        
        # Check if the played cards are valid
        is_valid = self.is_valid_play(all_cards)
        
        # Record the system challenge result
        self.game_record.record_challenge(
            was_challenged=True,
            reason="System automatic challenge",
            result=not is_valid,  # Successful challenge means the cards were played illegally
            challenge_thinking=""
        )
        
        if is_valid:
            print(f"System challenge unsuccessful！{current_player.name} 's hand is legal.")
            # Recording a special shooting result (no one shot)
            self.game_record.record_shooting(
                shooter_name="none",
                bullet_hit=False
            )
            self.reset_round(record_shooter=False)
        else:
            print(f"The system was successfully challenged! {current_player.name}'s hand is in violation and a shooting penalty will be enforced.")
            self.perform_penalty(current_player)

    def handle_reflection(self) -> List[Player]:
        """
        Handles the reflection process for all surviving players
        Invoked at the end of each round to allow players to reflect on and evaluate the behavior of other players
        
        Returns:
            List[Player]: List of surviving players
        """
        # Get a list of surviving players
        alive_players = [p for p in self.players if p.alive]
        alive_player_names = [p.name for p in alive_players]
        
        # Get basic information about the current round
        round_base_info = self.game_record.get_latest_round_info()
        
        # Let each surviving player reflect on the behavior of other players
        for player in alive_players:
            # Get information about actions taken in the current player's turn
            round_action_info = self.game_record.get_latest_round_actions(player.name, include_latest=True)
            # Get the results of the current round
            round_result = self.game_record.get_latest_round_result(player.name)
            
            # Add scoring information to the reflection
            scoring_info = f"Current scores: {self.scores}"
            
            # Let the player reflect on the behavior of other players
            player.reflect(
                alive_players=alive_player_names,
                round_base_info=round_base_info,
                round_action_info=round_action_info + "\n" + scoring_info,
                round_result=round_result
            )

        return alive_players

    def play_round(self) -> None:
        """Play a round of the game"""
        current_player = self.players[self.current_player_idx]

        # When all other surviving players have no cards in hand, the system automatically questions the current player
        if self.check_other_players_no_cards(current_player):
            self.handle_system_challenge(current_player)
            return

        print(f"\nIt's {current_player.name}'s turn, target card is {self.target_card}")
        current_player.print_status()
        
        # Display current scores before each player's turn
        print("\nCurrent Scores:")
        for name, score in self.scores.items():
            print(f"{name}: {score}")
        print()

        # Find the next player with a hand
        next_idx = self.find_next_player_with_cards(self.current_player_idx)
        next_player = self.players[next_idx]

        # Handle the play session
        played_cards = self.handle_play_cards(current_player, next_player)

        # Handle the challenge session
        if next_player != current_player:
            result = self.handle_challenge(current_player, next_player, played_cards)
            player_to_penalize, was_unsuccessful_challenge = result
            
            if player_to_penalize is not None:
                self.perform_penalty(player_to_penalize, was_unsuccessful_challenge)
                return
            else:
                print(f"{next_player.name} did not challenge {current_player.name}'s hand. Game continue...")
                
        # Update the current player index
        self.current_player_idx = next_idx

    def start_game(self) -> None:
        """Start the game and play until the game is over"""
        print("\nInitial Scores:")
        for name, score in self.scores.items():
            print(f"{name}: {score}")
        print()
        
        self.choose_target_card()
        self.deal_cards()
        self.start_round_record()
        while not self.game_over:
            self.play_round()

if __name__ == '__main__':
    # Configure the player information, where model is the name of the model you are calling through the API.
    player_configs = [
        {"name": "Lily", "model": "llama-3.1-8b"},
        {"name": "Luke", "model": "llama-3.2-3b"},
        {"name": "Mike", "model": "ministral-8b"},
        {"name": "Quinn", "model": "qwen2.5-7b"}
    ]
    print("Game Start! The player configuration is as follows:")
    for config in player_configs:
        print(f"Player：{config['name']}, Model used：{config['model']}")
    print("-" * 50)


    # Create a game instance and start the game
    game = Game(player_configs)
    game.start_game()
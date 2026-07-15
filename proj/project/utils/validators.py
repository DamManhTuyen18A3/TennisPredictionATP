"""
Input Validation Utilities

Provides validation functions for player names, dates, tournament data,
and other user inputs to ensure data integrity and quality.
"""

from typing import Tuple, Optional, List
import re
from datetime import datetime


class InputValidator:
    """General input validation utilities."""
    
    @staticmethod
    def is_valid_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
        """
        Validate date string format.
        
        Args:
            date_str: Date string to validate
            format: Expected date format
        
        Returns:
            True if valid date
            
        Examples:
            is_valid_date("2025-06-15") -> True
            is_valid_date("06/15/2025") -> False
        """
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_future_date(date_str: str) -> bool:
        """
        Check if date is in the future.
        
        Args:
            date_str: Date string (ISO format)
        
        Returns:
            True if date is in future
        """
        try:
            date_obj = datetime.fromisoformat(date_str)
            return date_obj > datetime.now()
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email string
        
        Returns:
            True if valid email
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def is_valid_float(value: str) -> bool:
        """
        Check if string can be converted to float.
        
        Args:
            value: String to validate
        
        Returns:
            True if valid float
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_integer(value: str) -> bool:
        """
        Check if string can be converted to integer.
        
        Args:
            value: String to validate
        
        Returns:
            True if valid integer
        """
        try:
            int(value)
            return True
        except ValueError:
            return False


class PlayerValidator:
    """Validation for player-related inputs."""
    
    @staticmethod
    def validate_player_name(name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate player name.
        
        Args:
            name: Player name string
        
        Returns:
            (is_valid, error_message)
            
        Examples:
            validate_player_name("Novak Djokovic") -> (True, None)
            validate_player_name("") -> (False, "Player name cannot be empty")
        """
        if not name or not isinstance(name, str):
            return False, "Player name cannot be empty"
        
        if len(name.strip()) < 2:
            return False, "Player name too short (minimum 2 characters)"
        
        if len(name) > 100:
            return False, "Player name too long (maximum 100 characters)"
        
        # Check for invalid characters
        if not re.match(r"^[a-zA-Z\s\-'\.]+$", name):
            return False, "Player name contains invalid characters"
        
        return True, None
    
    @staticmethod
    def validate_two_different_players(player_a: str, player_b: str) -> Tuple[bool, Optional[str]]:
        """
        Ensure two players are different.
        
        Args:
            player_a: First player name
            player_b: Second player name
        
        Returns:
            (is_valid, error_message)
        """
        if player_a.lower().strip() == player_b.lower().strip():
            return False, "Player A and Player B must be different"
        
        return True, None
    
    @staticmethod
    def validate_ranking(ranking: int) -> Tuple[bool, Optional[str]]:
        """
        Validate player ranking number.
        
        Args:
            ranking: Ranking number
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(ranking, int):
            return False, "Ranking must be an integer"
        
        if ranking < 1:
            return False, "Ranking must be positive"
        
        if ranking > 10000:
            return False, "Ranking seems invalid (> 10000)"
        
        return True, None
    
    @staticmethod
    def validate_hand(hand: str) -> Tuple[bool, Optional[str]]:
        """
        Validate player's playing hand.
        
        Args:
            hand: "L" (left) or "R" (right)
        
        Returns:
            (is_valid, error_message)
        """
        valid_hands = ["L", "R", "left", "right"]
        
        if hand.upper() not in valid_hands:
            return False, "Hand must be 'L' (left) or 'R' (right)"
        
        return True, None


class TournamentValidator:
    """Validation for tournament-related inputs."""
    
    VALID_SURFACES = ["Hard", "Clay", "Grass", "Carpet", "Indoor Hard", "Outdoor Hard"]
    VALID_ROUNDS = ["First Round", "Qualifying", "Round 1", "Round 2", "Round 3", 
                   "Round 4", "Quarterfinals", "Semifinals", "Finals"]
    VALID_BEST_OF = [1, 3, 5]
    
    @staticmethod
    def validate_surface(surface: str) -> Tuple[bool, Optional[str]]:
        """
        Validate match surface.
        
        Args:
            surface: Surface name
        
        Returns:
            (is_valid, error_message)
        """
        if surface not in TournamentValidator.VALID_SURFACES:
            valid = ", ".join(TournamentValidator.VALID_SURFACES)
            return False, f"Invalid surface. Valid: {valid}"
        
        return True, None
    
    @staticmethod
    def validate_round(round_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate match round.
        
        Args:
            round_name: Round name
        
        Returns:
            (is_valid, error_message)
        """
        if round_name not in TournamentValidator.VALID_ROUNDS:
            valid = ", ".join(TournamentValidator.VALID_ROUNDS)
            return False, f"Invalid round. Valid: {valid}"
        
        return True, None
    
    @staticmethod
    def validate_best_of(best_of: int) -> Tuple[bool, Optional[str]]:
        """
        Validate best-of format.
        
        Args:
            best_of: Best of format (1, 3, or 5)
        
        Returns:
            (is_valid, error_message)
        """
        if best_of not in TournamentValidator.VALID_BEST_OF:
            return False, "Best of must be 1, 3, or 5"
        
        return True, None
    
    @staticmethod
    def validate_tournament_name(name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate tournament name.
        
        Args:
            name: Tournament name
        
        Returns:
            (is_valid, error_message)
        """
        if not name or len(name.strip()) < 2:
            return False, "Tournament name too short"
        
        if len(name) > 200:
            return False, "Tournament name too long"
        
        return True, None


class ProbabilityValidator:
    """Validation for probability and odds values."""
    
    @staticmethod
    def validate_probability(prob: float) -> Tuple[bool, Optional[str]]:
        """
        Validate probability value (0-1).
        
        Args:
            prob: Probability value
        
        Returns:
            (is_valid, error_message)
        """
        try:
            prob = float(prob)
        except (ValueError, TypeError):
            return False, "Probability must be a number"
        
        if not (0 <= prob <= 1):
            return False, "Probability must be between 0 and 1"
        
        return True, None
    
    @staticmethod
    def validate_odds(odds: float) -> Tuple[bool, Optional[str]]:
        """
        Validate decimal odds value.
        
        Args:
            odds: Decimal odds
        
        Returns:
            (is_valid, error_message)
        """
        try:
            odds = float(odds)
        except (ValueError, TypeError):
            return False, "Odds must be a number"
        
        if odds <= 1:
            return False, "Odds must be greater than 1"
        
        if odds > 1000:
            return False, "Odds value seems invalid (> 1000)"
        
        return True, None
    
    @staticmethod
    def validate_confidence(confidence: str) -> Tuple[bool, Optional[str]]:
        """
        Validate confidence level string.
        
        Args:
            confidence: Confidence level
        
        Returns:
            (is_valid, error_message)
        """
        valid_levels = ["HIGH", "MEDIUM", "LOW"]
        
        if confidence.upper() not in valid_levels:
            return False, f"Confidence must be HIGH, MEDIUM, or LOW"
        
        return True, None


class PredictionInputValidator:
    """Comprehensive validator for prediction form inputs."""
    
    @staticmethod
    def validate_prediction_form(player_a: str, player_b: str, 
                                tournament: str, surface: str, 
                                round_name: str, date_str: str,
                                best_of: int) -> Tuple[bool, Optional[str]]:
        """
        Validate complete prediction form.
        
        Args:
            player_a: First player name
            player_b: Second player name
            tournament: Tournament name
            surface: Match surface
            round_name: Match round
            date_str: Match date (ISO format)
            best_of: Best of format
        
        Returns:
            (is_valid, error_message)
        """
        # Validate players
        valid, msg = PlayerValidator.validate_player_name(player_a)
        if not valid:
            return False, f"Player A: {msg}"
        
        valid, msg = PlayerValidator.validate_player_name(player_b)
        if not valid:
            return False, f"Player B: {msg}"
        
        # Validate different players
        valid, msg = PlayerValidator.validate_two_different_players(player_a, player_b)
        if not valid:
            return False, msg
        
        # Validate tournament
        valid, msg = TournamentValidator.validate_tournament_name(tournament)
        if not valid:
            return False, f"Tournament: {msg}"
        
        # Validate surface
        valid, msg = TournamentValidator.validate_surface(surface)
        if not valid:
            return False, msg
        
        # Validate round
        valid, msg = TournamentValidator.validate_round(round_name)
        if not valid:
            return False, msg
        
        # Validate date
        if not InputValidator.is_valid_date(date_str):
            return False, "Invalid date format (use YYYY-MM-DD)"
        
        # Validate best of
        valid, msg = TournamentValidator.validate_best_of(best_of)
        if not valid:
            return False, msg
        
        return True, None


# Export validators
__all__ = [
    "InputValidator",
    "PlayerValidator",
    "TournamentValidator",
    "ProbabilityValidator",
    "PredictionInputValidator",
]

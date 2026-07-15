"""
Data Formatting Utilities

Provides consistent formatting for numbers, probabilities, dates, and other data types
used throughout the application. This ensures a professional, uniform appearance.
"""

from typing import Union, Optional
from datetime import datetime, timedelta
import math
import textwrap


class NumberFormatter:
    """Format numbers for professional display."""
    
    @staticmethod
    def percentage(value: float, decimals: int = 1, include_symbol: bool = True) -> str:
        """
        Format as percentage.
        
        Args:
            value: Decimal value (0-1)
            decimals: Number of decimal places
            include_symbol: Include % symbol
        
        Returns:
            Formatted percentage string
            
        Examples:
            percentage(0.743) -> "74.3%"
            percentage(0.8, 2) -> "80.00%"
        """
        pct = value * 100
        formatted = f"{pct:.{decimals}f}"
        return f"{formatted}%" if include_symbol else formatted
    
    @staticmethod
    def probability(value: float, decimals: int = 1) -> str:
        """
        Format probability (0-1).
        
        Args:
            value: Probability value
            decimals: Number of decimal places
        
        Returns:
            Formatted probability with visual indicator
            
        Examples:
            probability(0.75) -> "75.0%"
            probability(0.25) -> "25.0%"
        """
        return NumberFormatter.percentage(value, decimals)
    
    @staticmethod
    def rating(value: float, decimals: int = 1) -> str:
        """
        Format rating/score (typically 0-100).
        
        Args:
            value: Rating value
            decimals: Number of decimal places
        
        Returns:
            Formatted rating
            
        Examples:
            rating(74.3) -> "74.3"
            rating(1750) -> "1,750.0"
        """
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def integer(value: int) -> str:
        """
        Format integer with thousand separators.
        
        Args:
            value: Integer to format
        
        Returns:
            Formatted integer
            
        Examples:
            integer(42350) -> "42,350"
        """
        return f"{value:,}"
    
    @staticmethod
    def float_value(value: float, decimals: int = 2) -> str:
        """
        Format float with thousand separators.
        
        Args:
            value: Float to format
            decimals: Number of decimal places
        
        Returns:
            Formatted float
        """
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def scientific(value: float, decimals: int = 2) -> str:
        """
        Format in scientific notation.
        
        Args:
            value: Value to format
            decimals: Number of decimal places
        
        Returns:
            Scientific notation string
            
        Examples:
            scientific(0.0000543) -> "5.43e-05"
        """
        return f"{value:.{decimals}e}"
    
    @staticmethod
    def change(value: float, decimals: int = 2, include_sign: bool = True) -> str:
        """
        Format change/delta value with +/- indicator.
        
        Args:
            value: Change value
            decimals: Number of decimal places
            include_sign: Include + for positive values
        
        Returns:
            Formatted change string
            
        Examples:
            change(0.8) -> "+0.80%"
            change(-1.2) -> "-1.20%"
        """
        formatted = f"{abs(value):.{decimals}f}"
        sign = "+" if value >= 0 else "-"
        return f"{sign}{formatted}" if include_sign else formatted


class ProbabilityFormatter:
    """Format probabilities with contextual information."""
    
    @staticmethod
    def confidence_level(probability: float) -> str:
        """
        Get confidence level label.
        
        Args:
            probability: Probability value (0-1)
        
        Returns:
            Confidence level label
            
        Examples:
            confidence_level(0.85) -> "HIGH"
            confidence_level(0.55) -> "MEDIUM"
            confidence_level(0.51) -> "LOW"
        """
        if probability >= 0.70:
            return "HIGH"
        elif probability >= 0.55:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def confidence_emoji(probability: float) -> str:
        """
        Get emoji for confidence level.
        
        Args:
            probability: Probability value (0-1)
        
        Returns:
            Emoji representation
        """
        if probability >= 0.70:
            return "🟢"
        elif probability >= 0.55:
            return "🟡"
        else:
            return "🔴"
    
    @staticmethod
    def odds_to_probability(odds: float, format_type: str = "decimal") -> float:
        """
        Convert betting odds to probability.
        
        Args:
            odds: Odds value
            format_type: "decimal", "fraction", or "moneyline"
        
        Returns:
            Probability (0-1)
            
        Examples:
            odds_to_probability(2.0, "decimal") -> 0.5
            odds_to_probability(100, "moneyline") -> 0.5
        """
        if format_type == "decimal":
            return 1 / odds
        elif format_type == "moneyline":
            if odds > 0:
                return 100 / (100 + odds)
            else:
                return -odds / (-odds + 100)
        return 0.5


class DateFormatter:
    """Format dates and times."""
    
    @staticmethod
    def date_short(date_obj: Union[str, datetime]) -> str:
        """
        Format date in short format.
        
        Args:
            date_obj: Date object or ISO string
        
        Returns:
            Short formatted date
            
        Examples:
            date_short("2025-06-15") -> "Jun 15"
        """
        if isinstance(date_obj, str):
            date_obj = datetime.fromisoformat(date_obj)
        return date_obj.strftime("%b %d")
    
    @staticmethod
    def date_long(date_obj: Union[str, datetime]) -> str:
        """
        Format date in long format.
        
        Args:
            date_obj: Date object or ISO string
        
        Returns:
            Long formatted date
            
        Examples:
            date_long("2025-06-15") -> "June 15, 2025"
        """
        if isinstance(date_obj, str):
            date_obj = datetime.fromisoformat(date_obj)
        return date_obj.strftime("%B %d, %Y")
    
    @staticmethod
    def time_ago(date_obj: Union[str, datetime]) -> str:
        """
        Format date as time ago.
        
        Args:
            date_obj: Date object or ISO string
        
        Returns:
            Relative time string
            
        Examples:
            time_ago(datetime.now() - timedelta(days=2)) -> "2 days ago"
        """
        if isinstance(date_obj, str):
            date_obj = datetime.fromisoformat(date_obj)
        
        delta = datetime.now() - date_obj
        
        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            mins = int(delta.total_seconds() / 60)
            return f"{mins}m ago"
        elif delta.total_seconds() < 86400:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        elif delta.days < 7:
            return f"{delta.days}d ago"
        elif delta.days < 30:
            weeks = delta.days // 7
            return f"{weeks}w ago"
        else:
            months = delta.days // 30
            return f"{months}mo ago"
    
    @staticmethod
    def iso_date(date_obj: Union[str, datetime]) -> str:
        """
        Format as ISO 8601 date.
        
        Args:
            date_obj: Date object or ISO string
        
        Returns:
            ISO format string
        """
        if isinstance(date_obj, str):
            date_obj = datetime.fromisoformat(date_obj)
        return date_obj.isoformat()


class StatFormatter:
    """Format various sports statistics."""
    
    @staticmethod
    def elo_rating(rating: float) -> str:
        """
        Format Elo rating.
        
        Args:
            rating: Elo rating value
        
        Returns:
            Formatted Elo rating
        """
        return f"{int(rating)}"
    
    @staticmethod
    def ranking(rank: int) -> str:
        """
        Format ranking number with ordinal suffix.
        
        Args:
            rank: Ranking number
        
        Returns:
            Formatted ranking
            
        Examples:
            ranking(1) -> "1st"
            ranking(21) -> "21st"
            ranking(22) -> "22nd"
        """
        if 10 <= rank % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
        return f"{rank}{suffix}"
    
    @staticmethod
    def win_loss(wins: int, losses: int) -> str:
        """
        Format win-loss record.
        
        Args:
            wins: Number of wins
            losses: Number of losses
        
        Returns:
            Formatted W-L record
            
        Examples:
            win_loss(45, 12) -> "45-12"
        """
        return f"{wins}-{losses}"
    
    @staticmethod
    def win_rate(wins: int, total: int, decimals: int = 1) -> str:
        """
        Calculate and format win rate.
        
        Args:
            wins: Number of wins
            total: Total matches
            decimals: Number of decimal places
        
        Returns:
            Formatted win rate percentage
        """
        if total == 0:
            return "0.0%"
        return NumberFormatter.percentage(wins / total, decimals)
    
    @staticmethod
    def serve_speed(speed_kmh: float) -> str:
        """
        Format serve speed.
        
        Args:
            speed_kmh: Speed in km/h
        
        Returns:
            Formatted speed
        """
        return f"{int(speed_kmh)} km/h"


class PercentageBar:
    """Generate ASCII/Unicode percentage bars."""
    
    @staticmethod
    def unicode_bar(percentage: float, width: int = 20, 
                   char_full: str = "█", char_empty: str = "░") -> str:
        """
        Generate Unicode percentage bar.
        
        Args:
            percentage: Value between 0-1
            width: Bar width in characters
            char_full: Character for filled portion
            char_empty: Character for empty portion
        
        Returns:
            Bar string
            
        Examples:
            unicode_bar(0.75, 20) -> "███████████████░░░░"
        """
        filled = int(width * min(1, max(0, percentage)))
        return char_full * filled + char_empty * (width - filled)
    
    @staticmethod
    def html_bar(percentage: float, height: int = 8, 
                color: str = "#22C55E") -> str:
        """
        Generate HTML progress bar.
        
        Args:
            percentage: Value between 0-1
            height: Bar height in pixels
            color: Bar color (hex)
        
        Returns:
            HTML string
        """
        pct = max(0, min(100, percentage * 100))
        # dedent: tránh Markdown hiểu nhầm HTML thụt lề thành code block
        return textwrap.dedent(f"""
        <div style="width:100%;height:{height}px;background:#e5e7eb;border-radius:4px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};transition:width 300ms ease;"/>
        </div>
        """)


# Export all formatters
__all__ = [
    "NumberFormatter",
    "ProbabilityFormatter",
    "DateFormatter",
    "StatFormatter",
    "PercentageBar",
]

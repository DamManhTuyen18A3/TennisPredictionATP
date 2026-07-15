"""
Gauge Components

Visualization components for displaying confidence levels, probabilities, and ratings
using circular gauge indicators and progress bars.
"""

from typing import Optional
import streamlit as st
import plotly.graph_objects as go
from project.utils.theme import ColorPalette
from project.utils.formatters import ProbabilityFormatter


def confidence_gauge(value: float, title: str = "Confidence", 
                     show_level_text: bool = True) -> None:
    """
    Display a circular confidence gauge.
    
    Args:
        value: Confidence value (0-1)
        title: Gauge title
        show_level_text: Show "HIGH/MEDIUM/LOW" text
    
    Example:
        confidence_gauge(0.75, "Model Confidence")
    """
    colors = ColorPalette()
    
    if value <= 0.33:
        gauge_color = colors.ERROR
        level = "LOW"
    elif value <= 0.66:
        gauge_color = colors.WARNING
        level = "MEDIUM"
    else:
        gauge_color = colors.SUCCESS
        level = "HIGH"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={'text': title, 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': colors.NEUTRAL_DARK},
            'bar': {'color': gauge_color},
            'bgcolor': colors.NEUTRAL_DARKER,
            'borderwidth': 2,
            'bordercolor': gauge_color,
            'steps': [
                {'range': [0, 33], 'color': colors.ERROR + '20'},
                {'range': [33, 66], 'color': colors.WARNING + '20'},
                {'range': [66, 100], 'color': colors.SUCCESS + '20'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 90,
            }
        },
        number={'suffix': "%", 'font': {'size': 32}},
        delta={'reference': 50, 'suffix': "", 'font': {'size': 14}},
    ))
    
    fig.update_layout(
        font={'family': 'Inter, Arial', 'color': colors.TEXT_PRIMARY, 'size': 14},
        plot_bgcolor=colors.PRIMARY,
        paper_bgcolor=colors.PRIMARY,
        margin=dict(l=20, r=20, t=60, b=20),
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False,
        'responsive': True,
    })
    
    if show_level_text:
        col1, col2, col3 = st.columns(3)
        with col2:
            emoji = ProbabilityFormatter.confidence_emoji(value)
            st.markdown(f"<h3 style='text-align: center;'>{emoji} {level} Confidence</h3>", 
                       unsafe_allow_html=True)


def win_probability_bar(probability: float, player_a: str, player_b: str,
                        show_values: bool = True) -> None:
    """
    Display a horizontal win probability bar.
    
    Args:
        probability: Win probability for Player A (0-1)
        player_a: Player A name
        player_b: Player B name
        show_values: Show percentage values
    
    Example:
        win_probability_bar(0.75, "Alcaraz", "Sinner")
    """
    colors = ColorPalette()
    prob_b = 1 - probability
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            y=['Win Probability'],
            x=[probability * 100],
            name=player_a,
            marker=dict(color=colors.SUCCESS),
            textposition='inside',
            text=[f'{probability*100:.1f}%'] if show_values else [],
            hovertext=f"{player_a}: {probability*100:.1f}%",
            hoverinfo='text',
            orientation='h',
        ),
        go.Bar(
            y=['Win Probability'],
            x=[prob_b * 100],
            name=player_b,
            marker=dict(color=colors.ERROR),
            textposition='inside',
            text=[f'{prob_b*100:.1f}%'] if show_values else [],
            hovertext=f"{player_b}: {prob_b*100:.1f}%",
            hoverinfo='text',
            orientation='h',
        ),
    ])
    
    fig.update_layout(
        barmode='relative',
        font={'family': 'Inter, Arial', 'color': colors.TEXT_PRIMARY, 'size': 12},
        plot_bgcolor=colors.PRIMARY,
        paper_bgcolor=colors.PRIMARY,
        xaxis={'showgrid': False, 'zeroline': False},
        yaxis={'showticklabels': True},
        margin=dict(l=10, r=10, t=30, b=10),
        height=100,
        showlegend=True,
        legend=dict(orientation='h', y=1.15, x=0),
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False,
        'responsive': True,
    })


def rating_radial(ratings: dict, title: str = "Player Ratings") -> None:
    """
    Display ratings in a radial/spider chart.
    
    Args:
        ratings: Dictionary of {category: value (0-100)}
        title: Chart title
    
    Example:
        rating_radial({
            'Serve': 85,
            'Return': 78,
            'Forehand': 92,
        })
    """
    colors = ColorPalette()
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=list(ratings.values()),
            theta=list(ratings.keys()),
            fill='toself',
            name='Rating',
            line_color=colors.SECONDARY,
            fillcolor=colors.SECONDARY + '30',
        ),
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                linewidth=1,
                gridcolor=colors.NEUTRAL_DARK,
                gridwidth=1,
            ),
            angularaxis=dict(
                showline=True,
                linewidth=1,
                gridcolor=colors.NEUTRAL_DARK,
            ),
            bgcolor=colors.PRIMARY,
        ),
        font={'family': 'Inter, Arial', 'color': colors.TEXT_PRIMARY, 'size': 12},
        plot_bgcolor=colors.PRIMARY,
        paper_bgcolor=colors.PRIMARY,
        title={'text': title, 'font': {'size': 16}},
        showlegend=False,
        margin=dict(l=80, r=80, t=80, b=80),
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False,
        'responsive': True,
    })


def score_indicator(score: float, max_score: float = 100,
                   title: str = "Score", threshold: float = 70) -> None:
    """
    Display a simple score indicator with color coding.
    
    Args:
        score: Current score
        max_score: Maximum possible score
        title: Indicator title
        threshold: Threshold for "good" score
    """
    colors = ColorPalette()
    
    if score < threshold * 0.5:
        color = colors.ERROR
    elif score < threshold:
        color = colors.WARNING
    else:
        color = colors.SUCCESS
    
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    
    fig = go.Figure(data=[
        go.Indicator(
            mode="number+gauge",
            value=score,
            title={'text': title},
            gauge={
                'axis': {'range': [0, max_score]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, threshold * 0.5], 'color': colors.ERROR + '20'},
                    {'range': [threshold * 0.5, threshold], 'color': colors.WARNING + '20'},
                    {'range': [threshold, max_score], 'color': colors.SUCCESS + '20'},
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': threshold,
                }
            }
        ),
    ])
    
    fig.update_layout(
        font={'family': 'Inter, Arial', 'color': colors.TEXT_PRIMARY, 'size': 14},
        plot_bgcolor=colors.PRIMARY,
        paper_bgcolor=colors.PRIMARY,
        margin=dict(l=20, r=20, t=60, b=20),
        height=250,
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False,
        'responsive': True,
    })


__all__ = [
    "confidence_gauge",
    "win_probability_bar",
    "rating_radial",
    "score_indicator",
]

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def roberta_barchat(text, name):
    # Build out later
    pass

def vaders_barchart(text, name):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    vaders = pd.DataFrame(sentiment_scores, index=[0])
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=vaders.index,
        y=vaders['pos'],
        name='Positive',
        marker=dict(color='green'),
        text=vaders['pos'],
        textposition='auto',
        hoverinfo='none'
    ))

    fig.add_trace(go.Bar(
        x=vaders.index,
        y=vaders['neu'],
        name='Neutral',
        marker=dict(color='orange'),
        text=vaders['neu'],
        textposition='auto',
        hoverinfo='none'
    ))

    fig.add_trace(go.Bar(
        x=vaders.index,
        y=vaders['neg'],
        name='Negative',
        marker=dict(color='red'),
        text=vaders['neu'],
        textposition='auto',
        hoverinfo='none'
    ))

    fig.update_layout(
        title='Vaders Score Distribution for {name}',
        xaxis_title='Index',
        yaxis_title='Proportion',
        barmode='group',
        showlegend=True,
        legend=dict(x=1, y=1),
        hovermode='x',
        height=380,
        width=675,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=12, color='black')
    )

    st.plotly_chart(fig)
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 

def plot_relevant_docs(projected_dataset_embeddings,
                       projected_query_embedding,
                       projected_retrieved_embeddings,
                       query, title, projected_augmented_embeddings=[]):
    """
    Plots the query, dataset, and retrieved documents in 2D embedding space using Plotly.
    Parameters:
    projected_dataset_embeddings (numpy.ndarray): 2D array of projected dataset embeddings.
    projected_query_embedding (numpy.ndarray): 2D array containing the projected query embedding.
    projected_retrieved_embeddings (numpy.ndarray): 2D array of projected embeddings for retrieved documents.
    query (str): The query text to be displayed on hover.
    title (str): Title of the plot.
    """
    # Convert data to DataFrame for easier plotting
    df_dataset = pd.DataFrame(projected_dataset_embeddings, columns=['x', 'y'])
    df_query = pd.DataFrame(projected_query_embedding, columns=['x', 'y'])
    df_query['text'] = query  # Add query text for hover information
    df_retrieved = pd.DataFrame(projected_retrieved_embeddings, columns=['x', 'y'])
    df_augmented = pd.DataFrame(projected_augmented_embeddings, columns=['x', 'y'])
    # Create a scatter plot for the dataset points
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_dataset['x'],
        y=df_dataset['y'],
        mode='markers',
        marker=dict(size=10, color='gray'),
        name='Dataset Embeddings'
    ))
    fig.add_trace(go.Scatter(
        x=df_query['x'],
        y=df_query['y'],
        mode='markers',
        marker=dict(size=15, symbol='x', color='red'),
        name='Query Embedding',
        hovertext=[query],  # Add hover text
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=df_retrieved['x'],
        y=df_retrieved['y'],
        mode='markers',
        marker=dict(size=12, symbol='circle-open', line=dict(color='green', width=2)),
        name='Retrieved Embeddings'
    ))

    fig.add_trace(go.Scatter(
        x=df_augmented['x'],
        y=df_augmented['y'],
        mode='markers',
        marker=dict(size=15, symbol='x', color='pink'),
        name='Augmented Embeddings'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        title_x=0.5
    )
    curr_dir=os.getcwd()
    fig.write_html(os.path.join(curr_dir,"doc_retreival.html"))

def plot_embeddings(projected_dataset_embeddings):
    """
    Plots 2D projected embeddings using Plotly Express.
    Parameters:
    projected_dataset_embeddings (numpy.ndarray): A 2D array containing the projected embeddings,
                                                  where each row represents a point in 2D space.
    """
    df_embeddings = pd.DataFrame(projected_dataset_embeddings, columns=['x', 'y'])
    fig = px.scatter(df_embeddings, x='x', y='y', title='Projected Embeddings')
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        title_x=0.5
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # Maintain aspect ratio

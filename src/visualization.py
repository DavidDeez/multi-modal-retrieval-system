import umap
import plotly.graph_objects as go
import numpy as np

class EmbeddingVisualizer:
    def __init__(self):
        self.reducer = None
        self.embeddings_2d = None

    def fit_transform(self, embeddings, n_neighbors=15):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', random_state=42)
        self.embeddings_2d = self.reducer.fit_transform(embeddings)
        return self.embeddings_2d

    def transform(self, query_embedding):
        return self.reducer.transform(query_embedding)

    def create_plot(self, embeddings_2d, query_point=None, top_results=None):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode='markers',
            marker=dict(size=4, color='lightblue', opacity=0.6),
            name='All Images',
            hovertemplate='Image Index: %{text}<extra></extra>',
            text=list(range(len(embeddings_2d)))
        ))

        if top_results:
            result_indices = [r['metadata']['index'] for r in top_results]
            result_coords = embeddings_2d[result_indices]
            fig.add_trace(go.Scatter(
                x=result_coords[:, 0],
                y=result_coords[:, 1],
                mode='markers',
                marker=dict(size=10, color='red', symbol='star'),
                name='Top Results',
                hovertemplate='Rank: %{text}<extra></extra>',
                text=[f"Rank {r['rank']}" for r in top_results]
            ))

        if query_point is not None:
            fig.add_trace(go.Scatter(
                x=[query_point[0, 0]],
                y=[query_point[0, 1]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='diamond'),
                name='Query',
                hovertemplate='Query Point<extra></extra>'
            ))

        fig.update_layout(
            title='Embedding Space Visualization (UMAP)',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            hovermode='closest',
            height=600,
            showlegend=True
        )
        return fig

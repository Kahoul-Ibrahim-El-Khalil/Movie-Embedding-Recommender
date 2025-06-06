import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import joblib
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

class MoviePredictor:
    def __init__(self, 
                 Raw_Data_Path="Data/data.csv", 
                 Sentence_Transformer_Model="all-MiniLM-L6-v2", 
                 Label_Encoding_Path="label_encoder.pkl",
                 Embeddings_Path="movie_embeddings.pkl",
                 Start_From_Raw_Data=False):
        
        self.sentence_transformer = SentenceTransformer(Sentence_Transformer_Model)
        
        if not Start_From_Raw_Data:
            try:
                self.label_encoder = joblib.load(Label_Encoding_Path)
                self.embeddings = joblib.load(Embeddings_Path)
                self.data = pd.read_csv(Raw_Data_Path)
                print("Loaded available embeddings and encodings")
            except Exception as e:
                print(f"Loading saved data failed: {e}, starting from raw data")
                Start_From_Raw_Data = True
                
        if Start_From_Raw_Data:
            self.data = pd.read_csv(Raw_Data_Path)
            self.label_encoder = LabelEncoder()
            self.embeddings = self.sentence_transformer.encode(
                self.data['Description'].tolist(), 
                show_progress_bar=True
            )
            self.data['Label'] = self.label_encoder.fit_transform(self.data['Title'])
            joblib.dump(self.label_encoder, Label_Encoding_Path)
            joblib.dump(self.embeddings, Embeddings_Path)

        self.result_data_frame = None
        self.input_vector = None

    def get_data_frame(self):
        return self.data.copy()

    def predict_movie(self, Input_Description, Top_K=3):
        input_embedding = self.sentence_transformer.encode([Input_Description])
        similarities = cosine_similarity(input_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-Top_K:][::-1]
        top_scores = similarities[top_indices]
        result = self.data.iloc[top_indices].copy()
        result["Score"] = top_scores
        self.result_data_frame = result.copy()
        self.input_vector = input_embedding
        return result

    def visualize_predictions(self):
        if self.result_data_frame is None or self.input_vector is None:
            raise ValueError("Call predict_movie() first")
        
        top_vectors = self.sentence_transformer.encode(self.result_data_frame['Description'].tolist())
        vectors = np.vstack([self.input_vector, top_vectors])
        
        pca = PCA(n_components=3)
        reduced_vectors = pca.fit_transform(vectors)
        
        titles = ['<INPUT>'] + self.result_data_frame['Title'].tolist()
        scores = [1.0] + self.result_data_frame['Score'].tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=reduced_vectors[1:, 0],
            y=reduced_vectors[1:, 1],
            z=reduced_vectors[1:, 2],
            mode='markers+text',
            marker=dict(
                size=[8 + 20 * score for score in scores[1:]],
                color=[f"rgba(30,136,229,{score})" for score in scores[1:]],
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=titles[1:],
            textposition="top center",
            hovertext=[f"{t}<br>Score: {s:.3f}" for t, s in zip(titles[1:], scores[1:])],
            hoverinfo='text',
            name="Top Matches"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[reduced_vectors[0, 0]],
            y=[reduced_vectors[0, 1]],
            z=[reduced_vectors[0, 2]],
            mode='markers+text',
            marker=dict(
                size=12,
                color='red',
                symbol='diamond'
            ),
            text=[titles[0]],
            textposition="top center",
            hoverinfo='text',
            name="Input"
        ))
        
        for i in range(1, len(reduced_vectors)):
            fig.add_trace(go.Scatter3d(
                x=[reduced_vectors[0, 0], reduced_vectors[i, 0]],
                y=[reduced_vectors[0, 1], reduced_vectors[i, 1]],
                z=[reduced_vectors[0, 2], reduced_vectors[i, 2]],
                mode='lines',
                line=dict(
                    color=f"rgba(150,150,150,{scores[i] * 0.7})",
                    width=1
                ),
                hoverinfo='none',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Movie Similarity Space",
            scene=dict(
                xaxis_title="Semantic Dimension 1",
                yaxis_title="Semantic Dimension 2",
                zaxis_title="Semantic Dimension 3",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
            ),
            width=800,
            height=700,
            showlegend=True
        )
        
        fig.show()
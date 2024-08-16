# vercel-disable-edge
import os
import logging
from flask import Flask, request, render_template_string, jsonify
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import plotly.graph_objs as go
import networkx as nx
from plotly.utils import PlotlyJSONEncoder
import json
from dotenv import load_dotenv


# set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        register_vector(conn)
        logging.debug("Database connection successful")
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        raise

# Generate embedding using pgai.openai_embed
def generate_embedding(phrase):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Call the openai_embed function
        cur.execute("SELECT pgai.openai_embed('text-embedding-ada-002', %s, %s)",
                    (phrase, os.getenv("API_KEY")))
        
        embedding = cur.fetchone()[0]
        cur.close()
        conn.close()
        logging.debug(f"Embedding generated for phrase: {phrase}")
        return embedding
    except Exception as e:
        logging.error(f"Embedding generation failed: {str(e)}")
        raise

# Calculate distances between all pairs of embeddings
def calculate_distances(embeddings):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        distances = []
        for i, (phrase1, embedding1) in enumerate(embeddings):
            for j, (phrase2, embedding2) in enumerate(embeddings):
                if i < j:  # Only calculate upper triangle
                    cur.execute("SELECT ((%s::vector) <=> (%s::vector)) AS distance", (embedding1, embedding2))
                    distance = cur.fetchone()[0]
                    distances.append((phrase1, phrase2, distance))

        cur.close()
        conn.close()
        logging.debug(f"Distances calculated for {len(embeddings)} phrases")
        return distances
    except Exception as e:
        logging.error(f"Distance calculation failed: {str(e)}")
        raise

# Create a distance matrix from the pairwise distances
def create_distance_matrix(phrases, distances):
    n = len(phrases)
    dist_matrix = np.zeros((n, n))
    for phrase1, phrase2, dist in distances:
        i, j = phrases.index(phrase1), phrases.index(phrase2)
        dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

# Simple MDS implementation
def simple_mds(dist_matrix, dims=2):
    n = dist_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(dist_matrix**2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs[:, :dims] * np.sqrt(eigvals[:dims])

# Use simple MDS to position nodes based on distances
def mds_layout(G, distances):
    phrases = list(G.nodes())
    dist_matrix = create_distance_matrix(phrases, distances)
    pos = simple_mds(dist_matrix)
    return {phrase: position for phrase, position in zip(phrases, pos)}

# Visualize distances using Plotly (Force-directed graph with distance-based positioning)
def visualize_distances(phrases, distances):
    try:
        # Create a graph
        G = nx.Graph()
        G.add_nodes_from(phrases)
        for source, target, weight in distances:
            G.add_edge(source, target, weight=weight)

        # Use MDS layout for positioning
        pos = mds_layout(G, distances)

        # Create edges
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[0]} - {edge[1]}<br>Distance: {edge[2]['weight']:.4f}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')

        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,  # Remove color scale
                colorscale='YlGnBu',
                size=10,
            ),
            text=node_text,
            textposition="top center"
        )

        # Color node points by the number of connections
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))

        node_trace.marker.color = node_adjacencies
        node_trace.marker.size = [v * 5 + 10 for v in node_adjacencies]  # Adjust size based on connections

        # Add edge labels
        edge_labels = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_labels.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    xref='x',
                    yref='y',
                    text=f"{edge[2]['weight']:.2f}",
                    showarrow=False,
                    font=dict(size=8),
                    bgcolor='#ffffff',
                    bordercolor='#888',
                    borderwidth=1,
                    borderpad=1,
                    opacity=0.8
                )
            )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Phrase Distance Network',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=edge_labels,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        logging.debug("Visualization created successfully")
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    except Exception as e:
        logging.error(f"Visualization creation failed: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        phrases_input = ""
        plot_json = None

        if request.method == 'POST':
            phrases_input = request.form['phrases']
            phrases = [phrase.strip() for phrase in phrases_input.split(',')]
            phrases = list(dict.fromkeys(phrases))  # Remove duplicates while preserving order
            
            logging.info(f"Processing request for {len(phrases)} phrases")
            
            # Generate embeddings
            embeddings = [(phrase, generate_embedding(phrase)) for phrase in phrases]
            
            # Calculate distances
            distances = calculate_distances(embeddings)
            
            # Visualize distances
            plot_json = visualize_distances(phrases, distances)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phrase Embedding Demo</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
                h1 { color: #333; }
                form { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                label { display: block; margin-top: 10px; }
                input[type="text"] { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; }
                input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; }
                input[type="submit"]:hover { background-color: #45a049; }
                #plot { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Phrase Embedding Visualization</h1>
 <p>
                When you embed phrases you are converting them into <b>vectors</b> that can be stored in a database and compared to other vectors.
            </p>
            <p>
                A comparison between vectors results in the distance between them, with smaller distances representing more closely related items.
            </p>
            <p>
                You can use the text box below to input some phrases and then visualize the vector distances between them.
            </p>
            <form method="post">
                <label for="phrases">Phrases (comma-separated):</label>
                <input type="text" id="phrases" name="phrases" value="{{ phrases_input }}" required>
                <input type="submit" value="Visualize">
            </form>
            <div id="plot"></div>
            {% if plot_json %}
            <script>
                var plotData = {{ plot_json | safe }};
                Plotly.newPlot('plot', plotData.data, plotData.layout);
            </script>
            {% endif %}
        </body>
        </html>
        """, phrases_input=phrases_input, plot_json=plot_json)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

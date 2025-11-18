import pandas as pd
from collections import defaultdict

# Load CSV
df = pd.read_csv("articles.csv")

# Function to get co-authorship counts
def get_collaboration_counts(author_id):
    collaborations = defaultdict(int)
    papers = df[df['author_id'] == author_id]

    for _, row in papers.iterrows():
        if pd.isna(row['authors']):
            continue
        coauthors = [a.strip().rstrip('/').replace('\n','') for a in row['authors'].split(',') if a.strip()]
        main_author = row['author_name'].strip()
        for coauthor in coauthors:
            if coauthor != main_author:
                collaborations[coauthor] += 1

    return collaborations

import networkx as nx
from pyvis.network import Network

def create_graph(author_name, collaborations):
    G = nx.Graph()
    G.add_node(author_name, title=author_name, color='red')  # Main author
    
    for coauthor, count in collaborations.items():
        G.add_node(coauthor, title=coauthor)
        G.add_edge(author_name, coauthor, value=count, title=f"Co-authored {count} papers")
    
    # Create interactive pyvis graph
    net = Network(height='600px', width='100%', notebook=False)
    net.from_nx(G)
    return net

import streamlit as st

st.title("Co-authorship Network")

author_id_input = st.text_input("Enter Author ID:", "")

if author_id_input:
    # Get author's name
    author_name = df[df['author_id'] == author_id_input]['author_name'].iloc[0]
    
    # Get collaborations
    collaborations = get_collaboration_counts(author_id_input)
    
    # Build graph
    net = create_graph(author_name, collaborations)
    
    # Save and display in Streamlit
    net.save_graph("coauthor_network.html")
    st.components.v1.html(open("coauthor_network.html", 'r').read(), height=650)

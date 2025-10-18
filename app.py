import os
import torch
import gradio as gr
from src.data_loader import download_coco_dataset, load_images
from src.ensemble_model import MultiModalEnsemble
from src.search_engine import FAISSSearchEngine
from src.visualization import EmbeddingVisualizer
from src.analytics import AnalyticsDashboard

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Initialize system
def initialize_system():
    # Download dataset
    dataset_available = download_coco_dataset()
    
    # Load images
    images, image_paths = load_images(max_images=1000)
    
    # Initialize models
    ensemble = MultiModalEnsemble(device=device)
    
    # Encode images and build index
    print("Encoding images...")
    image_embeddings = ensemble.encode_images(images)
    
    print("Building FAISS index...")
    search_engine = FAISSSearchEngine(dimension=512, use_gpu=False)
    metadata = [{'index': i, 'path': str(image_paths[i])} for i in range(len(images))]
    search_engine.add_vectors(image_embeddings, metadata)
    
    print("Computing UMAP projection...")
    visualizer = EmbeddingVisualizer()
    embeddings_2d = visualizer.fit_transform(image_embeddings)
    
    analytics = AnalyticsDashboard()
    
    return ensemble, search_engine, visualizer, analytics, images, embeddings_2d

# Initialize everything
ensemble, search_engine, visualizer, analytics, images, embeddings_2d = initialize_system()

# Search function
def search_system(query_type, text_query, image_query, top_k, diversity_threshold, show_viz):
    try:
        if query_type == "Text":
            if not text_query:
                return [], None, "Please enter a text query"

            query_emb = ensemble.encode_text_ensemble(text_query)
            query_emb_viz = ensemble.clip_model.get_text_features(
                **ensemble.clip_processor(text=[text_query], return_tensors="pt").to(device)
            ).cpu().numpy()
            query_emb_viz = query_emb_viz / np.linalg.norm(query_emb_viz)
        else:
            if image_query is None:
                return [], None, "Please upload an image"

            query_emb = ensemble.encode_image_query(image_query)
            query_emb_viz = query_emb

        results = search_engine.search(query_emb, k=int(top_k), diversity_threshold=diversity_threshold)

        avg_sim = np.mean([r['similarity'] for r in results])
        analytics.log_search(query_type, text_query if query_type == "Text" else "image", len(results), avg_sim)

        gallery_output = []
        for r in results:
            idx = r['metadata']['index']
            img = images[idx]
            caption = f"Rank {r['rank']} | Similarity: {r['similarity']:.4f}"
            gallery_output.append((img, caption))

        viz_plot = None
        if show_viz:
            query_2d = visualizer.transform(query_emb_viz)
            viz_plot = visualizer.create_plot(embeddings_2d, query_2d, results)

        stats = analytics.get_stats()

        return gallery_output, viz_plot, stats

    except Exception as e:
        return [], None, f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Multi-Modal Intelligence System") as demo:
    gr.Markdown("""
    # 🚀 Advanced Multi-Modal Intelligence System
    ### CLIP + Sentence Transformers | FAISS Vector Search | UMAP Visualization | Diversity Filtering
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Search Configuration")
            query_type = gr.Radio(["Text", "Image"], label="Search Mode", value="Text")
            text_input = gr.Textbox(
                label="Text Query",
                placeholder="e.g., a golden retriever playing in a park during sunset",
                lines=3,
                visible=True
            )
            image_input = gr.Image(label="Image Query", type="pil", visible=False)

            with gr.Row():
                top_k = gr.Slider(1, 50, value=10, step=1, label="Results Count")
                diversity_threshold = gr.Slider(0.5, 1.0, value=0.95, step=0.05, label="Diversity Filter")

            show_viz = gr.Checkbox(label="Show UMAP Visualization", value=True)
            search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

            gr.Markdown("### 📊 Analytics")
            stats_output = gr.Markdown("No searches yet")

        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Search Results")
            gallery_output = gr.Gallery(
                label="Retrieved Images",
                columns=4,
                height="auto",
                object_fit="contain"
            )

            gr.Markdown("### 🗺️ Embedding Space")
            viz_output = gr.Plot(label="UMAP Projection")

    def toggle_inputs(choice):
        return gr.update(visible=(choice == "Text")), gr.update(visible=(choice == "Image"))

    query_type.change(toggle_inputs, inputs=[query_type], outputs=[text_input, image_input])

    search_btn.click(
        search_system,
        inputs=[query_type, text_input, image_input, top_k, diversity_threshold, show_viz],
        outputs=[gallery_output, viz_output, stats_output]
    )

if __name__ == "__main__":
    print("Launching demo...")
    demo.launch(share=True, debug=True)

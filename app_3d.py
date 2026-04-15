import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage
from realtime_data_fetcher import MedicalDataFetcher
from main import PharmaRAGPipeline

def generate_heart_volume(grid_size=40):
    """
    Working Algorithm: Synthetic 3D Organ Reconstruction
    Generates a 3D volume based on anatomical mathematical models.
    """
    x, y, z = np.ogrid[-1.5:1.5:grid_size*1j, -1.5:1.5:grid_size*1j, -1.5:1.5:grid_size*1j]
    
    # Heart-like SDF algorithm (Taubin's Heart Surface)
    # Adjusted for voxel visualization
    volume = (x**2 + (9/4)*y**2 + z**2 - 1)**3 - x**2 * z**3 - (9/80) * y**2 * z**3
    
    # Low to High knowledge: Add noise and smoothing to simulate 'real' tissue
    noise = np.random.normal(0, 0.05, volume.shape)
    volume += noise
    
    # High Knowledge: Apply Gaussian filter to simulate organic density
    volume = ndimage.gaussian_filter(volume, sigma=1.0)
    
    return volume

def run_3d_visualization():
    st.set_page_config(page_title="PharmaRAG 3D Vision", layout="wide", page_icon="🏥")
    
    # Initialize Fetcher and Pipeline
    fetcher = MedicalDataFetcher()
    pipeline = PharmaRAGPipeline(llm_provider="google") # Improved Google MedTech Model
    
    st.title("🏥 PharmaRAG 3D Medical Intelligence")
    st.markdown("""
    ### Interactive 3D Organ Reconstruction Algorithm
    This dashboard demonstrates the latest **3D Image Base** update, ranging from low-level voxel data to high-level anatomical rendering.
    """)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Algorithm Parameters")
        grid_res = st.slider("Voxel Resolution (Low to High)", 20, 100, 40)
        threshold = st.slider("Density Threshold", -0.5, 0.5, 0.0, 0.01)
        
        st.info("""
        **How it works:**
        1. **Zero State**: Blank 3D coordinate system.
        2. **Low Knowledge**: Basic scalar field generation using Heart SDF.
        3. **Med Knowledge**: Noise injection & Gaussian smoothing for tissue realism.
        4. **High Knowledge**: Isosurface extraction and  mesh rendering.
        """)

        st.subheader("🌐 Real-time Medical Data")
        drug_search = st.text_input("Search Drug for Live Safety Data", "Lisinopril")
        if st.button("Fetch Live Insights"):
            with st.spinner("Connecting to OpenFDA..."):
                alerts = fetcher.fetch_drug_events(drug_search)
                if alerts:
                    for alert in alerts:
                        with st.expander(f"⚠️ {alert.headline}"):
                            st.write(alert.details)
                            st.caption(f"Source: {alert.source} | Date: {alert.timestamp}")
                else:
                    st.warning("No recent adverse events found for this drug.")

    with col2:
        st.subheader("3D Reconstruction")
        with st.spinner("Calculating 3D Isosurface..."):
            volume = generate_heart_volume(grid_res)
            
            # Create Plotly 3D Isosurface
            fig = go.Figure(data=go.Isosurface(
                x=np.linspace(-1.5, 1.5, grid_res).repeat(grid_res*grid_res),
                y=np.tile(np.linspace(-1.5, 1.5, grid_res), grid_res*grid_res),
                z=np.tile(np.linspace(-1.5, 1.5, grid_res).repeat(grid_res), grid_res),
                value=volume.flatten(),
                isomin=threshold - 0.1,
                isomax=threshold + 0.1,
                surface_count=3,
                caps=dict(x_show=False, y_show=False),
                colorscale='Portland',
                opacity=0.6,
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X-Axis',
                    yaxis_title='Y-Axis',
                    zaxis_title='Z-Axis',
                    bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💡 Google MedTech AI Diagnostic Assistant")
    patient_query = st.text_area("Analyze patient case with Med-Gemini logic:", "Patient reports high blood pressure and recurring headaches.")
    if st.button("Generate MedTech Insight"):
        with st.spinner("Synthesizing clinical data..."):
            medical_response = pipeline.run(patient_query)
            st.success("MedTech Analysis Complete")
            st.write(medical_response)

if __name__ == "__main__":
    run_3d_visualization()

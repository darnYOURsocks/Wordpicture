#!/usr/bin/env python3
"""
AIDNA 3D Sonic Ripple Generator - Streamlit Implementation
Adapted from the original PyGame version for web deployment
"""

import streamlit as st
import numpy as np
import math
import hashlib
import time
from typing import Dict, Tuple, List, Optional
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy import ndimage
import io
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="AIDNA 3D Sonic Ripple Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4a9eff;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #4a9eff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .pattern-card {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #4a9eff;
    }
    .param-slider {
        margin-bottom: 1.5rem;
    }
    .stButton button {
        background: linear-gradient(45deg, #4a9eff, #45b7d1);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(45deg, #45b7d1, #4a9eff);
    }
</style>
""", unsafe_allow_html=True)

class ReactionDiffusionSimulator:
    """Gray-Scott reaction-diffusion system simulator"""
    
    def __init__(self, size: int = 64):
        self.size = size
        self.dt = 1.0
        self.Du = 0.16  # Diffusion rate for u
        self.Dv = 0.08  # Diffusion rate for v
        
        # Initialize chemical concentrations
        self.u = np.ones((size, size)) + np.random.uniform(-0.05, 0.05, (size, size))
        self.v = np.random.uniform(-0.05, 0.05, (size, size))
        
        # Add initial perturbation in center
        center = size // 2
        self.v[center-5:center+5, center-5:center+5] = 0.5

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Calculate Laplacian using finite differences"""
        lapl = np.zeros_like(field)
        lapl[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] - 4 * field[1:-1, 1:-1]
        )
        return lapl

    def step(self, F: float, k: float):
        """Perform one simulation step"""
        lap_u = self.laplacian(self.u)
        lap_v = self.laplacian(self.v)
        
        uvv = self.u * self.v * self.v
        
        du_dt = self.Du * lap_u - uvv + F * (1 - self.u)
        dv_dt = self.Dv * lap_v + uvv - (F + k) * self.v
        
        self.u += self.dt * du_dt
        self.v += self.dt * dv_dt
        
        # Clamp values
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)

    def simulate(self, F: float, k: float, steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation for specified steps"""
        # Reset state
        self.__init__(self.size)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(steps):
            self.step(F, k)
            
            # Update progress
            if i % 10 == 0:
                progress = (i + 1) / steps
                progress_bar.progress(progress)
                status_text.text(f"Simulation progress: {int(progress * 100)}%")
        
        progress_bar.empty()
        status_text.empty()
        
        return self.u.copy(), self.v.copy()

class AIDNA3DRippleGenerator:
    """Main application class for Streamlit"""
    
    def __init__(self):
        # Initialize components
        self.simulator = ReactionDiffusionSimulator()
        
        # State
        self.input_word = ""
        self.current_pattern = "stripes"
        
        # AIDNA duality mappings
        self.duality_mappings = {
            'ambition': {'F': 0.035, 'k': 0.060, 'pattern': 'stripes', 'freq': 440},
            'rest': {'F': 0.025, 'k': 0.055, 'pattern': 'spots', 'freq': 220},
            'order': {'F': 0.020, 'k': 0.050, 'pattern': 'labyrinth', 'freq': 330},
            'chaos': {'F': 0.040, 'k': 0.065, 'pattern': 'mixed', 'freq': 660},
            'logic': {'F': 0.030, 'k': 0.062, 'pattern': 'stripes', 'freq': 523},
            'emotion': {'F': 0.038, 'k': 0.058, 'pattern': 'spots', 'freq': 293},
            'create': {'F': 0.042, 'k': 0.061, 'pattern': 'mixed', 'freq': 392},
            'destroy': {'F': 0.028, 'k': 0.054, 'pattern': 'labyrinth', 'freq': 196},
            'hope': {'F': 0.033, 'k': 0.059, 'pattern': 'spots', 'freq': 494},
            'fear': {'F': 0.026, 'k': 0.053, 'pattern': 'stripes', 'freq': 147},
            'love': {'F': 0.031, 'k': 0.057, 'pattern': 'mixed', 'freq': 369},
            'doubt': {'F': 0.036, 'k': 0.063, 'pattern': 'labyrinth', 'freq': 261}
        }
        
        self.pattern_colors = {
            'stripes': (255, 107, 107),
            'spots': (78, 205, 196),
            'labyrinth': (69, 183, 209),
            'mixed': (249, 202, 36)
        }
        
        self.pattern_info = {
            'stripes': 'Linear patterns - Associated with goal-oriented thinking',
            'spots': 'Clustered patterns - Associated with emotional resonance',
            'labyrinth': 'Complex patterns - Associated with introspective processing',
            'mixed': 'Dynamic patterns - Associated with creative synthesis'
        }

    def calculate_parameters(self, word: str) -> Dict:
        """Calculate reaction-diffusion parameters from word"""
        if not word:
            return {'F': 0.035, 'k': 0.060, 'pattern': 'stripes', 'freq': 440}
        
        word_lower = word.lower()
        
        # Check for direct duality matches
        for key, params in self.duality_mappings.items():
            if key in word_lower:
                return params
        
        # Hash-based parameter generation
        hash_obj = hashlib.md5(word.encode())
        hash_bytes = hash_obj.digest()
        hash_int = int.from_bytes(hash_bytes[:4], 'big')
        normalized_hash = hash_int / (2**32)
        
        F = 0.020 + normalized_hash * 0.025  # Range: 0.020-0.045
        k = 0.050 + normalized_hash * 0.015  # Range: 0.050-0.065
        freq = 200 + normalized_hash * 400   # Range: 200-600 Hz
        
        patterns = ['stripes', 'spots', 'labyrinth', 'mixed']
        pattern = patterns[int(normalized_hash * len(patterns))]
        
        return {'F': F, 'k': k, 'pattern': pattern, 'freq': freq}

    def create_enhanced_mesh_data(self, height_data: np.ndarray, pattern: str, params: Dict) -> Dict:
        """Create enhanced 3D mesh data with better visual features"""
        size = height_data.shape[0]
        vertices = []
        colors = []
        
        base_color = self.pattern_colors[pattern]
        
        # Calculate mesh scale based on pattern complexity
        height_scale = 2.5 + (params['k'] - 0.05) * 20  # Dynamic height scaling
        mesh_size = 12
        
        # Generate enhanced vertices with height mapping
        for i in range(size):
            for j in range(size):
                x = (i / (size - 1) - 0.5) * mesh_size
                z = (j / (size - 1) - 0.5) * mesh_size
                y = height_data[i, j] * height_scale
                
                vertices.append([x, y, z])
                
                # Enhanced color calculation with pattern-specific effects
                intensity = height_data[i, j]
                
                if pattern == 'stripes':
                    # Add directional color variation
                    stripe_factor = 0.8 + 0.2 * np.sin(i * 0.3)
                    color_mult = stripe_factor * (0.4 + 0.6 * intensity)
                elif pattern == 'spots':
                    # Radial color variation
                    center_dist = np.sqrt((i - size//2)**2 + (j - size//2)**2) / (size//2)
                    color_mult = (0.3 + 0.7 * intensity) * (1.0 - 0.3 * center_dist)
                elif pattern == 'labyrinth':
                    # Complex color variation
                    color_mult = 0.2 + 0.8 * intensity + 0.1 * np.sin(i * 0.5) * np.cos(j * 0.5)
                else:  # mixed
                    # Dynamic color mixing
                    color_mult = 0.3 + 0.7 * intensity * (1 + 0.2 * np.sin(i * j * 0.01))
                
                color = [int(c * np.clip(color_mult, 0, 1)) for c in base_color]
                colors.append(color)
        
        # Generate triangles
        triangles = []
        for i in range(size - 1):
            for j in range(size - 1):
                v1 = i * size + j
                v2 = i * size + j + 1
                v3 = (i + 1) * size + j
                v4 = (i + 1) * size + j + 1
                
                # Create two triangles per quad
                triangles.extend([[v1, v2, v3], [v2, v4, v3]])
        
        return {
            'vertices': np.array(vertices),
            'triangles': np.array(triangles),
            'colors': np.array(colors),
            'pattern': pattern,
            'params': params
        }

    def generate_pattern(self, word: str):
        """Generate new pattern based on word with enhanced features"""
        if not word.strip():
            st.warning("Please enter a word to generate a pattern")
            return
            
        with st.spinner("Generating pattern..."):
            params = self.calculate_parameters(word)
            self.last_params = params  # Store for UI display
            
            # Enhanced simulation with adaptive parameters
            steps = min(200, max(100, len(word) * 10))  # Adaptive step count
            u, v = self.simulator.simulate(params['F'], params['k'], steps)
            
            # Create enhanced mesh with better detail
            self.mesh_data = self.create_enhanced_mesh_data(v, params['pattern'], params)
            self.current_pattern = params['pattern']
            self.input_word = word
            
            # Update session state
            st.session_state.mesh_data = self.mesh_data
            st.session_state.current_pattern = self.current_pattern
            st.session_state.input_word = word
            st.session_state.last_params = params

    def render_3d_plot(self):
        """Render 3D visualization using Plotly"""
        if not hasattr(self, 'mesh_data') or 'mesh_data' not in st.session_state:
            return None
            
        mesh_data = st.session_state.mesh_data
        vertices = mesh_data['vertices']
        triangles = mesh_data['triangles']
        colors = mesh_data['colors']
        
        # Extract x, y, z coordinates
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        
        # Convert colors to Plotly format
        colors_plotly = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colors]
        
        # Create mesh3d plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                vertexcolor=colors_plotly,
                opacity=0.9,
                flatshading=True,
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.8,
                    fresnel=0.1,
                    specular=1,
                    roughness=0.5
                ),
                lightposition=dict(x=100, y=100, z=100)
            )
        ])
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor='rgb(10, 10, 15)'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def render_2d_visualization(self):
        """Render 2D visualization of the pattern"""
        if not hasattr(self, 'mesh_data') or 'mesh_data' not in st.session_state:
            return None
            
        mesh_data = st.session_state.mesh_data
        vertices = mesh_data['vertices']
        size = int(np.sqrt(len(vertices)))
        
        # Extract height data and reshape to 2D
        height_data = vertices[:, 1].reshape(size, size)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap with color mapping
        im = ax.imshow(height_data, cmap='viridis', interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Height', rotation=270, labelpad=15)
        
        # Customize plot
        ax.set_title(f'2D Visualization: {st.session_state.current_pattern.capitalize()} Pattern')
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig

    def export_pattern_data(self):
        """Export current pattern data to JSON file"""
        if not hasattr(self, 'mesh_data') or 'mesh_data' not in st.session_state:
            st.warning("No pattern data to export")
            return
            
        mesh_data = st.session_state.mesh_data
        
        export_data = {
            'word': st.session_state.input_word,
            'pattern': st.session_state.current_pattern,
            'parameters': st.session_state.last_params,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'mesh_info': {
                'vertex_count': len(mesh_data['vertices']),
                'triangle_count': len(mesh_data['triangles']),
                'pattern_type': mesh_data.get('pattern', 'unknown')
            }
        }
        
        # Convert to JSON string
        json_str = json.dumps(export_data, indent=2)
        
        # Create download button
        st.download_button(
            label="Download Pattern Data",
            data=json_str,
            file_name=f"aidna_pattern_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def load_pattern_preset(self, preset_name: str):
        """Load predefined pattern presets"""
        presets = {
            'meditation': 'peace',
            'energy': 'power',
            'creativity': 'create',
            'focus': 'logic',
            'calm': 'rest',
            'excitement': 'chaos',
            'love': 'love',
            'strength': 'ambition'
        }
        
        if preset_name in presets:
            word = presets[preset_name]
            self.generate_pattern(word)
            st.success(f"Loaded preset: {preset_name} -> {word}")
        else:
            st.error(f"Unknown preset: {preset_name}")

    def run(self):
        """Main application method"""
        # Initialize session state
        if 'mesh_data' not in st.session_state:
            st.session_state.mesh_data = None
        if 'current_pattern' not in st.session_state:
            st.session_state.current_pattern = "stripes"
        if 'input_word' not in st.session_state:
            st.session_state.input_word = ""
        if 'last_params' not in st.session_state:
            st.session_state.last_params = {}
        
        # Header
        st.markdown('<h1 class="main-header">AIDNA 3D Sonic Ripple Generator</h1>', unsafe_allow_html=True)
        
        # Create sidebar
        with st.sidebar:
            st.header("Pattern Controls")
            
            # Word input
            word_input = st.text_input(
                "Enter a word or concept:",
                value=st.session_state.input_word,
                placeholder="Type a word to generate a pattern..."
            )
            
            # Generate button
            if st.button("Generate Pattern", use_container_width=True):
                self.generate_pattern(word_input)
            
            # Preset buttons
            st.subheader("Presets")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Meditation", use_container_width=True):
                    self.load_pattern_preset('meditation')
                if st.button("Energy", use_container_width=True):
                    self.load_pattern_preset('energy')
                if st.button("Creativity", use_container_width=True):
                    self.load_pattern_preset('creativity')
                if st.button("Focus", use_container_width=True):
                    self.load_pattern_preset('focus')
            
            with col2:
                if st.button("Calm", use_container_width=True):
                    self.load_pattern_preset('calm')
                if st.button("Excitement", use_container_width=True):
                    self.load_pattern_preset('excitement')
                if st.button("Love", use_container_width=True):
                    self.load_pattern_preset('love')
                if st.button("Strength", use_container_width=True):
                    self.load_pattern_preset('strength')
            
            # Pattern info
            if st.session_state.current_pattern:
                st.divider()
                st.subheader("Pattern Information")
                
                pattern_color = self.pattern_colors.get(st.session_state.current_pattern, (255, 255, 255))
                st.markdown(f"**Type**: {st.session_state.current_pattern.capitalize()}")
                st.markdown(f"**Description**: {self.pattern_info.get(st.session_state.current_pattern, 'No description available')}")
                
                # Display parameters if available
                if st.session_state.last_params:
                    st.markdown("**Parameters**:")
                    st.markdown(f"- F: {st.session_state.last_params.get('F', 0):.4f}")
                    st.markdown(f"- k: {st.session_state.last_params.get('k', 0):.4f}")
                    st.markdown(f"- Frequency: {st.session_state.last_params.get('freq', 0):.1f} Hz")
            
            # Export button
            st.divider()
            self.export_pattern_data()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 3D Visualization
            st.subheader("3D Visualization")
            fig_3d = self.render_3d_plot()
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("Enter a word and click 'Generate Pattern' to create a 3D visualization")
        
        with col2:
            # Pattern details and 2D visualization
            st.subheader("Pattern Details")
            
            if st.session_state.mesh_data:
                # 2D Visualization
                fig_2d = self.render_2d_visualization()
                if fig_2d:
                    st.pyplot(fig_2d)
                
                # Pattern statistics
                mesh_data = st.session_state.mesh_data
                st.markdown("**Pattern Statistics**:")
                st.markdown(f"- Vertices: {len(mesh_data['vertices'])}")
                st.markdown(f"- Triangles: {len(mesh_data['triangles'])}")
                st.markdown(f"- Pattern Type: {mesh_data.get('pattern', 'unknown')}")
            else:
                st.info("No pattern data available. Generate a pattern to see details here.")
        
        # Information section
        st.divider()
        st.subheader("About AIDNA 3D Sonic Ripple Generator")
        
        st.markdown("""
        This interactive tool transforms words and concepts into unique 3D visual patterns using 
        reaction-diffusion systems. Each word generates a distinct pattern based on its semantic properties.
        
        **How it works**:
        1. Enter a word or concept in the sidebar
        2. The system calculates parameters based on the word's properties
        3. A reaction-diffusion simulation generates a pattern
        4. The pattern is visualized in 3D and 2D
        
        **Pattern Types**:
        - **Stripes**: Linear patterns associated with goal-oriented thinking
        - **Spots**: Clustered patterns associated with emotional resonance
        - **Labyrinth**: Complex patterns associated with introspective processing
        - **Mixed**: Dynamic patterns associated with creative synthesis
        """)
        
        # Add some examples
        st.subheader("Example Words to Try")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("- Ambition")
            st.markdown("- Chaos")
            st.markdown("- Love")
            
        with col2:
            st.markdown("- Order")
            st.markdown("- Logic")
            st.markdown("- Hope")
            
        with col3:
            st.markdown("- Creativity")
            st.markdown("- Emotion")
            st.markdown("- Fear")
            
        with col4:
            st.markdown("- Destruction")
            st.markdown("- Peace")
            st.markdown("- Strength")

# Create and run the application
if __name__ == "__main__":
    app = AIDNA3DRippleGenerator()
    app.run()

    def generate_pattern(self, word: str):
        """Enhanced pattern generation with advanced analysis"""
        if self.simulation_running:
            return
            
        self.simulation_running = True
        params = self.calculate_parameters(word)
        self.last_params = params
        
        # Run enhanced simulation in thread
        def simulate():
            try:
                # Enhanced simulation with adaptive parameters
                steps = min(250, max(100, len(word) * 12))
                u, v = self.simulator.simulate(params['F'], params['k'], steps)
                
                # Advanced pattern analysis
                self.current_analysis = self.pattern_analyzer.analyze_pattern_complexity(v, params)
                
                # Create enhanced mesh
                self.mesh_data = self.create_enhanced_mesh_data(v, params['pattern'], params)
                self.current_pattern = params['pattern']
                
                # Save to pattern library
                self.pattern_library.save_pattern(word, params, self.current_analysis)
                
                # Create particle effects
                if self.show_particles:
                    screen_center = (self.width // 2, self.height // 2)
                    self.particles.extend(
                        EnhancedVisualizationEffects.create_particle_system(
                            screen_center, params['pattern']
                        )
                    )
                
                # Enhanced audio feedback
                if self.audio_enabled and self.is_playing:
                    self.play_enhanced_audio(params)
                    # Add to audio visualizer
                    self.audio_visualizer.add_frequency_data(params['freq'], 0.7)
                
                self.simulation_running = False
                
            except Exception as e:
                print(f"Enhanced simulation error: {e}")
                self.simulation_running = False
        
        threading.Thread(target=simulate, daemon=True).start()
    
    def draw_enhanced_analysis_panel(self):
        """Draw advanced pattern analysis panel"""
        if not self.current_analysis:
            return
        
        panel_rect = pygame.Rect(self.width - 300, self.height - 300, 280, 280)
        pygame.draw.rect(self.screen, (15, 15, 25), panel_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), panel_rect, 2)
        
        # Title
        title_surface = self.font.render("Pattern Analysis", True, (100, 200, 255))
        self.screen.blit(title_surface, (panel_rect.x + 10, panel_rect.y + 10))
        
        y_offset = panel_rect.y + 40
        line_height = 18
        
        # Analysis metrics
        metrics = [
            ("Complexity", f"{self.current_analysis.get('complexity_score', 0):.3f}"),
            ("Variance", f"{self.current_analysis.get('variance', 0):.3f}"),
            ("Edge Density", f"{self.current_analysis.get('edge_density', 0):.3f}"),
            ("Fractal Dim", f"{self.current_analysis.get('fractal_dimension', 0):.2f}"),
            ("Spatial Freq", f"{self.current_analysis.get('spatial_frequency', 0):.3f}"),
            ("Emotion Res", f"{self.current_analysis.get('emotional_resonance', 0):.3f}"),
            ("Pattern Type", self.current_analysis.get('pattern_type', 'unknown'))
        ]
        
        for label, value in metrics:
            # Label
            label_surface = self.small_font.render(f"{label}:", True, (200, 200, 200))
            self.screen.blit(label_surface, (panel_rect.x + 10, y_offset))
            
            # Value with color coding
            if label == "Emotion Res":
                resonance = float(value) if value.replace('.', '').isdigit() else 0
                color = (int(255 * (1 - resonance)), int(255 * resonance), 100)
            elif label == "Complexity":
                complexity = float(value) if value.replace('.', '').isdigit() else 0
                color = (255, int(255 * (1 - complexity)), int(255 * complexity))
            else:
                color = (150, 255, 150)
            
            value_surface = self.small_font.render(str(value), True, color)
            self.screen.blit(value_surface, (panel_rect.x + 120, y_offset))
            
            y_offset += line_height
        
        # Pattern recommendations
        y_offset += 10
        rec_title = self.small_font.render("Recommendations:", True, (100, 200, 255))
        self.screen.blit(rec_title, (panel_rect.x + 10, y_offset))
        y_offset += 20
        
        recommendations = self.pattern_library.get_pattern_recommendations(self.input_word)
        for i, rec_word in enumerate(recommendations[:4]):
            rec_surface = self.small_font.render(f"• {rec_word}", True, (180, 180, 180))
            self.screen.blit(rec_surface, (panel_rect.x + 15, y_offset))
            y_offset += 15
    
    def draw_ui(self):
        """Enhanced UI with all new features"""
        # Main info panel
        ui_rect = pygame.Rect(10, 10, 380, 220)
        pygame.draw.rect(self.screen, (20, 20, 30), ui_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), ui_rect, 2)
        
        y_offset = 25
        
        # Title with version
        title = self.font.render("AIDNA 3D Sonic Ripple Generator v2.0", True, (100, 200, 255))
        self.screen.blit(title, (20, y_offset))
        y_offset += 35
        
        # Input word with cursor and suggestions
        cursor = "_" if int(time.time() * 2) % 2 else " "
        input_text = f"Word: {self.input_word}{cursor}"
        word_surface = self.small_font.render(input_text, True, (200, 255, 200))
        self.screen.blit(word_surface, (20, y_offset))
        y_offset += 25
        
        # Pattern with advanced info
        pattern_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
        pattern_text = f"Pattern: {self.current_pattern}"
        pattern_surface = self.small_font.render(pattern_text, True, pattern_color)
        self.screen.blit(pattern_surface, (20, y_offset))
        
        # Pattern color indicator with complexity
        color_rect = pygame.Rect(180, y_offset + 2, 15, 15)
        pygame.draw.rect(self.screen, pattern_color, color_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
        
        # Complexity indicator
        if self.current_analysis:
            complexity = self.current_analysis.get('complexity_score', 0)
            comp_text = f"C:{complexity:.2f}"
            comp_surface = self.small_font.render(comp_text, True, (255, 200, 100))
            self.screen.blit(comp_surface, (210, y_offset))
        
        y_offset += 25
        
        # Enhanced status indicators
        self.draw_status_indicators(y_offset)
        
        # Audio visualizer
        if self.show_audio_viz and self.audio_enabled:
            self.audio_visualizer.render_waveform(self.screen, 20, y_offset + 80)
            self.audio_visualizer.render_frequency_circle(
                self.screen, (320, y_offset + 110), 30
            )
        
        # Control buttons
        self.draw_control_buttons()
        
        # Enhanced analysis panel
        self.draw_enhanced_analysis_panel()
        
        # Performance metrics
        self.draw_performance_metrics()
            def generate_pattern(self, word: str):
        """Generate new pattern based on word with enhanced features"""
        if self.simulation_running:
            return
            
        self.simulation_running = True
        params = self.calculate_parameters(word)
        self.last_params = params  # Store for UI display
        
        # Run simulation in thread to avoid blocking
        def simulate():
            try:
                # Enhanced simulation with adaptive parameters
                steps = min(200, max(100, len(word) * 10))  # Adaptive step count
                u, v = self.simulator.simulate(params['F'], params['k'], steps)
                
                # Create enhanced mesh with better detail
                self.mesh_data = self.create_enhanced_mesh_data(v, params['pattern'], params)
                self.current_pattern = params['pattern']
                
                # Enhanced audio feedback
                if self.audio_enabled and self.is_playing:
                    self.play_enhanced_audio(params)
                
                self.simulation_running = False
                
            except Exception as e:
                print(f"Simulation error: {e}")
                self.simulation_running = False
        
        threading.Thread(target=simulate, daemon=True).start()
    
    def create_enhanced_mesh_data(self, height_data: np.ndarray, pattern: str, params: Dict) -> Dict:
        """Create enhanced 3D mesh data with better visual features"""
        size = height_data.shape[0]
        vertices = []
        triangles = []
        colors = []
        normals = []
        
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
        
        # Generate triangles with better topology
        for i in range(size - 1):
            for j in range(size - 1):
                v1 = i * size + j
                v2 = i * size + j + 1
                v3 = (i + 1) * size + j
                v4 = (i + 1) * size + j + 1
                
                # Create two triangles per quad
                triangles.extend([[v1, v2, v3], [v2, v4, v3]])
        
        # Calculate vertex normals for better lighting
        vertex_normals = np.zeros((len(vertices), 3))
        for tri in triangles:
            v1, v2, v3 = [np.array(vertices[i]) for i in tri]
            normal = np.cross(v2 - v1, v3 - v1)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
                for i in tri:
                    vertex_normals[i] += normal
        
        # Normalize vertex normals
        for i in range(len(vertex_normals)):
            if np.linalg.norm(vertex_normals[i]) > 0:
                vertex_normals[i] = vertex_normals[i] / np.linalg.norm(vertex_normals[i])
        
        return {
            'vertices': np.array(vertices),
            'triangles': np.array(triangles),
            'colors': np.array(colors),
            'normals': vertex_normals,
            'pattern': pattern,
            'params': params
        }#!/usr/bin/env python3
"""
AIDNA 3D Sonic Ripple Generator - Python Implementation
Advanced 3D visualization with reaction-diffusion patterns and audio synthesis

Dependencies:
pip install pygame numpy matplotlib scipy sounddevice moderngl PyOpenGL PyOpenGL_accelerate

Optional for enhanced audio:
pip install pyaudio

Usage:
python aidna_3d_sonic_ripple.py
"""

import pygame
import numpy as np
import moderngl
import math
import hashlib
import threading
import time
from typing import Dict, Tuple, List, Optional
import json
import os

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    print("Audio disabled: Install sounddevice for audio features")
    AUDIO_AVAILABLE = False

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    print("OpenGL not available, using software rendering")
    OPENGL_AVAILABLE = False


class AudioSynthesizer:
    """Advanced audio synthesizer for sonic feedback"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.current_frequency = 440.0
        self.volume = 0.3
        self.wave_type = 'sine'
        
        if AUDIO_AVAILABLE:
            self.stream = None
            self.initialize_audio()
    
    def initialize_audio(self):
        """Initialize audio output stream"""
        if not AUDIO_AVAILABLE:
            return
            
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=1024
            )
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            global AUDIO_AVAILABLE
            AUDIO_AVAILABLE = False
    
    def _audio_callback(self, outdata, frames, time, status):
        """Audio callback for real-time synthesis"""
        if not self.is_playing:
            outdata[:] = 0
            return
            
        t = np.linspace(0, frames / self.sample_rate, frames, False)
        
        if self.wave_type == 'sine':
            wave = np.sin(2 * np.pi * self.current_frequency * t)
        elif self.wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * self.current_frequency * t))
        elif self.wave_type == 'sawtooth':
            wave = 2 * (t * self.current_frequency - np.floor(t * self.current_frequency + 0.5))
        else:
            wave = np.sin(2 * np.pi * self.current_frequency * t)
        
        # Apply envelope and volume
        envelope = np.exp(-t * 2)  # Decay envelope
        outdata[:, 0] = wave * envelope * self.volume
    
    def play_frequency(self, frequency: float, duration: float = 0.5):
        """Play a specific frequency"""
        if not AUDIO_AVAILABLE or not self.stream:
            return
            
        self.current_frequency = frequency
        self.is_playing = True
        
        if not self.stream.active:
            self.stream.start()
        
        # Stop after duration
        threading.Timer(duration, self.stop).start()
    
    def play_chord(self, frequencies: List[float], duration: float = 1.0):
        """Play multiple frequencies as a chord"""
        if not frequencies:
            return
            
        # For simplicity, play the root frequency
        # In a full implementation, you'd mix multiple oscillators
        self.play_frequency(frequencies[0], duration)
    
    def stop(self):
        """Stop audio playback"""
        self.is_playing = False
    
    def cleanup(self):
        """Clean up audio resources"""
        if AUDIO_AVAILABLE and self.stream:
            self.stream.stop()
            self.stream.close()


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
        
        for _ in range(steps):
            self.step(F, k)
        
        return self.u.copy(), self.v.copy()


class Camera3D:
    """3D camera with controls"""
    
    def __init__(self):
        self.distance = 15.0
        self.elevation = 20.0
        self.azimuth = 0.0
        self.target = np.array([0.0, 0.0, 0.0])
        self.position = self._calculate_position()
    
    def _calculate_position(self) -> np.ndarray:
        """Calculate camera position from spherical coordinates"""
        elev_rad = math.radians(self.elevation)
        azim_rad = math.radians(self.azimuth)
        
        x = self.distance * math.cos(elev_rad) * math.cos(azim_rad)
        y = self.distance * math.sin(elev_rad)
        z = self.distance * math.cos(elev_rad) * math.sin(azim_rad)
        
        return self.target + np.array([x, y, z])
    
    def rotate(self, d_azimuth: float, d_elevation: float):
        """Rotate camera"""
        self.azimuth += d_azimuth
        self.elevation = np.clip(self.elevation + d_elevation, -89, 89)
        self.position = self._calculate_position()
    
    def zoom(self, factor: float):
        """Zoom camera"""
        self.distance = np.clip(self.distance * factor, 5.0, 50.0)
        self.position = self._calculate_position()


class AdvancedPatternAnalyzer:
    """Advanced pattern analysis and classification system"""
    
    def __init__(self):
        self.pattern_history = []
        self.complexity_metrics = {}
        
    def analyze_pattern_complexity(self, height_data: np.ndarray, params: Dict) -> Dict:
        """Analyze pattern complexity using multiple metrics"""
        # Calculate various complexity measures
        variance = np.var(height_data)
        edge_density = self._calculate_edge_density(height_data)
        fractal_dimension = self._estimate_fractal_dimension(height_data)
        spatial_frequency = self._analyze_spatial_frequency(height_data)
        
        complexity_score = (
            variance * 0.3 + 
            edge_density * 0.25 + 
            fractal_dimension * 0.25 + 
            spatial_frequency * 0.2
        )
        
        return {
            'complexity_score': complexity_score,
            'variance': variance,
            'edge_density': edge_density,
            'fractal_dimension': fractal_dimension,
            'spatial_frequency': spatial_frequency,
            'pattern_type': self._classify_pattern_type(height_data),
            'emotional_resonance': self._calculate_emotional_resonance(params)
        }
    
    def _calculate_edge_density(self, data: np.ndarray) -> float:
        """Calculate edge density using Sobel operators"""
        from scipy import ndimage
        
        # Sobel edge detection
        sx = ndimage.sobel(data, axis=0, mode='constant')
        sy = ndimage.sobel(data, axis=1, mode='constant')
        edge_magnitude = np.sqrt(sx**2 + sy**2)
        
        return np.mean(edge_magnitude)
    
    def _estimate_fractal_dimension(self, data: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method"""
        # Simple fractal dimension estimation
        sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in sizes:
            # Count non-empty boxes at each scale
            h, w = data.shape
            count = 0
            for i in range(0, h, size):
                for j in range(0, w, size):
                    box = data[i:min(i+size, h), j:min(j+size, w)]
                    if np.any(box > 0.1):  # Threshold for "non-empty"
                        count += 1
            counts.append(count)
        
        # Linear regression to estimate dimension
        if len(counts) > 1:
            log_sizes = np.log(sizes)
            log_counts = np.log(np.array(counts) + 1)
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            return abs(slope)
        return 1.5
    
    def _analyze_spatial_frequency(self, data: np.ndarray) -> float:
        """Analyze spatial frequency content using FFT"""
        # 2D FFT to analyze frequency content
        fft = np.fft.fft2(data)
        magnitude = np.abs(fft)
        
        # Calculate frequency distribution
        freq_energy = np.sum(magnitude[1:, 1:])  # Exclude DC component
        total_energy = np.sum(magnitude)
        
        return freq_energy / total_energy if total_energy > 0 else 0
    
    def _classify_pattern_type(self, data: np.ndarray) -> str:
        """Advanced pattern classification beyond basic types"""
        # Analyze directional patterns
        h_gradient = np.abs(np.gradient(data, axis=1))
        v_gradient = np.abs(np.gradient(data, axis=0))
        
        h_dominance = np.mean(h_gradient)
        v_dominance = np.mean(v_gradient)
        
        # Analyze clustering
        from scipy.ndimage import label
        binary = data > np.mean(data)
        labeled, num_features = label(binary)
        
        if h_dominance > v_dominance * 1.5:
            return "horizontal_stripes"
        elif v_dominance > h_dominance * 1.5:
            return "vertical_stripes"
        elif num_features > data.size * 0.1:
            return "scattered_spots"
        elif num_features < data.size * 0.01:
            return "large_domains"
        else:
            return "complex_pattern"
    
    def _calculate_emotional_resonance(self, params: Dict) -> float:
        """Calculate emotional resonance score"""
        F = params.get('F', 0.035)
        k = params.get('k', 0.060)
        freq = params.get('freq', 440)
        
        # Emotional resonance based on parameter ranges
        f_resonance = 1 - abs(F - 0.0325) / 0.025  # Peak at 0.0325
        k_resonance = 1 - abs(k - 0.0575) / 0.015  # Peak at 0.0575
        freq_resonance = 1 - abs(freq - 369) / 300  # Peak at 369 Hz (love frequency)
        
        return np.clip((f_resonance + k_resonance + freq_resonance) / 3, 0, 1)


class PatternLibrary:
    """Library for saving, loading, and managing pattern collections"""
    
    def __init__(self, library_path: str = "pattern_library.json"):
        self.library_path = library_path
        self.patterns = self.load_library()
    
    def load_library(self) -> Dict:
        """Load pattern library from file"""
        try:
            if os.path.exists(self.library_path):
                with open(self.library_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading pattern library: {e}")
        return {}
    
    def save_library(self):
        """Save pattern library to file"""
        try:
            with open(self.library_path, 'w') as f:
                json.dump(self.patterns, f, indent=2)
        except Exception as e:
            print(f"Error saving pattern library: {e}")
    
    def save_pattern(self, word: str, params: Dict, analysis: Dict):
        """Save a pattern to the library"""
        pattern_data = {
            'word': word,
            'parameters': params,
            'analysis': analysis,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'usage_count': self.patterns.get(word, {}).get('usage_count', 0) + 1
        }
        
        self.patterns[word] = pattern_data
        self.save_library()
    
    def get_similar_patterns(self, target_params: Dict, threshold: float = 0.1) -> List[str]:
        """Find patterns with similar parameters"""
        similar = []
        target_f = target_params.get('F', 0.035)
        target_k = target_params.get('k', 0.060)
        
        for word, data in self.patterns.items():
            params = data.get('parameters', {})
            f_diff = abs(params.get('F', 0.035) - target_f)
            k_diff = abs(params.get('k', 0.060) - target_k)
            
            if f_diff < threshold and k_diff < threshold:
                similar.append(word)
        
        return similar
    
    def get_pattern_recommendations(self, current_word: str) -> List[str]:
        """Get pattern recommendations based on usage and similarity"""
        if current_word in self.patterns:
            current_params = self.patterns[current_word]['parameters']
            similar = self.get_similar_patterns(current_params)
            
            # Sort by usage count
            recommendations = sorted(
                [(word, self.patterns[word]['usage_count']) for word in similar],
                key=lambda x: x[1],
                reverse=True
            )
            
            return [word for word, _ in recommendations[:5]]
        
        # Return most popular patterns
        popular = sorted(
            self.patterns.items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )
        
        return [word for word, _ in popular[:5]]


class EnhancedVisualizationEffects:
    """Advanced visualization effects and post-processing"""
    
    @staticmethod
    def apply_glow_effect(surface: pygame.Surface, glow_color: Tuple[int, int, int], 
                         intensity: float = 0.3) -> pygame.Surface:
        """Apply glow effect to surface"""
        # Create glow surface
        glow_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        
        # Draw multiple layers with decreasing alpha
        for i in range(5):
            alpha = int(255 * intensity * (1 - i * 0.2))
            temp_surface = surface.copy()
            temp_surface.set_alpha(alpha)
            
            # Blur effect simulation by drawing offset copies
            for dx in range(-i, i+1):
                for dy in range(-i, i+1):
                    glow_surface.blit(temp_surface, (dx, dy))
        
        return glow_surface
    
    @staticmethod
    def create_particle_system(center: Tuple[int, int], pattern_type: str) -> List[Dict]:
        """Create particle system based on pattern type"""
        particles = []
        
        particle_configs = {
            'stripes': {'count': 20, 'speed': 2, 'spread': 180},
            'spots': {'count': 30, 'speed': 1, 'spread': 360},
            'labyrinth': {'count': 40, 'speed': 0.5, 'spread': 90},
            'mixed': {'count': 50, 'speed': 3, 'spread': 270}
        }
        
        config = particle_configs.get(pattern_type, particle_configs['mixed'])
        
        for _ in range(config['count']):
            angle = np.random.uniform(0, config['spread'])
            speed = np.random.uniform(0.5, config['speed'])
            
            particle = {
                'x': center[0],
                'y': center[1],
                'vx': speed * np.cos(np.radians(angle)),
                'vy': speed * np.sin(np.radians(angle)),
                'life': 1.0,
                'decay': np.random.uniform(0.01, 0.03)
            }
            particles.append(particle)
        
        return particles
    
    @staticmethod
    def update_particles(particles: List[Dict]) -> List[Dict]:
        """Update particle system"""
        active_particles = []
        
        for particle in particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= particle['decay']
            
            if particle['life'] > 0:
                active_particles.append(particle)
        
        return active_particles
    
    @staticmethod
    def render_particles(screen: pygame.Surface, particles: List[Dict], 
                        color: Tuple[int, int, int]):
        """Render particle system"""
        for particle in particles:
            alpha = int(255 * particle['life'])
            particle_color = (*color, alpha)
            
            size = max(1, int(3 * particle['life']))
            pos = (int(particle['x']), int(particle['y']))
            
            pygame.draw.circle(screen, color, pos, size)


class AudioVisualizer:
    """Real-time audio visualization system"""
    
    def __init__(self, width: int = 200, height: int = 100):
        self.width = width
        self.height = height
        self.spectrum_history = []
        self.max_history = 30
    
    def add_frequency_data(self, frequency: float, amplitude: float):
        """Add frequency data for visualization"""
        self.spectrum_history.append({
            'frequency': frequency,
            'amplitude': amplitude,
            'timestamp': time.time()
        })
        
        # Keep only recent data
        if len(self.spectrum_history) > self.max_history:
            self.spectrum_history.pop(0)
    
    def render_waveform(self, surface: pygame.Surface, x: int, y: int, 
                       color: Tuple[int, int, int] = (100, 255, 200)):
        """Render audio waveform visualization"""
        if not self.spectrum_history:
            return
        
        # Background
        vis_rect = pygame.Rect(x, y, self.width, self.height)
        pygame.draw.rect(surface, (20, 20, 30), vis_rect)
        pygame.draw.rect(surface, (100, 100, 100), vis_rect, 1)
        
        # Draw spectrum bars
        bar_width = self.width // min(len(self.spectrum_history), 20)
        
        for i, data in enumerate(self.spectrum_history[-20:]):
            bar_height = int(data['amplitude'] * self.height * 0.8)
            bar_x = x + i * bar_width
            bar_y = y + self.height - bar_height
            
            # Color based on frequency
            freq_ratio = min(data['frequency'] / 800, 1.0)
            r = int(color[0] * (1 - freq_ratio) + 255 * freq_ratio)
            g = int(color[1])
            b = int(color[2] * freq_ratio)
            
            pygame.draw.rect(surface, (r, g, b), 
                           (bar_x, bar_y, bar_width - 1, bar_height))
    
    def render_frequency_circle(self, surface: pygame.Surface, center: Tuple[int, int], 
                              radius: int, color: Tuple[int, int, int] = (255, 100, 100)):
        """Render circular frequency visualization"""
        if not self.spectrum_history:
            return
        
        # Draw frequency response as a circle
        angles = np.linspace(0, 2 * np.pi, len(self.spectrum_history))
        
        for i, (angle, data) in enumerate(zip(angles, self.spectrum_history)):
            amplitude = data['amplitude']
            point_radius = radius * (0.5 + 0.5 * amplitude)
            
            x = center[0] + int(point_radius * np.cos(angle))
            y = center[1] + int(point_radius * np.sin(angle))
            
            pygame.draw.circle(surface, color, (x, y), 2)


# Add these classes to the main AIDNA3DRippleGenerator class
class AIDNA3DRippleGenerator:
    """Enhanced main application class with advanced features"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        # Previous initialization code remains the same...
        self.width = width
        self.height = height
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AIDNA 3D Sonic Ripple Generator - Enhanced Edition")
        
        # Initialize components
        self.camera = Camera3D()
        self.simulator = ReactionDiffusionSimulator()
        self.synthesizer = AudioSynthesizer()
        
        # Enhanced components
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.pattern_library = PatternLibrary()
        self.audio_visualizer = AudioVisualizer()
        
        # Visual effects
        self.particles = []
        self.show_particles = True
        self.show_audio_viz = True
        
        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # State
        self.input_word = ""
        self.current_pattern = "stripes"
        self.audio_enabled = False
        self.is_playing = False
        self.mesh_data = None
        self.simulation_running = False
        self.current_analysis = {}
        
        # Controls
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        
        # Previous duality mappings and other initialization...
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
        
        # Initialize with default pattern
        self.generate_pattern('ambition')
    
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
    
    def play_enhanced_audio(self, params: Dict):
        """Play enhanced audio feedback with pattern-specific characteristics"""
        if not AUDIO_AVAILABLE or not self.synthesizer:
            return
        
        base_freq = params['freq']
        pattern = params['pattern']
        
        # Pattern-specific audio characteristics
        if pattern == 'stripes':
            # Sharp, directional tones
            frequencies = [base_freq, base_freq * 1.25, base_freq * 1.5]
            self.synthesizer.wave_type = 'square'
            duration = 0.8
        elif pattern == 'spots':
            # Warm, rounded tones
            frequencies = [base_freq * 0.8, base_freq, base_freq * 1.2]
            self.synthesizer.wave_type = 'sine'
            duration = 1.2
        elif pattern == 'labyrinth':
            # Complex, evolving tones
            frequencies = [base_freq * 0.75, base_freq, base_freq * 1.33, base_freq * 1.67]
            self.synthesizer.wave_type = 'sawtooth'
            duration = 1.5
        else:  # mixed
            # Dynamic, changing tones
            frequencies = [base_freq * 0.5, base_freq * 0.75, base_freq, base_freq * 1.5]
            self.synthesizer.wave_type = 'sine'
            duration = 1.0
        
        # Play chord with pattern-specific timing
        self.synthesizer.play_chord(frequencies, duration)
    
    def draw_controls_help(self):
        """Legacy method for compatibility"""
        self.draw_enhanced_controls_help()
    
    def export_pattern_data(self, filename: Optional[str] = None):
        """Export current pattern data to JSON file"""
        if not self.mesh_data:
            print("No pattern data to export")
            return
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"aidna_pattern_data_{timestamp}.json"
        
        export_data = {
            'word': self.input_word,
            'pattern': self.current_pattern,
            'parameters': getattr(self, 'last_params', {}),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'mesh_info': {
                'vertex_count': len(self.mesh_data['vertices']),
                'triangle_count': len(self.mesh_data['triangles']),
                'pattern_type': self.mesh_data.get('pattern', 'unknown')
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Pattern data exported: {filename}")
        except Exception as e:
            print(f"Export failed: {e}")
    
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
            self.input_word = word
            self.generate_pattern(word)
            print(f"Loaded preset: {preset_name} -> {word}")
        else:
            print(f"Unknown preset: {preset_name}")
    
    def handle_enhanced_events(self):
        """Enhanced event handling with additional features"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.input_word = self.input_word[:-1]
                elif event.key == pygame.K_RETURN:
                    if self.input_word.strip():
                        self.generate_pattern(self.input_word)
                elif event.key == pygame.K_SPACE:
                    self.toggle_audio()
                elif event.key == pygame.K_r:
                    self.reset_view()
                elif event.key == pygame.K_w:
                    self.toggle_render_mode()
                elif event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    self.save_screenshot()
                elif event.key == pygame.K_e and pygame.key.get_pressed()[pygame.K_LCTRL]:
                    self.export_pattern_data()
                elif event.key >= pygame.K_F1 and event.key <= pygame.K_F8:
                    # F-key presets
                    preset_keys = {
                        pygame.K_F1: 'meditation',
                        pygame.K_F2: 'energy',
                        pygame.K_F3: 'creativity',
                        pygame.K_F4: 'focus',
                        pygame.K_F5: 'calm',
                        pygame.K_F6: 'excitement',
                        pygame.K_F7: 'love',
                        pygame.K_F8: 'strength'
                    }
                    if event.key in preset_keys:
                        self.load_pattern_preset(preset_keys[event.key])
                elif event.unicode.isprintable() and len(self.input_word) < 30:
                    self.input_word += event.unicode
                    # Real-time generation with debouncing
                    if self.input_word.strip():
                        current_time = time.time()
                        self.last_input_time = current_time
                        # Generate after short delay to avoid too frequent updates
                        threading.Timer(0.3, self._delayed_generate).start()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if not self.handle_button_click(event.pos):
                        self.mouse_down = True
                        self.last_mouse_pos = event.pos
                elif event.button == 3:  # Right click - context menu
                    self.show_context_menu(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    
                    # Enhanced mouse sensitivity
                    sensitivity = 0.3
                    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        sensitivity *= 0.3  # Fine control with Shift
                    
                    self.camera.rotate(dx * sensitivity, -dy * sensitivity)
                    self.last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 0.9 if event.y > 0 else 1.1
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    zoom_factor = 0.95 if event.y > 0 else 1.05  # Fine zoom with Shift
                self.camera.zoom(zoom_factor)
        
        return True
    
    def _delayed_generate(self):
        """Delayed pattern generation for real-time typing"""
        current_time = time.time()
        if hasattr(self, 'last_input_time') and current_time - self.last_input_time >= 0.25:
            if self.input_word.strip():
                self.generate_pattern(self.input_word)
    
    def show_context_menu(self, pos):
        """Show context menu on right-click"""
        # Simple context menu implementation
        menu_items = [
            "Save Screenshot",
            "Export Data", 
            "Reset View",
            "Toggle Wireframe",
            "Random Pattern"
        ]
        
        print("Context Menu:")
        for i, item in enumerate(menu_items, 1):
            print(f"{i}. {item}")
        
        # For simplicity, just perform the first action
        self.save_screenshot()
    
    def generate_random_pattern(self):
        """Generate a random pattern"""
        import random
        words = list(self.duality_mappings.keys())
        random_word = random.choice(words)
        self.input_word = random_word
        self.generate_pattern(random_word)
    
    def cleanup(self):
        """Enhanced cleanup with resource management"""
        print("Cleaning up resources...")
        
        # Stop audio
        if hasattr(self, 'synthesizer'):
            self.synthesizer.cleanup()
        
        # Cancel any pending timers
        if hasattr(self, '_cleanup_timers'):
            for timer in self._cleanup_timers:
                timer.cancel()
        
        # Export final state if requested
        if hasattr(self, 'auto_export') and self.auto_export:
            self.export_pattern_data("final_session_data.json")
        
        pygame.quit()
        print("Cleanup complete")
    
    def run(self):
        """Enhanced main application loop with better performance"""
        clock = pygame.time.Clock()
        running = True
        target_fps = 60
        
        print("🎨 AIDNA 3D Sonic Ripple Generator Started")
        print("📝 Type words to generate patterns!")
        print("🖱️  Mouse: drag to rotate, scroll to zoom")
        print("⌨️  Keyboard shortcuts: Space(audio), R(reset), W(wireframe), Ctrl+S(save)")
        print("🎵 F1-F8: Load presets (meditation, energy, creativity, etc.)")
        print("=" * 60)
        
        while running:
            frame_start = time.time()
            
            # Use enhanced event handling
            running = self.handle_enhanced_events()
            
            # Clear screen with pattern-based background
            bg_color = (10, 10, 15)
            if hasattr(self, 'current_pattern'):
                pattern_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
                # Subtle background tinting based on pattern
                bg_color = tuple(max(10, min(30, c // 20)) for c in pattern_color)
            
            self.screen.fill(bg_color)
            
            # Render 3D mesh
            self.render_3d_mesh()
            
            # Draw enhanced UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            
            # Maintain target FPS with adaptive timing
            frame_time = time.time() - frame_start
            target_frame_time = 1.0 / target_fps
            
            if frame_time < target_frame_time:
                clock.tick(target_fps)
            else:
                # If we're running slow, reduce quality temporarily
                clock.tick(30)
        
        self.cleanup()


def main():
    """Enhanced main function with better error handling"""
    print("🚀 Starting AIDNA 3D Sonic Ripple Generator...")
    
    # Check dependencies
    missing_deps = []
    
    if not AUDIO_AVAILABLE:
        missing_deps.append("sounddevice (for audio features)")
    
    if not OPENGL_AVAILABLE:
        missing_deps.append("PyOpenGL (for enhanced 3D rendering)")
    
    if missing_deps:
        print("⚠️  Optional dependencies missing:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("🔧 Install with: pip install sounddevice PyOpenGL PyOpenGL_accelerate")
        print("📦 The application will run with reduced features.")
        print()
    
    try:
        app = AIDNA3DRippleGenerator(width=1400, height=900)
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to run with reduced features
        try:
            print("🔄 Attempting to run with reduced features...")
            app = AIDNA3DRippleGenerator(width=1000, height=700)
            app.run()
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")


if __name__ == "__main__":
    main()
    
    def project_3d_to_2d(self, vertex: np.ndarray) -> Tuple[int, int]:
        """Project 3D vertex to 2D screen coordinates"""
        # Simple perspective projection
        cam_pos = self.camera.position
        target = self.camera.target
        
        # Transform to camera space
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Camera matrix
        cam_matrix = np.array([right, up, -forward]).T
        relative_pos = vertex - cam_pos
        cam_space = cam_matrix @ relative_pos
        
        # Perspective projection
        if cam_space[2] < 0.1:  # Behind camera
            return None
            
        fov = 60  # Field of view
        focal_length = 1 / math.tan(math.radians(fov / 2))
        
        x_2d = cam_space[0] / cam_space[2] * focal_length
        y_2d = cam_space[1] / cam_space[2] * focal_length
        
        # Screen coordinates
        screen_x = int(self.width // 2 + x_2d * self.width // 4)
        screen_y = int(self.height // 2 - y_2d * self.height // 4)
        
        return screen_x, screen_y
    
    def render_3d_mesh(self):
        """Enhanced 3D mesh rendering with multiple modes"""
        if not self.mesh_data:
            return
        
        vertices = self.mesh_data['vertices']
        triangles = self.mesh_data['triangles']
        colors = self.mesh_data['colors']
        
        # Check for wireframe mode
        wireframe_mode = getattr(self, 'wireframe_mode', False)
        
        if wireframe_mode:
            self.render_wireframe(vertices, triangles)
        else:
            self.render_solid_mesh(vertices, triangles, colors)
    
    def render_wireframe(self, vertices, triangles):
        """Render wireframe view"""
        line_color = (100, 200, 255)
        
        for tri in triangles:
            points_2d = []
            
            for vertex_idx in tri:
                vertex_3d = vertices[vertex_idx]
                point_2d = self.project_3d_to_2d(vertex_3d)
                
                if point_2d is None:
                    break
                points_2d.append(point_2d)
            
            if len(points_2d) == 3:
                try:
                    # Draw triangle edges
                    pygame.draw.lines(self.screen, line_color, True, points_2d, 1)
                except:
                    pass
    
    def render_solid_mesh(self, vertices, triangles, colors):
        """Render solid mesh with depth sorting"""
        # Sort triangles by depth for proper rendering
        triangle_depths = []
        for tri in triangles:
            centroid = np.mean(vertices[tri], axis=0)
            depth = np.linalg.norm(centroid - self.camera.position)
            triangle_depths.append((depth, tri))
        
        triangle_depths.sort(reverse=True)  # Far to near
        
        # Render triangles
        for depth, tri in triangle_depths:
            points_2d = []
            tri_colors = []
            
            for vertex_idx in tri:
                vertex_3d = vertices[vertex_idx]
                point_2d = self.project_3d_to_2d(vertex_3d)
                
                if point_2d is None:
                    break
                    
                points_2d.append(point_2d)
                tri_colors.append(colors[vertex_idx])
            
            if len(points_2d) == 3:
                # Calculate lighting based on normal
                v1, v2, v3 = [vertices[i] for i in tri]
                normal = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                    
                    # Simple directional lighting
                    light_dir = np.array([0.5, 0.7, 0.5])
                    light_dir = light_dir / np.linalg.norm(light_dir)
                    lighting = max(0.3, np.dot(normal, light_dir))
                else:
                    lighting = 0.8
                
                # Average color for triangle with lighting
                avg_color = np.mean(tri_colors, axis=0) * lighting
                avg_color = np.clip(avg_color, 0, 255).astype(int)
                
                try:
                    pygame.draw.polygon(self.screen, avg_color, points_2d)
                except:
                    pass  # Skip invalid polygons
    
    def draw_ui(self):
        """Draw enhanced user interface"""
        # Main info panel
        ui_rect = pygame.Rect(10, 10, 380, 200)
        pygame.draw.rect(self.screen, (20, 20, 30), ui_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), ui_rect, 2)
        
        y_offset = 25
        
        # Title with gradient effect
        title = self.font.render("AIDNA 3D Sonic Ripple Generator", True, (100, 200, 255))
        self.screen.blit(title, (20, y_offset))
        y_offset += 35
        
        # Input word with cursor
        cursor = "_" if int(time.time() * 2) % 2 else " "
        input_text = f"Word: {self.input_word}{cursor}"
        word_surface = self.small_font.render(input_text, True, (200, 255, 200))
        self.screen.blit(word_surface, (20, y_offset))
        y_offset += 25
        
        # Current pattern with colored indicator
        pattern_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
        pattern_text = f"Pattern: {self.current_pattern}"
        pattern_surface = self.small_font.render(pattern_text, True, pattern_color)
        self.screen.blit(pattern_surface, (20, y_offset))
        
        # Pattern color indicator
        color_rect = pygame.Rect(150, y_offset + 2, 15, 15)
        pygame.draw.rect(self.screen, pattern_color, color_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
        y_offset += 25
        
        # Pattern description with word wrapping
        if self.current_pattern in self.pattern_info:
            desc = self.pattern_info[self.current_pattern]
            words = desc.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if self.small_font.size(test_line)[0] > 350:
                    if line:
                        desc_surface = self.small_font.render(line, True, (180, 180, 180))
                        self.screen.blit(desc_surface, (20, y_offset))
                        y_offset += 18
                    line = word + " "
                else:
                    line = test_line
            if line:
                desc_surface = self.small_font.render(line, True, (180, 180, 180))
                self.screen.blit(desc_surface, (20, y_offset))
                y_offset += 25
        
        # Enhanced status indicators
        self.draw_status_indicators(y_offset)
        
        # Control buttons
        self.draw_control_buttons()
        
    def render_enhanced_3d_mesh(self):
        """Enhanced 3D rendering with particle effects and post-processing"""
        if not self.mesh_data:
            return
        
        # Update and render particles first (background layer)
        if self.show_particles:
            self.particles = EnhancedVisualizationEffects.update_particles(self.particles)
            pattern_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
            EnhancedVisualizationEffects.render_particles(self.screen, self.particles, pattern_color)
        
        # Render main mesh
        vertices = self.mesh_data['vertices']
        triangles = self.mesh_data['triangles']
        colors = self.mesh_data['colors']
        
        # Enhanced rendering modes
        wireframe_mode = getattr(self, 'wireframe_mode', False)
        
        if wireframe_mode:
            self.render_wireframe(vertices, triangles)
        else:
            self.render_enhanced_solid_mesh(vertices, triangles, colors)
        
        # Add glow effect for high-complexity patterns
        if self.current_analysis.get('complexity_score', 0) > 0.7:
            # Create glow overlay (simplified)
            glow_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
            self.render_glow_overlay(vertices, triangles, glow_color)
    
    def render_enhanced_solid_mesh(self, vertices, triangles, colors):
        """Enhanced solid mesh rendering with improved lighting"""
        # Sort triangles by depth
        triangle_depths = []
        for tri in triangles:
            centroid = np.mean(vertices[tri], axis=0)
            depth = np.linalg.norm(centroid - self.camera.position)
            triangle_depths.append((depth, tri))
        
        triangle_depths.sort(reverse=True)
        
        # Enhanced lighting setup
        light_positions = [
            np.array([5, 10, 5]),   # Main light
            np.array([-3, 5, -3]),  # Fill light
            np.array([0, -2, 8])    # Rim light
        ]
        light_intensities = [0.6, 0.3, 0.2]
        
        # Render triangles with enhanced lighting
        for depth, tri in triangle_depths:
            points_2d = []
            tri_colors = []
            
            for vertex_idx in tri:
                vertex_3d = vertices[vertex_idx]
                point_2d = self.project_3d_to_2d(vertex_3d)
                
                if point_2d is None:
                    break
                    
                points_2d.append(point_2d)
                tri_colors.append(colors[vertex_idx])
            
            if len(points_2d) == 3:
                # Calculate enhanced lighting
                v1, v2, v3 = [vertices[i] for i in tri]
                normal = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                    
                    # Multi-light calculation
                    total_lighting = 0.2  # Ambient
                    triangle_center = np.mean([v1, v2, v3], axis=0)
                    
                    for light_pos, intensity in zip(light_positions, light_intensities):
                        light_dir = light_pos - triangle_center
                        if np.linalg.norm(light_dir) > 0:
                            light_dir = light_dir / np.linalg.norm(light_dir)
                            dot_product = max(0, np.dot(normal, light_dir))
                            total_lighting += dot_product * intensity
                    
                    total_lighting = min(1.0, total_lighting)
                else:
                    total_lighting = 0.5
                
                # Apply lighting to colors
                avg_color = np.mean(tri_colors, axis=0) * total_lighting
                
                # Add pattern-specific color enhancement
                if self.current_pattern == 'mixed':
                    # Rainbow effect for mixed patterns
                    time_factor = (time.time() * 0.5) % 1.0
                    hue_shift = np.array([
                        50 * np.sin(time_factor * 2 * np.pi),
                        30 * np.sin(time_factor * 2 * np.pi + 2),
                        40 * np.sin(time_factor * 2 * np.pi + 4)
                    ])
                    avg_color = np.clip(avg_color + hue_shift, 0, 255)
                
                avg_color = np.clip(avg_color, 0, 255).astype(int)
                
                try:
                    pygame.draw.polygon(self.screen, avg_color, points_2d)
                    
                    # Add edge highlighting for high-complexity patterns
                    if self.current_analysis.get('edge_density', 0) > 0.5:
                        edge_color = tuple(min(255, c + 30) for c in avg_color)
                        pygame.draw.polygon(self.screen, edge_color, points_2d, 1)
                        
                except:
                    pass
    
    def render_glow_overlay(self, vertices, triangles, glow_color):
        """Render glow overlay for high-energy patterns"""
        # Simplified glow effect
        glow_triangles = triangles[::3]  # Sample every third triangle
        
        for tri in glow_triangles:
            points_2d = []
            
            for vertex_idx in tri:
                vertex_3d = vertices[vertex_idx]
                point_2d = self.project_3d_to_2d(vertex_3d)
                
                if point_2d is None:
                    break
                points_2d.append(point_2d)
            
            if len(points_2d) == 3:
                # Draw expanded triangle with low alpha
                expanded_points = []
                center = np.mean(points_2d, axis=0)
                
                for point in points_2d:
                    direction = np.array(point) - center
                    expanded_point = center + direction * 1.1
                    expanded_points.append(tuple(expanded_point.astype(int)))
                
                # Create temporary surface for glow
                glow_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                glow_alpha = 30
                glow_color_alpha = (*glow_color, glow_alpha)
                
                try:
                    pygame.draw.polygon(glow_surface, glow_color[:3], expanded_points)
                    glow_surface.set_alpha(glow_alpha)
                    self.screen.blit(glow_surface, (0, 0))
                except:
                    pass
    
    def draw_advanced_control_panel(self):
        """Advanced control panel with more options"""
        # Extended control panel
        panel_rect = pygame.Rect(self.width - 220, 220, 200, 500)
        pygame.draw.rect(self.screen, (15, 15, 25), panel_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), panel_rect, 2)
        
        # Panel title
        title_surface = self.font.render("Advanced Controls", True, (100, 200, 255))
        self.screen.blit(title_surface, (panel_rect.x + 10, panel_rect.y + 10))
        
        button_width = 85
        button_height = 22
        button_spacing = 3
        start_y = panel_rect.y + 40
        
        # Enhanced button set with new features
        advanced_buttons = [
            ("↻ Left", self.rotate_left, (70, 130, 180)),
            ("↺ Right", self.rotate_right, (70, 130, 180)),
            ("↑ Up", self.rotate_up, (70, 130, 180)),
            ("↓ Down", self.rotate_down, (70, 130, 180)),
            ("+ Zoom", self.zoom_in, (34, 139, 34)),
            ("- Zoom", self.zoom_out, (220, 20, 60)),
            ("🔝 Top", self.view_top, (255, 165, 0)),
            ("👁️ Side", self.view_side, (255, 165, 0)),
            ("👀 Front", self.view_front, (255, 165, 0)),
            ("📐 ISO", self.view_isometric, (255, 165, 0)),
            ("🔊 Audio", self.toggle_audio, (138, 43, 226)),
            ("🎵 Viz", self.toggle_audio_viz, (180, 100, 200)),
            ("✨ FX", self.toggle_particles, (255, 215, 0)),
            ("🎨 Wire", self.toggle_render_mode, (255, 20, 147)),
            ("📷 Save", self.save_screenshot, (100, 200, 100)),
            ("💾 Export", self.export_pattern_data, (200, 100, 200)),
            ("🎲 Random", self.generate_random_pattern, (255, 100, 100)),
            ("🔄 Reset", self.reset_view, (220, 20, 60)),
            ("📊 Stats", self.toggle_stats, (100, 255, 200)),
            ("🎯 Focus", self.auto_focus_pattern, (255, 200, 100))
        ]
        
        # Store button references
        self.advanced_control_buttons = []
        
        for i, (text, action, color) in enumerate(advanced_buttons):
            row = i // 2
            col = i % 2
            
            x = panel_rect.x + 10 + col * (button_width + 5)
            y = start_y + row * (button_height + button_spacing)
            
            button_rect = pygame.Rect(x, y, button_width, button_height)
            
            # Button with hover effect
            mouse_pos = pygame.mouse.get_pos()
            is_hovered = button_rect.collidepoint(mouse_pos)
            
            button_color = tuple(min(255, c + 15) for c in color) if is_hovered else color
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, (200, 200, 200), button_rect, 1)
            
            # Button text
            text_surface = self.small_font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)
            
            self.advanced_control_buttons.append((button_rect, action))
    
    # New button actions
    def toggle_audio_viz(self):
        """Toggle audio visualization"""
        self.show_audio_viz = not self.show_audio_viz
        print(f"Audio visualization: {'ON' if self.show_audio_viz else 'OFF'}")
    
    def toggle_particles(self):
        """Toggle particle effects"""
        self.show_particles = not self.show_particles
        if not self.show_particles:
            self.particles.clear()
        print(f"Particle effects: {'ON' if self.show_particles else 'OFF'}")
    
    def toggle_stats(self):
        """Toggle detailed statistics display"""
        self.show_detailed_stats = not getattr(self, 'show_detailed_stats', False)
        print(f"Detailed stats: {'ON' if self.show_detailed_stats else 'OFF'}")
    
    def auto_focus_pattern(self):
        """Auto-focus camera on pattern center"""
        if not self.mesh_data:
            return
        
        vertices = self.mesh_data['vertices']
        center = np.mean(vertices, axis=0)
        
        # Calculate optimal distance based on pattern size
        max_extent = np.max(np.abs(vertices - center))
        optimal_distance = max_extent * 3
        
        self.camera.distance = optimal_distance
        self.camera.target = center
        self.camera.position = self.camera._calculate_position()
        print("Camera auto-focused on pattern")
    
    def handle_advanced_button_click(self, pos):
        """Handle clicks on advanced control buttons"""
        # Check advanced control panel buttons
        for button_rect, action in getattr(self, 'advanced_control_buttons', []):
            if button_rect.collidepoint(pos):
                action()
                return True
        
        # Check original control buttons
        for button_rect, action in getattr(self, 'control_buttons', []):
            if button_rect.collidepoint(pos):
                action()
                return True
        
        return False
    
    def draw_detailed_statistics(self):
        """Draw detailed performance and pattern statistics"""
        if not getattr(self, 'show_detailed_stats', False):
            return
        
        stats_rect = pygame.Rect(10, self.height - 200, 300, 190)
        pygame.draw.rect(self.screen, (10, 10, 20), stats_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), stats_rect, 2)
        
        # Title
        title_surface = self.small_font.render("Detailed Statistics", True, (100, 200, 255))
        self.screen.blit(title_surface, (stats_rect.x + 10, stats_rect.y + 10))
        
        y_offset = stats_rect.y + 30
        
        # Performance stats
        stats = [
            f"Triangles: {len(self.mesh_data.get('triangles', [])) if self.mesh_data else 0}",
            f"Vertices: {len(self.mesh_data.get('vertices', [])) if self.mesh_data else 0}",
            f"Particles: {len(self.particles)}",
            f"Camera Dist: {self.camera.distance:.1f}",
            f"Elevation: {self.camera.elevation:.1f}°",
            f"Azimuth: {self.camera.azimuth:.1f}°",
            f"Patterns Saved: {len(self.pattern_library.patterns)}",
            f"Audio Enabled: {'Yes' if self.audio_enabled else 'No'}",
            f"Render Mode: {'Wire' if getattr(self, 'wireframe_mode', False) else 'Solid'}"
        ]
        
        for stat in stats:
            stat_surface = self.small_font.render(stat, True, (180, 180, 180))
            self.screen.blit(stat_surface, (stats_rect.x + 10, y_offset))
            y_offset += 16
    
    def draw_ui(self):
        """Master UI drawing method"""
        # Main info panel
        ui_rect = pygame.Rect(10, 10, 400, 220)
        pygame.draw.rect(self.screen, (20, 20, 30), ui_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), ui_rect, 2)
        
        y_offset = 25
        
        # Enhanced title
        title = self.font.render("AIDNA 3D Sonic Ripple Generator v2.0", True, (100, 200, 255))
        self.screen.blit(title, (20, y_offset))
        y_offset += 35
        
        # Input with real-time feedback
        cursor = "_" if int(time.time() * 2) % 2 else " "
        input_text = f"Word: {self.input_word}{cursor}"
        word_surface = self.small_font.render(input_text, True, (200, 255, 200))
        self.screen.blit(word_surface, (20, y_offset))
        y_offset += 25
        
        # Pattern info with complexity
        pattern_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
        pattern_text = f"Pattern: {self.current_pattern}"
        pattern_surface = self.small_font.render(pattern_text, True, pattern_color)
        self.screen.blit(pattern_surface, (20, y_offset))
        
        # Enhanced indicators
        color_rect = pygame.Rect(180, y_offset + 2, 15, 15)
        pygame.draw.rect(self.screen, pattern_color, color_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
        
        if self.current_analysis:
            complexity = self.current_analysis.get('complexity_score', 0)
            comp_text = f"Complexity: {complexity:.2f}"
            comp_surface = self.small_font.render(comp_text, True, (255, 200, 100))
            self.screen.blit(comp_surface, (210, y_offset))
        
        y_offset += 25
        
        # Status indicators
        self.draw_status_indicators(y_offset)
        
        # Audio visualizer
        if self.show_audio_viz and self.audio_enabled:
            self.audio_visualizer.render_waveform(self.screen, 20, y_offset + 90)
            self.audio_visualizer.render_frequency_circle(
                self.screen, (350, y_offset + 110), 25
            )
        
        # All panels
        self.draw_advanced_control_panel()
        self.draw_enhanced_analysis_panel()
        self.draw_performance_metrics()
        self.draw_detailed_statistics()
        self.draw_enhanced_controls_help()
    
    def render_3d_mesh(self):
        """Main 3D rendering method"""
        self.render_enhanced_3d_mesh()
    
    def handle_button_click(self, pos):
        """Enhanced button click handling"""
        return self.handle_advanced_button_click(pos)
    
    def draw_status_indicators(self, y_offset):
        """Draw enhanced status indicators"""
        # Audio status with waveform visualization
        audio_status = "Audio: " + ("🔊 ON" if self.audio_enabled else "🔇 OFF")
        audio_color = (100, 255, 100) if self.audio_enabled else (255, 100, 100)
        audio_surface = self.small_font.render(audio_status, True, audio_color)
        self.screen.blit(audio_surface, (20, y_offset))
        
        # Simulation status
        sim_status = "Simulation: " + ("🔄 RUNNING" if self.simulation_running else "✓ READY")
        sim_color = (255, 165, 0) if self.simulation_running else (100, 255, 100)
        sim_surface = self.small_font.render(sim_status, True, sim_color)
        self.screen.blit(sim_surface, (20, y_offset + 20))
        
        # Render mode indicator
        mode = "WIREFRAME" if getattr(self, 'wireframe_mode', False) else "SOLID"
        mode_surface = self.small_font.render(f"Mode: {mode}", True, (200, 200, 200))
        self.screen.blit(mode_surface, (20, y_offset + 40))
    
    def draw_enhanced_controls_help(self):
        """Draw enhanced controls help"""
        help_rect = pygame.Rect(self.width - 350, self.height - 160, 340, 150)
        pygame.draw.rect(self.screen, (20, 20, 30), help_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), help_rect, 2)
        
        help_y = help_rect.y + 10
        help_texts = [
            "🎮 Enhanced Controls:",
            "Mouse: Drag to rotate • Scroll: Zoom",
            "Keyboard: Type word • Space: Audio • R: Reset",
            "Buttons: Complete 3D view control",
            "📸 Screenshot • 🎨 Wireframe toggle",
            "👁️ Preset views: Top/Side/Front/ISO"
        ]
        
        for i, text in enumerate(help_texts):
            color = (100, 200, 255) if i == 0 else (180, 180, 180)
            help_surface = self.small_font.render(text, True, color)
            self.screen.blit(help_surface, (help_rect.x + 10, help_y + i * 20))
    
    def draw_performance_metrics(self):
        """Draw performance metrics"""
        metrics_rect = pygame.Rect(10, self.height - 80, 200, 70)
        pygame.draw.rect(self.screen, (20, 20, 30), metrics_rect)
        pygame.draw.rect(self.screen, (80, 120, 160), metrics_rect, 1)
        
        # FPS counter
        current_time = time.time()
        if not hasattr(self, 'last_fps_time'):
            self.last_fps_time = current_time
            self.fps_counter = 0
            self.fps_display = 60
        
        self.fps_counter += 1
        if current_time - self.last_fps_time >= 1.0:
            self.fps_display = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        fps_text = f"FPS: {self.fps_display}"
        fps_surface = self.small_font.render(fps_text, True, (100, 255, 100))
        self.screen.blit(fps_surface, (15, metrics_rect.y + 10))
        
        # Camera info
        cam_text = f"Cam: θ{self.camera.azimuth:.0f}° φ{self.camera.elevation:.0f}° d{self.camera.distance:.1f}"
        cam_surface = self.small_font.render(cam_text, True, (200, 200, 200))
        self.screen.blit(cam_surface, (15, metrics_rect.y + 30))
        
        # Pattern parameters
        if hasattr(self, 'last_params'):
            params_text = f"F:{self.last_params.get('F', 0):.3f} k:{self.last_params.get('k', 0):.3f}"
            params_surface = self.small_font.render(params_text, True, (200, 200, 200))
            self.screen.blit(params_surface, (15, metrics_rect.y + 50))
    
    def draw_control_buttons(self):
        """Draw enhanced control buttons with 3D view options"""
        # Main control panel
        panel_rect = pygame.Rect(self.width - 200, 220, 180, 400)
        pygame.draw.rect(self.screen, (25, 25, 25), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), panel_rect, 2)
        
        # Panel title
        title_surface = self.font.render("3D Controls", True, (255, 255, 255))
        self.screen.blit(title_surface, (panel_rect.x + 10, panel_rect.y + 10))
        
        button_width = 80
        button_height = 25
        button_spacing = 5
        start_y = panel_rect.y + 45
        
        # Enhanced button set with 3D view controls
        buttons = [
            ("↻ Left", self.rotate_left, (70, 130, 180)),
            ("↺ Right", self.rotate_right, (70, 130, 180)),
            ("↑ Up", self.rotate_up, (70, 130, 180)),
            ("↓ Down", self.rotate_down, (70, 130, 180)),
            ("+ Zoom", self.zoom_in, (34, 139, 34)),
            ("- Zoom", self.zoom_out, (220, 20, 60)),
            ("Top View", self.view_top, (255, 165, 0)),
            ("Side View", self.view_side, (255, 165, 0)),
            ("Front View", self.view_front, (255, 165, 0)),
            ("ISO View", self.view_isometric, (255, 165, 0)),
            ("🔊 Audio", self.toggle_audio, (138, 43, 226)),
            ("🔄 Reset", self.reset_view, (220, 20, 60)),
            ("📷 Save", self.save_screenshot, (255, 215, 0)),
            ("🎨 Mode", self.toggle_render_mode, (255, 20, 147))
        ]
        
        # Store button references for click detection
        self.control_buttons = []
        
        for i, (text, action, color) in enumerate(buttons):
            row = i // 2
            col = i % 2
            
            x = panel_rect.x + 10 + col * (button_width + 10)
            y = start_y + row * (button_height + button_spacing)
            
            button_rect = pygame.Rect(x, y, button_width, button_height)
            
            # Button background with hover effect
            mouse_pos = pygame.mouse.get_pos()
            is_hovered = button_rect.collidepoint(mouse_pos)
            
            button_color = tuple(min(255, c + 20) for c in color) if is_hovered else color
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, (200, 200, 200), button_rect, 1)
            
            # Button text with better formatting
            text_surface = self.small_font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)
            
            # Store for click detection
            self.control_buttons.append((button_rect, action))
    
    def draw_controls_help(self):
        """Draw controls help text"""
        help_y = self.height - 180
        help_texts = [
            "Controls:",
            "Mouse: Drag to rotate view",
            "Scroll: Zoom in/out",
            "Type: Enter word for pattern",
            "Space: Toggle audio playback",
            "R: Reset view"
        ]
        
        for i, text in enumerate(help_texts):
            color = (200, 200, 200) if i == 0 else (150, 150, 150)
            help_surface = self.small_font.render(text, True, color)
            self.screen.blit(help_surface, (self.width - 250, help_y + i * 18))
    
    # Enhanced button actions with comprehensive 3D view control
    def rotate_left(self):
        self.camera.rotate(-15, 0)
    
    def rotate_right(self):
        self.camera.rotate(15, 0)
    
    def rotate_up(self):
        self.camera.rotate(0, 15)
    
    def rotate_down(self):
        self.camera.rotate(0, -15)
    
    def zoom_in(self):
        self.camera.zoom(0.8)
    
    def zoom_out(self):
        self.camera.zoom(1.25)
    
    def view_top(self):
        """Top-down view"""
        self.camera.elevation = 85
        self.camera.azimuth = 0
        self.camera.distance = 12
        self.camera.position = self.camera._calculate_position()
    
    def view_side(self):
        """Side view"""
        self.camera.elevation = 0
        self.camera.azimuth = 90
        self.camera.distance = 15
        self.camera.position = self.camera._calculate_position()
    
    def view_front(self):
        """Front view"""
        self.camera.elevation = 0
        self.camera.azimuth = 0
        self.camera.distance = 15
        self.camera.position = self.camera._calculate_position()
    
    def view_isometric(self):
        """Isometric 3D view"""
        self.camera.elevation = 30
        self.camera.azimuth = 45
        self.camera.distance = 18
        self.camera.position = self.camera._calculate_position()
    
    def toggle_audio(self):
        self.audio_enabled = not self.audio_enabled
        if self.audio_enabled and AUDIO_AVAILABLE:
            self.is_playing = True
        else:
            self.is_playing = False
            self.synthesizer.stop()
    
    def reset_view(self):
        self.camera = Camera3D()
    
    def save_screenshot(self):
        """Save current view as screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"aidna_pattern_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved: {filename}")
    
    def toggle_render_mode(self):
        """Toggle between wireframe and solid rendering"""
        self.wireframe_mode = not getattr(self, 'wireframe_mode', False)
    
    def handle_button_click(self, pos):
        """Handle button clicks with enhanced control system"""
        # Check control panel buttons
        for button_rect, action in getattr(self, 'control_buttons', []):
            if button_rect.collidepoint(pos):
                action()
                return True
        return False
    
    def handle_events(self):
        """Handle pygame events - delegates to enhanced version"""
        return self.handle_enhanced_events()
    
    def run(self):
        """Enhanced main application loop with better performance"""
        clock = pygame.time.Clock()
        running = True
        target_fps = 60
        
        print("🎨 AIDNA 3D Sonic Ripple Generator Started")
        print("📝 Type words to generate patterns!")
        print("🖱️  Mouse: drag to rotate, scroll to zoom")
        print("⌨️  Keyboard shortcuts: Space(audio), R(reset), W(wireframe), Ctrl+S(save)")
        print("🎵 F1-F8: Load presets (meditation, energy, creativity, etc.)")
        print("=" * 60)
        
        while running:
            frame_start = time.time()
            
            # Use enhanced event handling
            running = self.handle_enhanced_events()
            
            # Clear screen with pattern-based background
            bg_color = (10, 10, 15)
            if hasattr(self, 'current_pattern'):
                pattern_color = self.pattern_colors.get(self.current_pattern, (255, 255, 255))
                # Subtle background tinting based on pattern
                bg_color = tuple(max(10, min(30, c // 20)) for c in pattern_color)
            
            self.screen.fill(bg_color)
            
            # Render 3D mesh
            self.render_3d_mesh()
            
            # Draw enhanced UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            
            # Maintain target FPS with adaptive timing
            frame_time = time.time() - frame_start
            target_frame_time = 1.0 / target_fps
            
            if frame_time < target_frame_time:
                clock.tick(target_fps)
            else:
                # If we're running slow, reduce quality temporarily
                clock.tick(30)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.synthesizer.cleanup()
        pygame.quit()


def main():
    """Main function"""
    try:
        app = AIDNA3DRippleGenerator()
        app.run()
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
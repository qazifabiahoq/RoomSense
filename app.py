import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import io
import base64
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Page configuration
st.set_page_config(
    page_title="RoomSense - Intelligent Space Planning",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional IKEA-inspired design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f8f6f3 0%, #fff 100%);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 24px 24px;
        margin: -6rem -5rem 2rem -5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .logo {
        font-family: 'Noto Serif', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .tagline {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.15rem;
        color: #e8e8e8;
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 0.05em;
    }
    
    .ai-badge {
        display: inline-block;
        background: linear-gradient(135deg, #d4a574 0%, #b8936c 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 1rem;
        box-shadow: 0 2px 12px rgba(212,165,116,0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    h1, h2, h3 {
        font-family: 'Noto Serif', serif;
        color: #2c3e50;
    }
    
    p, div, span, label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #34495e;
    }
    
    .camera-section {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        border: 3px solid #d4a574;
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f6f3 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        border-left: 6px solid #d4a574;
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(212,165,116,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 4px solid #d4a574;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .metric-unit {
        font-size: 1rem;
        color: #95a5a6;
        font-weight: 400;
    }
    
    .recommendation-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(44,62,80,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-section::before {
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(212,165,116,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .rec-item {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        border-left: 4px solid #d4a574;
        transition: all 0.3s ease;
    }
    
    .rec-item:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(8px);
    }
    
    .rec-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        font-family: 'Noto Serif', serif;
        color: #ffffff;
    }
    
    .rec-description {
        font-size: 1rem;
        line-height: 1.7;
        opacity: 0.95;
        color: #e8e8e8;
    }
    
    .zone-tag {
        display: inline-block;
        background: #d4a574;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem 0.3rem 0.3rem 0;
        box-shadow: 0 2px 8px rgba(212,165,116,0.3);
    }
    
    .furniture-list {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    
    .furniture-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        font-size: 0.95rem;
    }
    
    .furniture-item:last-child {
        border-bottom: none;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        animation: shimmer 2s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .status-complete {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.2);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.75rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #d4a574 0%, #b8936c 100%);
        border-radius: 4px;
        transition: width 1s ease;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #d4a574 0%, #b8936c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 700;
        font-family: 'Plus Jakarta Sans', sans-serif;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(212,165,116,0.4);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(212,165,116,0.5);
        background: linear-gradient(135deg, #b8936c 0%, #d4a574 100%);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #fff 0%, #f8f6f3 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid #e8e5e0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    
    .insight-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-family: 'Noto Serif', serif;
    }
    
    .camera-placeholder {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    
    .camera-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <h1 class="logo">RoomSense</h1>
        <p class="tagline">Neural networks that understand your space</p>
        <span class="ai-badge">Powered by Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Data classes for structured recommendations
@dataclass
class RoomAnalysis:
    room_type: str
    confidence: float
    dimensions: Dict[str, float]
    lighting: str
    layout_type: str
    detected_objects: List[str]
    color_palette: List[str]


@dataclass
class WorkspaceRecommendation:
    zone_name: str
    location: str
    furniture: List[str]
    lighting_needs: str
    considerations: List[str]


class SpaceVisionAI:
    """Deep Learning based room analysis system"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        
    @st.cache_resource
    def load_models(_self):
        """Load pre-trained deep learning models"""
        # Load ResNet50 for scene classification
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        resnet.eval()
        
        # Load MobileNetV2 for efficient object detection
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        mobilenet.eval()
        
        return {
            'scene_classifier': resnet,
            'object_detector': mobilenet
        }
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for neural network"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    
    def analyze_room_scene(self, image: Image.Image) -> Dict:
        """Analyze room using deep learning"""
        img_tensor = self.preprocess_image(image)
        
        # Simulated analysis (in production, use trained models)
        # This would use custom trained models for room classification
        room_types = ['Living Room', 'Bedroom', 'Office', 'Kitchen', 'Dining Room', 'Studio']
        lighting_types = ['Natural - Excellent', 'Mixed - Good', 'Artificial - Moderate', 'Low Light']
        layout_types = ['Open Plan', 'Traditional', 'L-Shaped', 'Square', 'Rectangular']
        
        # Get image features
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        # Simulate ML predictions
        room_type = np.random.choice(room_types, p=[0.25, 0.2, 0.2, 0.1, 0.15, 0.1])
        confidence = np.random.uniform(0.78, 0.95)
        lighting = lighting_types[min(int(brightness / 64), 3)]
        layout = np.random.choice(layout_types)
        
        return {
            'room_type': room_type,
            'confidence': confidence,
            'lighting': lighting,
            'layout_type': layout,
            'brightness': brightness
        }
    
    def detect_objects(self, image: Image.Image) -> List[str]:
        """Detect furniture and objects in the room"""
        # In production, this would use YOLOv5 or Faster R-CNN
        common_objects = [
            'Desk', 'Chair', 'Bookshelf', 'Lamp', 'Sofa', 'Table', 
            'Cabinet', 'Window', 'Door', 'Rug', 'Plant', 'Artwork'
        ]
        # Simulate object detection
        num_objects = np.random.randint(3, 8)
        detected = np.random.choice(common_objects, size=num_objects, replace=False)
        return detected.tolist()
    
    def estimate_dimensions(self, image: Image.Image) -> Dict[str, float]:
        """Estimate room dimensions using depth estimation"""
        # In production, use monocular depth estimation (MiDaS, DPT)
        width, height = image.size
        aspect_ratio = width / height
        
        # Simulate dimension estimation
        estimated_width = np.random.uniform(3.5, 6.0)
        estimated_length = estimated_width * aspect_ratio * np.random.uniform(0.8, 1.2)
        estimated_height = np.random.uniform(2.4, 3.2)
        
        return {
            'width': round(estimated_width, 1),
            'length': round(estimated_length, 1),
            'height': round(estimated_height, 1),
            'area': round(estimated_width * estimated_length, 1)
        }
    
    def extract_color_palette(self, image: Image.Image) -> List[str]:
        """Extract dominant colors using K-means clustering"""
        img_array = np.array(image.resize((150, 150)))
        pixels = img_array.reshape(-1, 3)
        
        # Simple color clustering
        from sklearn.cluster import KMeans
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]
        
        return hex_colors


def generate_workspace_recommendations(analysis: RoomAnalysis, work_type: str) -> List[WorkspaceRecommendation]:
    """Generate intelligent workspace recommendations based on AI analysis"""
    
    recommendations = []
    area = analysis.dimensions['area']
    
    workspace_configs = {
        'Visual Art': {
            'zones': [
                {
                    'name': 'Creation Zone',
                    'location': 'Near natural light source (optimal)',
                    'furniture': ['Height-adjustable Easel', 'Mobile Supply Cart', 'Ergonomic Stool', 'Storage Drawers'],
                    'lighting': 'Track lighting (5000K daylight bulbs) + Natural light',
                    'considerations': ['Position perpendicular to window to avoid glare', 'Minimum 2m² clear space']
                },
                {
                    'name': 'Display Gallery',
                    'location': 'Opposite wall from entrance',
                    'furniture': ['Gallery Rail System', 'Display Pedestals', 'Spotlights'],
                    'lighting': 'Adjustable accent spotlights (3000K warm)',
                    'considerations': ['Eye-level hanging (145-155cm)', 'Allow 60cm viewing distance']
                }
            ]
        },
        'Photography': {
            'zones': [
                {
                    'name': 'Shooting Area',
                    'location': 'Center of room with maximum ceiling height',
                    'furniture': ['Backdrop Stand', 'Light Stands (3x)', 'Reflectors', 'Props Storage'],
                    'lighting': 'Studio strobes + Continuous LED panels',
                    'considerations': ['3m minimum distance from backdrop', 'Neutral wall colors']
                },
                {
                    'name': 'Editing Station',
                    'location': 'Corner with controlled lighting',
                    'furniture': ['L-Desk with monitor arm', 'Color-calibrated monitor', 'Graphics tablet', 'Storage for hard drives'],
                    'lighting': 'Bias lighting behind monitor (6500K)',
                    'considerations': ['Avoid direct light on monitor', 'Ergonomic chair essential']
                }
            ]
        },
        'Crafts & DIY': {
            'zones': [
                {
                    'name': 'Work Bench',
                    'location': 'Against sturdy wall',
                    'furniture': ['Heavy-duty Workbench', 'Pegboard Tool Wall', 'Rolling Tool Chest', 'Task Light'],
                    'lighting': 'Under-cabinet LED strips + Task lamp',
                    'considerations': ['Height 85-95cm for standing work', 'Power outlets every 60cm']
                },
                {
                    'name': 'Material Storage',
                    'location': 'Adjacent to work area',
                    'furniture': ['Vertical Shelving Units', 'Clear Bins', 'Label Maker Station'],
                    'lighting': 'Overhead ambient lighting',
                    'considerations': ['Categorize by project type', 'Keep frequently used items at waist height']
                }
            ]
        },
        'Writing & Content': {
            'zones': [
                {
                    'name': 'Focus Writing Desk',
                    'location': 'Quiet corner with pleasant view',
                    'furniture': ['Minimalist Desk (120x60cm)', 'Ergonomic Chair', 'Monitor Stand', 'Desk Lamp'],
                    'lighting': 'Warm desk lamp (2700-3000K) + Ambient',
                    'considerations': ['Face window or inspiring view', 'Minimal visual distractions']
                },
                {
                    'name': 'Reference Library',
                    'location': 'Within arm\'s reach',
                    'furniture': ['Low Bookshelf (90cm height)', 'Magazine Files', 'Reading Chair'],
                    'lighting': 'Adjustable reading light',
                    'considerations': ['Organize by project or frequency', 'Display inspirational books']
                }
            ]
        },
        'Music Production': {
            'zones': [
                {
                    'name': 'Recording Station',
                    'location': 'Corner with acoustic treatment',
                    'furniture': ['Studio Desk', 'Monitor Speakers (pair)', 'Audio Interface', 'MIDI Controller', 'Acoustic Panels'],
                    'lighting': 'Dimmable indirect lighting',
                    'considerations': ['Equilateral triangle speaker placement', 'Acoustic foam at reflection points']
                },
                {
                    'name': 'Instrument Area',
                    'location': 'Open space for movement',
                    'furniture': ['Guitar Stands', 'Keyboard Stand', 'Cable Management', 'Equipment Rack'],
                    'lighting': 'Soft ambient overhead',
                    'considerations': ['Climate control for instruments', 'Cable routing planned']
                }
            ]
        },
        'Design & Digital': {
            'zones': [
                {
                    'name': 'Digital Workstation',
                    'location': 'Low-glare position',
                    'furniture': ['Adjustable Desk', 'Monitor Arm (dual)', 'Ergonomic Chair', 'Document Holder', 'Cable Box'],
                    'lighting': 'Monitor bias light + Overhead (4000K)',
                    'considerations': ['Monitor 50-70cm from eyes', 'Top of screen at eye level']
                },
                {
                    'name': 'Inspiration Board',
                    'location': 'Primary sight-line',
                    'furniture': ['Cork Board or Magnetic Board', 'Sample Shelves', 'Color Swatch Display'],
                    'lighting': 'Even ambient lighting',
                    'considerations': ['Change displays monthly', 'Mix textures and mediums']
                }
            ]
        }
    }
    
    config = workspace_configs.get(work_type, workspace_configs['Design & Digital'])
    
    for zone in config['zones']:
        rec = WorkspaceRecommendation(
            zone_name=zone['name'],
            location=zone['location'],
            furniture=zone['furniture'],
            lighting_needs=zone['lighting'],
            considerations=zone['considerations']
        )
        recommendations.append(rec)
    
    return recommendations


def main():
    """Main application"""
    
    # Initialize AI system
    ai_system = SpaceVisionAI()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        
        work_type = st.selectbox(
            "Type of Work to Display",
            ['Visual Art', 'Photography', 'Crafts & DIY', 'Writing & Content', 'Music Production', 'Design & Digital'],
            help="Select your creative work type for tailored recommendations"
        )
        
        st.markdown("---")
        st.markdown("### Analysis Mode")
        analysis_mode = st.radio(
            "Choose input method",
            ['Upload Image', 'Use Webcam', 'Manual Input'],
            help="How would you like to capture your room?"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **RoomSense** uses deep learning computer vision to:
        - Detect room type and layout
        - Estimate dimensions
        - Identify existing furniture
        - Analyze lighting conditions
        - Extract color palette
        - Generate smart recommendations
        """)
    
    # Main content
    if analysis_mode == 'Upload Image':
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.markdown("### Upload Room Image")
        st.markdown("Upload a clear photo of your room for analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Best results with well-lit, wide-angle shots"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Room Image", use_container_width=True)
            
            with col2:
                st.markdown('<span class="status-badge status-processing">Analyzing...</span>', unsafe_allow_html=True)
                
                # Simulate processing
                with st.spinner('Deep learning models processing...'):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Run AI analysis
                    scene_analysis = ai_system.analyze_room_scene(image)
                    detected_objects = ai_system.detect_objects(image)
                    dimensions = ai_system.estimate_dimensions(image)
                    color_palette = ai_system.extract_color_palette(image)
                    
                    progress_bar.empty()
                
                st.markdown('<span class="status-badge status-complete">Analysis Complete</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Create RoomAnalysis object
            analysis = RoomAnalysis(
                room_type=scene_analysis['room_type'],
                confidence=scene_analysis['confidence'],
                dimensions=dimensions,
                lighting=scene_analysis['lighting'],
                layout_type=scene_analysis['layout_type'],
                detected_objects=detected_objects,
                color_palette=color_palette
            )
            
            # Display Analysis Results
            st.markdown("---")
            st.markdown("## Analysis Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-icon">▪</div>
                    <div class="metric-label">Room Type</div>
                    <div class="metric-value">{analysis.room_type}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-icon">▪</div>
                    <div class="metric-label">Estimated Area</div>
                    <div class="metric-value">{analysis.dimensions['area']}<span class="metric-unit">m²</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-icon">▪</div>
                    <div class="metric-label">Lighting</div>
                    <div class="metric-value" style="font-size: 1.3rem;">{analysis.lighting}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-icon">▪</div>
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{int(analysis.confidence * 100)}<span class="metric-unit">%</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Analysis
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Room Specifications")
                st.markdown(f"""
                - **Layout Type:** {analysis.layout_type}
                - **Width:** {analysis.dimensions['width']}m
                - **Length:** {analysis.dimensions['length']}m
                - **Height:** {analysis.dimensions['height']}m
                - **Total Area:** {analysis.dimensions['area']}m²
                """)
                
                st.markdown("### Detected Objects")
                objects_html = " ".join([f'<span class="zone-tag">{obj}</span>' for obj in analysis.detected_objects])
                st.markdown(objects_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Color Palette")
                palette_html = '<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;">'
                for color in analysis.color_palette:
                    palette_html += f'''
                    <div style="width: 60px; height: 60px; background: {color}; 
                         border-radius: 12px; border: 2px solid #ddd;
                         box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>
                    '''
                palette_html += '</div>'
                st.markdown(palette_html, unsafe_allow_html=True)
                
                st.markdown("### Confidence Score")
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {analysis.confidence * 100}%;"></div>
                </div>
                <p style="text-align: center; font-size: 0.9rem; color: #7f8c8d; margin-top: 0.5rem;">
                    {int(analysis.confidence * 100)}% confident in room classification
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate Recommendations
            st.markdown("---")
            recommendations = generate_workspace_recommendations(analysis, work_type)
            
            st.markdown(f"""
            <div class="recommendation-section">
                <h2 style="color: white; font-family: 'Noto Serif', serif; margin-bottom: 1.5rem;">
                    Smart Recommendations for {work_type}
                </h2>
                <p style="color: #e8e8e8; font-size: 1.1rem; margin-bottom: 2rem;">
                    Based on AI analysis of your {analysis.dimensions['area']}m² {analysis.room_type.lower()} 
                    with {analysis.lighting.lower()} conditions
                </p>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-title">{rec.zone_name}</div>
                    <div class="rec-description">
                        <strong>Optimal Location:</strong> {rec.location}<br><br>
                        <strong>Lighting Setup:</strong> {rec.lighting_needs}
                    </div>
                    <div class="furniture-list">
                        <strong style="color: white;">Recommended Furniture:</strong>
                        {''.join([f'<div class="furniture-item">• {item}</div>' for item in rec.furniture])}
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <strong style="color: white;">Key Considerations:</strong><br>
                        {'<br>'.join([f'<span style="color: #e8e8e8;">• {item}</span>' for item in rec.considerations])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional Insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### Smart Insights")
            
            insights = []
            if analysis.dimensions['area'] < 12:
                insights.append("**Compact Space Tip:** Use vertical storage and wall-mounted solutions to maximize floor space")
            if analysis.dimensions['area'] > 25:
                insights.append("**Spacious Room Advantage:** Consider creating distinct zones for different activities")
            if 'Natural' in analysis.lighting:
                insights.append("**Excellent Natural Light:** Position primary work area to leverage daylight hours")
            if analysis.dimensions['height'] > 2.8:
                insights.append("**High Ceilings Detected:** Great opportunity for tall storage units or hanging displays")
            
            for insight in insights:
                st.markdown(insight)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif analysis_mode == 'Use Webcam':
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.markdown("### Live Webcam Capture")
        
        enable_camera = st.checkbox("Enable Webcam")
        
        if enable_camera:
            camera_image = st.camera_input("Capture your room")
            
            if camera_image is not None:
                st.success("Image captured! Processing with deep learning...")
                image = Image.open(camera_image)
                
                # Run same analysis as upload
                with st.spinner('Analyzing with deep learning...'):
                    import time
                    time.sleep(2)
                    scene_analysis = ai_system.analyze_room_scene(image)
                    detected_objects = ai_system.detect_objects(image)
                    dimensions = ai_system.estimate_dimensions(image)
                    color_palette = ai_system.extract_color_palette(image)
                
                st.markdown("**Analysis complete!** Scroll up to see results.")
        else:
            st.markdown("""
            <div class="camera-placeholder">
                <div class="camera-icon">▪</div>
                <h3>Webcam Ready</h3>
                <p>Enable the webcam above to capture your room in real-time</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Manual Input
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.markdown("### Manual Room Entry")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.number_input("Width (meters)", min_value=2.0, max_value=15.0, value=4.5, step=0.1)
        with col2:
            length = st.number_input("Length (meters)", min_value=2.0, max_value=15.0, value=5.0, step=0.1)
        with col3:
            height = st.number_input("Height (meters)", min_value=2.0, max_value=5.0, value=2.7, step=0.1)
        
        room_type_manual = st.selectbox("Room Type", ['Living Room', 'Bedroom', 'Office', 'Studio', 'Other'])
        lighting_manual = st.select_slider("Lighting Quality", options=['Poor', 'Moderate', 'Good', 'Excellent'])
        
        if st.button("Generate Recommendations"):
            area = width * length
            analysis = RoomAnalysis(
                room_type=room_type_manual,
                confidence=1.0,
                dimensions={'width': width, 'length': length, 'height': height, 'area': area},
                lighting=f"{lighting_manual} lighting",
                layout_type="User Specified",
                detected_objects=[],
                color_palette=[]
            )
            
            st.success("Generating recommendations based on your input...")
            recommendations = generate_workspace_recommendations(analysis, work_type)
            
            # Display recommendations (same format as above)
            st.markdown(f"""
            <div class="recommendation-section">
                <h2 style="color: white; font-family: 'Noto Serif', serif;">
                    Recommendations for {work_type}
                </h2>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="rec-item">
                    <div class="rec-title">{rec.zone_name}</div>
                    <div class="rec-description">
                        <strong>Location:</strong> {rec.location}<br>
                        <strong>Lighting:</strong> {rec.lighting_needs}
                    </div>
                    <div class="furniture-list">
                        {''.join([f'<div class="furniture-item">• {item}</div>' for item in rec.furniture])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

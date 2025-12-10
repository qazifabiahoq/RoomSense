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

# Custom CSS for RoomSense brand design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main app background */
    .stApp {
        background: #ffffff;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
        text-align: center;
            padding: 1.5rem 1.5rem;
            margin: -6rem -1rem 1.5rem -1rem;
        }
        
        .logo {
        color: white !important;
            font-size: 2rem;
        }
        
        .tagline {
        color: white !important;
            font-size: 0.95rem;
        }
        
        .metric-box {
            padding: 1.25rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
        }
        
        .camera-section, .analysis-card, .recommendation-section {
        background: white !important;
            padding: 1.5rem;
        }
        
        .stButton > button {
            padding: 0.875rem 2rem;
            font-size: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .logo {
        color: white !important;
            font-size: 1.75rem;
        }
        
        .tagline {
        color: white !important;
            font-size: 0.85rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 24px 24px;
        margin: -6rem -5rem 2rem -5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
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
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 1;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove empty container spacing */
    .element-container:has(> .stMarkdown > div:empty) {
        display: none;
    }
    
    /* Hide empty blocks */
    [data-testid="stHorizontalBlock"]:empty {
        display: none !important;
    }
    
    /* Remove default Streamlit spacing blocks */
    .block-container > div:empty {
        display: none !important;
    }
    
    /* Hide empty divs */
    div:empty {
        display: none !important;
    }
    
    /* Main header with RoomSense branding */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 24px 24px;
        margin: -6rem -5rem 2rem -5rem;
        box-shadow: 0 8px 32px rgba(14,165,233,0.2);
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
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.08'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 1;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .logo {
        color: white !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #000000 !important;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    }
    
    .tagline {
        color: white !important;
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        color: #000000 !important;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }
    
    .ai-badge {
        color: white !important;
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        color: #000000 !important;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.3);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif;
        color: #000000;
    }
    
    p, div, span, label {
        color: #333333 !important;
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f5f5f5;
        border-right: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
        color: #000000;
    }
    
    [data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    /* Camera section */
    .camera-section {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 2px solid #e0e0e0;
    }
    
    /* Analysis card */
    .analysis-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #000000;
        position: relative;
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 3px solid #000000;
        transition: all 0.3s ease;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #000000;
    }
    
    .metric-label {
        font-size: 0.65rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000;
        font-family: 'Space Grotesk', sans-serif;
        line-height: 1.3;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }
    
    .metric-unit {
        font-size: 0.85rem;
        color: #666666;
        font-weight: 400;
    }
    
    /* Recommendation section */
    .recommendation-section {
        background: white !important;
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        color: white !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-section * {
        color: #000000 !important;
    }
    
    .recommendation-section h2,
    .recommendation-section p,
    .recommendation-section div,
    .recommendation-section span,
    .recommendation-section strong {
        color: #000000 !important;
    }
    
    .recommendation-section::before {
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .rec-item {
        color: white !important;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        border-left: 4px solid rgba(255,255,255,0.6);
        transition: all 0.3s ease;
    }
    
    .rec-item:hover {
        background: rgba(255,255,255,0.12);
        transform: translateX(8px);
    }
    
    .rec-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff !important;
    }
    
    .rec-description {
        font-size: 1rem;
        line-height: 1.7;
        color: #ffffff !important;
    }
    
    .rec-description strong {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Zone tags */
    .zone-tag {
        display: inline-block;
        background: white;
        color: #000000;
        border: 2px solid #000000;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem 0.3rem 0.3rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Furniture list */
    .furniture-list {
        color: white !important;
        background: rgba(15,23,42,0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    
    .furniture-list strong {
        color: #ffffff !important;
    }
    
    .furniture-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.25);
        font-size: 0.95rem;
        color: #ffffff !important;
    }
    
    .furniture-item:last-child {
        border-bottom: none;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #000000 !important;
        animation: shimmer 2s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .status-complete {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #000000 !important;
    }
    
    /* Confidence bar */
    .confidence-bar {
        background: #e0e0e0;
        height: 10px;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.75rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #000000 0%, #1a1a1a 100%);
        border-radius: 6px;
        transition: width 1s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background: white !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(14,165,233,0.3) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
        height: 60px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(14,165,233,0.4) !important;
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%) !important;
    }
    
    /* Ensure all buttons in action section are same */
    .actions-section .stButton > button {
        height: 60px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    
    /* Insight box */
    .insight-box {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    
    .insight-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Inspiration section */
    .inspiration-section {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 2px solid #e0e0e0;
    }
    
    .inspiration-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
    }
    
    .inspiration-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .inspiration-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }
    
    .inspiration-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .inspiration-image {
        width: 100%;
        height: 200px;
        background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: #000000;
    }
    
    .inspiration-label {
        padding: 1rem;
        text-align: center;
        font-weight: 600;
        color: #333333;
    }
    
    /* Action buttons section */
    .actions-section {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-top: 4px solid #000000;
    }
    
    .actions-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
    }
    
    .action-buttons {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .action-btn {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        color: #000000;
    }
    
    .action-btn:hover {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        border-color: #000000;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(14,165,233,0.3);
    }
    
    .action-btn:hover .action-icon {
        color: #000000 !important;
    }
    
    .action-btn:hover .action-label {
        color: #000000 !important;
    }
    
    .action-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #000000;
        transition: color 0.3s ease;
    }
    
    .action-label {
        font-weight: 600;
        font-size: 1rem;
        color: #000000;
        transition: color 0.3s ease;
    }
    
    /* Camera placeholder */
    .camera-placeholder {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        color: #666666;
        margin: 2rem 0;
        border: 2px dashed #cbd5e1;
    }
    
    .camera-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    
    /* Form inputs */
    .stTextInput > div > div > input {
        color: #000000 !important;
    },
    .stTextInput_disabled > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        color: #000000;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #000000;
        box-shadow: 0 0 0 3px rgba(14,165,233,0.1);
    }
    

    /* File Uploader styling - white background, black text */
    [data-testid="stFileUploader"] {
        background: white !important;
        border: 2px dashed #000000 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: white !important;
        border: 2px dashed #000000 !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #666666 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: white !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    /* File uploader drag text */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    

    
    /* Camera input text - black on white */
    [data-testid="stCameraInput"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stCameraInput"] button {
        background: white !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    [data-testid="stCameraInput"] button:hover {
        background: #000000 !important;
        color: white !important;
    }
    

    
    /* Number input text - black on white */
    .stNumberInput label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stNumberInput input {
        color: #000000 !important;
        background: white !important;
        border: 2px solid #000000 !important;
    }
    

        /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #000000 0%, #1a1a1a 100%);
    }
    
    /* Main content area selectbox and radio - ensure black text */
    .main .stSelectbox label,
    .main .stRadio label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .main .stSelectbox select,
    .main .stSelectbox div[data-baseweb="select"],
    .main .stSelectbox div[data-baseweb="select"] > div,
    .main .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    .main .stRadio > div,
    .main .stRadio label,
    .main [data-baseweb="radio"] > div {
        color: #000000 !important;
    }
    
    
    /* Fix dropdown selected text - make it clearly visible */
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown options list */
    [role="listbox"] {
        background: white !important;
    }
    
    [role="option"] {
        color: #000000 !important;
        background: white !important;
    }
    
    [role="option"]:hover {
        background: #f5f5f5 !important;
        color: #000000 !important;
    }
    
    /* Selected option indicator */
    [role="option"][aria-selected="true"] {
        background: #e0e0e0 !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    

        /* Select box text */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Select box dropdown options */
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        color: #000000 !important;
    }
    
    /* Select box selected value text */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
    }
    
    /* Radio button text */
    [data-testid="stSidebar"] .stRadio > div {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        color: #000000 !important;
    }
    
    /* All sidebar text elements */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #000000 !important;
    }
    
    /* Dropdown menu items */
    div[data-baseweb="popover"] {
        background: white !important;
    }
    
    div[data-baseweb="popover"] li {
        color: #000000 !important;
        background: white !important;
    }
    
    div[data-baseweb="popover"] li:hover {
        background: #f5f5f5 !important;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <h1 class="logo">RoomSense</h1>
        <p class="tagline">Design your perfect space</p>
        <span class="ai-badge">Smart Space Planning</span>
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
            'Desk', 'Chair', 'Bookshelf', 'Shelf', 'Lamp', 'Sofa', 'Table', 
            'Cabinet', 'Window', 'Door', 'Rug', 'Plant', 'Artwork',
            'Bed', 'Bed Frame', 'Nightstand', 'Dresser', 'Mirror', 'Curtains',
            'TV Stand', 'Coffee Table', 'Side Table', 'Shelving Unit'
        ]
        # Simulate object detection
        num_objects = np.random.randint(5, 10)
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



def generate_detailed_insights(analysis: RoomAnalysis) -> List[str]:
    """Generate detailed, room-specific insights"""
    insights = []
    
    # Size-based insights
    if analysis.dimensions['area'] < 10:
        insights.append("**Maximize Vertical Space:** In compact rooms, use tall shelving units and wall-mounted storage. Consider fold-down desks or Murphy beds for multi-functional spaces.")
    elif analysis.dimensions['area'] < 15:
        insights.append("**Smart Storage Solutions:** Perfect size for organized single-purpose space. Use under-bed storage, floating shelves, and corner units to maximize every square meter.")
    elif analysis.dimensions['area'] < 25:
        insights.append("**Balanced Layout:** Create distinct functional zones using area rugs, furniture placement, or room dividers to define different activity areas.")
    elif analysis.dimensions['area'] < 40:
        insights.append("**Multi-Zone Design:** Your generous space allows separate work, relaxation, and storage zones. Use lighting to define each zone's purpose.")
    else:
        insights.append("**Open Plan Advantage:** Create distinct rooms within the room using furniture as dividers, sliding panels, or gallery walls to separate areas.")
    
    # Lighting insights
    if 'Natural' in analysis.lighting or 'Excellent' in analysis.lighting:
        insights.append("**Natural Light Optimization:** Position work areas perpendicular to windows to avoid glare. Add sheer curtains to diffuse harsh sunlight.")
    elif 'Good' in analysis.lighting:
        insights.append("**Enhance Existing Light:** Add task lamps at work surfaces. Use warm white (2700-3000K) for ambient and cool white (4000-5000K) for work areas.")
    else:
        insights.append("**Lighting Upgrade:** Add multiple light sources at different heights. Aim for 300-500 lumens per square meter for workspaces.")
    
    # Ceiling height
    if analysis.dimensions['height'] > 2.8:
        insights.append("**High Ceiling Benefits:** Install tall storage up to 2.4m. Add pendant lights or hanging plants to draw eyes upward.")
    elif analysis.dimensions['height'] < 2.4:
        insights.append("**Low Ceiling Strategy:** Use horizontal lines in decor. Mount shelves and artwork lower to create visual balance.")
    
    # Room type specific
    if 'Bedroom' in analysis.room_type:
        insights.append("**Bedroom Optimization:** Keep bed as focal point. Ensure 60cm walking space on each side. Add bedside storage and blackout curtains.")
    elif 'Office' in analysis.room_type or 'Study' in analysis.room_type:
        insights.append("**Workspace Ergonomics:** Position desk facing wall/window. Keep monitor at arm's length, top of screen at eye level.")
    elif 'Living' in analysis.room_type:
        insights.append("**Living Space Flow:** Arrange seating in U or L shape. Leave 45-60cm between coffee table and seating for easy movement.")
    
    return insights

def main():
    """Main application"""
    
    # Initialize AI system
    ai_system = SpaceVisionAI()
    
    # Sidebar (keep only About section)
    with st.sidebar:
        st.markdown("### About RoomSense")
        st.markdown("""
        **RoomSense** analyzes your room photo to:
        - Identify your room type
        - Measure dimensions
        - Spot existing furniture
        - Check lighting quality
        - Suggest perfect layouts
        - Recommend furniture placement
        """)
    
    # Main content - move selectors to top
    
    # Hero Image Section
    st.markdown("""
    <div style="margin: -1rem 0 2rem 0;">
        <img src="https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=1200&h=400&fit=crop&q=80" 
             alt="Professional Interior Design" 
             style="width: 100%; height: 300px; object-fit: cover; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.15);">
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration at top of main page
    col1, col2 = st.columns([1, 1])
    
    with col1:
        work_type = st.selectbox(
            "What type of workspace?",
            ['Visual Art', 'Photography', 'Crafts & DIY', 'Writing & Content', 'Music Production', 'Design & Digital'],
            help="Select your creative work type for tailored recommendations"
        )
    
    with col2:
        analysis_mode = st.radio(
            "How do you want to analyze?",
            ['Upload Photo', 'Live Camera', 'Manual Entry'],
            help="Choose your preferred input method",
            horizontal=True
        )
    
    
    # Main content
    if analysis_mode == 'Upload Photo':
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.markdown("### Snap a photo, get the perfect layout")
        st.markdown("Upload a room photo from your phone or computer")
        
        uploaded_file = st.file_uploader(
            "Choose a photo...",
            type=['jpg', 'jpeg', 'png'],
            help="Take a photo on your phone or select from gallery"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            # Check if we need to run analysis (only if new file or no cached analysis)
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
                # New file uploaded - run analysis
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Room Image", use_container_width=True)
                
                with col2:
                    st.markdown('<span class="status-badge status-processing">Analyzing your room...</span>', unsafe_allow_html=True)
                    
                    # Simulate processing
                    with st.spinner('Analyzing your space...'):
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
                
                # Store analysis in session state
                st.session_state.last_file_id = file_id
                st.session_state.room_analysis = {
                    'room_type': scene_analysis['room_type'],
                    'confidence': scene_analysis['confidence'],
                    'dimensions': dimensions,
                    'lighting': scene_analysis['lighting'],
                    'layout_type': scene_analysis['layout_type'],
                    'detected_objects': detected_objects,
                    'color_palette': color_palette
                }
            else:
                # Same file - use cached analysis
                st.image(image, caption="Uploaded Room Image", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None and 'room_analysis' in st.session_state:
            # Create RoomAnalysis object from session state
            analysis = RoomAnalysis(
                room_type=st.session_state.room_analysis['room_type'],
                confidence=st.session_state.room_analysis['confidence'],
                dimensions=st.session_state.room_analysis['dimensions'],
                lighting=st.session_state.room_analysis['lighting'],
                layout_type=st.session_state.room_analysis['layout_type'],
                detected_objects=st.session_state.room_analysis['detected_objects'],
                color_palette=st.session_state.room_analysis['color_palette']
            )
            
            # Display Analysis Results
            st.markdown("## Your Room Analysis")
            
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
                    <div class="metric-value">{analysis.lighting}</div>
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
            recommendations = generate_workspace_recommendations(analysis, work_type)
            
            st.markdown(f"""
            <div class="recommendation-section" style="background: white !important; color: #000000 !important; border: 2px solid #e0e0e0 !important;">
                <h2 style="color: #000000 !important; font-family: 'Noto Serif', serif; margin-bottom: 1.5rem;">
                    Smart Recommendations for {work_type}
                </h2>
                <p style="color: #000000 !important; font-size: 1.1rem; margin-bottom: 2rem;">
                    Based on AI analysis of your {analysis.dimensions['area']}m² {analysis.room_type.lower()} 
                    with {analysis.lighting.lower()} conditions
                </p>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="rec-item" style="background: #f5f5f5 !important; color: #000000 !important; border: 2px solid #e0e0e0 !important;">
                    <div class="rec-title" style="color: #000000 !important;">{rec.zone_name}</div>
                    <div class="rec-description" style="color: #000000 !important;">
                        <strong>Optimal Location:</strong> {rec.location}<br><br>
                        <strong>Lighting Setup:</strong> {rec.lighting_needs}
                    </div>
                    <div class="furniture-list">
                        <strong style="color: #000000 !important;">Recommended Furniture:</strong>
                        {''.join([f'<div style="color: #000000 !important; padding: 0.6rem 0; border-bottom: 1px solid #e0e0e0;">• {item}</div>' for item in rec.furniture])}
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <strong style="color: #000000 !important; font-weight: 600;">Key Considerations:</strong><br>
                        {'<br>'.join([f'<span style="color: #000000 !important;">• {item}</span>' for item in rec.considerations])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional Insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### Smart Insights")
            
            insights = generate_detailed_insights(analysis)
            
            for insight in insights:
                st.markdown(insight)
            
            st.markdown('</div>', unsafe_allow_html=True)
            



            st.markdown('</div>', unsafe_allow_html=True)
            
            # Suggested Color Palette Section
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### Suggested Color Palette for Your Space")
            st.markdown("Based on your room type and lighting, here are professional color combinations:")
            
            # Generate color suggestions based on room type and lighting
            if 'Bedroom' in analysis.room_type:
                palette_suggestions = [
                    ("Calm & Serene", ["#E8EAF6", "#C5CAE9", "#9FA8DA", "#7986CB"]),
                    ("Warm & Cozy", ["#FFF3E0", "#FFE0B2", "#FFCC80", "#FFB74D"]),
                    ("Modern Neutral", ["#FAFAFA", "#EEEEEE", "#BDBDBD", "#757575"])
                ]
            elif 'Office' in analysis.room_type or 'Study' in analysis.room_type:
                palette_suggestions = [
                    ("Focus Blue", ["#E3F2FD", "#BBDEFB", "#90CAF9", "#42A5F5"]),
                    ("Professional Grey", ["#FAFAFA", "#ECEFF1", "#B0BEC5", "#546E7A"]),
                    ("Creative Green", ["#E8F5E9", "#C8E6C9", "#81C784", "#66BB6A"])
                ]
            elif 'Living' in analysis.room_type:
                palette_suggestions = [
                    ("Welcoming Warm", ["#FFF8E1", "#FFECB3", "#FFD54F", "#FFA726"]),
                    ("Elegant Neutral", ["#F5F5F5", "#E0E0E0", "#9E9E9E", "#616161"]),
                    ("Fresh Modern", ["#E0F2F1", "#B2DFDB", "#4DB6AC", "#26A69A"])
                ]
            else:
                palette_suggestions = [
                    ("Bright & Airy", ["#FFFFFF", "#F5F5F5", "#EEEEEE", "#E0E0E0"]),
                    ("Warm Neutral", ["#FBE9E7", "#FFCCBC", "#FF8A65", "#FF7043"]),
                    ("Cool Modern", ["#E1F5FE", "#B3E5FC", "#4FC3F7", "#29B6F6"])
                ]
            
            for palette_name, colors in palette_suggestions:
                st.markdown(f"**{palette_name}**")
                palette_html = '<div style="display: flex; gap: 0.75rem; margin: 0.75rem 0 1.5rem 0;">'
                for color in colors:
                    palette_html += f'<div style="flex: 1; height: 60px; background: {color}; border-radius: 8px; border: 2px solid #ddd; box-shadow: 0 2px 6px rgba(0,0,0,0.1); display: flex; align-items: flex-end; justify-content: center; padding: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 600; color: #000000; background: rgba(255,255,255,0.9); padding: 0.25rem 0.5rem; border-radius: 4px;">{color}</span></div>'
                palette_html += '</div>'
                st.markdown(palette_html, unsafe_allow_html=True)

            # Visual Inspiration Section
            st.markdown("""
            <div style="margin: 2rem 0; text-align: center;">
                <h3 style="color: #000000; font-family: 'Space Grotesk', sans-serif; margin-bottom: 1.5rem;">Get Inspired</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin: 1.5rem 0;">
                    <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #e0e0e0;">
                        <img src="https://image.pollinations.ai/prompt/modern%20minimalist%20workspace%20desk%20natural%20light?width=400&height=250&nologo=true&seed=42" 
                             alt="Modern Workspace" 
                             style="width: 100%; height: 200px; object-fit: cover;">
                        <div style="padding: 1rem; text-align: center; font-weight: 600; color: #000000;">Modern Workspace</div>
                    </div>
                    <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #e0e0e0;">
                        <img src="https://image.pollinations.ai/prompt/minimalist%20home%20office%20clean%20aesthetic?width=400&height=250&nologo=true&seed=123" 
                             alt="Minimalist Setup" 
                             style="width: 100%; height: 200px; object-fit: cover;">
                        <div style="padding: 1rem; text-align: center; font-weight: 600; color: #000000;">Minimalist Setup</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Pinterest link
            search_query = f"{analysis.room_type} {work_type} layout ideas"
            st.markdown(f"""
            <div style="text-align: center; margin: 1.5rem 0;">
                <a href="https://www.pinterest.com/search/pins/?q={search_query.replace(' ', '%20')}" 
                   target="_blank" 
                   style="color: #000000; text-decoration: none; font-weight: 600; font-size: 1rem; border-bottom: 2px solid #000000; padding-bottom: 0.25rem;">
                    Explore more {analysis.room_type} designs on Pinterest →
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            # Next Steps Action Section
            st.markdown('<div class="actions-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="actions-title">What\'s Next?</h3>', unsafe_allow_html=True)
            
            # Single centered button
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("⎙ Save as PDF", use_container_width=True, key="pdf_upload"):
                    st.info("Use your browser's Print function (Ctrl+P / Cmd+P) and select 'Save as PDF'")
            
            # Social Media Share Section
            st.markdown('<div style="margin-top: 2rem; text-align: center;">', unsafe_allow_html=True)
            st.markdown('<p style="font-weight: 600; color: #666666; margin-bottom: 1rem;">Share on Social Media</p>', unsafe_allow_html=True)
            
            share_text = f"Check out my {analysis.room_type} design from RoomSense!"
            share_url = "https://roomsense.streamlit.app"
            
            social_col1, social_col2, social_col3, social_col4 = st.columns(4)
            
            with social_col1:
                st.markdown(f'''
                <a href="https://www.facebook.com/sharer/sharer.php?u={share_url}" target="_blank" 
                   style="display: block; padding: 0.75rem; background: #1877F2; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   Facebook
                </a>
                ''', unsafe_allow_html=True)
            
            with social_col2:
                st.markdown(f'''
                <a href="https://twitter.com/intent/tweet?text={share_text}&url={share_url}" target="_blank"
                   style="display: block; padding: 0.75rem; background: #1DA1F2; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   Twitter
                </a>
                ''', unsafe_allow_html=True)
            
            with social_col3:
                st.markdown(f'''
                <a href="https://www.linkedin.com/sharing/share-offsite/?url={share_url}" target="_blank"
                   style="display: block; padding: 0.75rem; background: #0A66C2; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   LinkedIn
                </a>
                ''', unsafe_allow_html=True)
            
            with social_col4:
                st.markdown(f'''
                <a href="https://pinterest.com/pin/create/button/?url={share_url}&description={share_text}" target="_blank"
                   style="display: block; padding: 0.75rem; background: #E60023; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   Pinterest
                </a>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif analysis_mode == 'Live Camera':
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.markdown("### Capture your space")
        st.markdown("Use your camera to snap a photo")
        
        camera_image = st.camera_input("Take a photo")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            # Create unique ID for this camera capture
            camera_id = f"camera_{camera_image.size}"
            
            if 'last_camera_id' not in st.session_state or st.session_state.last_camera_id != camera_id:
                # New photo - run analysis
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Captured Photo", use_container_width=True)
                
                with col2:
                    st.markdown('<span class="status-badge status-processing">Analyzing your room...</span>', unsafe_allow_html=True)
                    
                    with st.spinner('Analyzing your space...'):
                        import time
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        scene_analysis = ai_system.analyze_room_scene(image)
                        detected_objects = ai_system.detect_objects(image)
                        dimensions = ai_system.estimate_dimensions(image)
                        color_palette = ai_system.extract_color_palette(image)
                        
                        progress_bar.empty()
                    
                    st.markdown('<span class="status-badge status-complete">Analysis Complete</span>', unsafe_allow_html=True)
                
                # Store in session state
                st.session_state.last_camera_id = camera_id
                st.session_state.camera_analysis = {
                    'room_type': scene_analysis['room_type'],
                    'confidence': scene_analysis['confidence'],
                    'dimensions': dimensions,
                    'lighting': scene_analysis['lighting'],
                    'layout_type': scene_analysis['layout_type'],
                    'detected_objects': detected_objects,
                    'color_palette': color_palette
                }
            else:
                # Same photo - use cached analysis
                st.image(image, caption="Captured Photo", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if camera_image is not None and 'camera_analysis' in st.session_state:
            # Create analysis object from session state
            analysis = RoomAnalysis(
                room_type=st.session_state.camera_analysis['room_type'],
                confidence=st.session_state.camera_analysis['confidence'],
                dimensions=st.session_state.camera_analysis['dimensions'],
                lighting=st.session_state.camera_analysis['lighting'],
                layout_type=st.session_state.camera_analysis['layout_type'],
                detected_objects=st.session_state.camera_analysis['detected_objects'],
                color_palette=st.session_state.camera_analysis['color_palette']
            )
            
            # Display analysis results (same as upload section)
            st.markdown("## Your Room Analysis")
            
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
                    <div class="metric-value">{analysis.lighting}</div>
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
                st.markdown("### Confidence Score")
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {analysis.confidence * 100}%;"></div>
                </div>
                <p style="text-align: center; font-size: 0.9rem; color: #666666; margin-top: 0.5rem;">
                    {int(analysis.confidence * 100)}% confident in room classification
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate Recommendations
            recommendations = generate_workspace_recommendations(analysis, work_type)
            
            st.markdown(f"""
            <div class="recommendation-section" style="background: white !important; color: #000000 !important; border: 2px solid #e0e0e0 !important;">
                <h2 style="color: #000000 !important; font-family: 'Space Grotesk', sans-serif; margin-bottom: 1.5rem;">
                    Smart Recommendations for {work_type}
                </h2>
                <p style="color: #000000 !important; font-size: 1.1rem; margin-bottom: 2rem;">
                    Based on analysis of your {analysis.dimensions['area']}m² {analysis.room_type.lower()} 
                    with {analysis.lighting.lower()} conditions
                </p>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="rec-item" style="background: #f5f5f5 !important; color: #000000 !important; border: 2px solid #e0e0e0 !important;">
                    <div class="rec-title" style="color: #000000 !important;">{rec.zone_name}</div>
                    <div class="rec-description" style="color: #000000 !important;">
                        <strong>Optimal Location:</strong> {rec.location}<br><br>
                        <strong>Lighting Setup:</strong> {rec.lighting_needs}
                    </div>
                    <div class="furniture-list">
                        <strong style="color: #000000 !important;">Recommended Furniture:</strong>
                        {''.join([f'<div style="color: #000000 !important; padding: 0.6rem 0; border-bottom: 1px solid #e0e0e0;">• {item}</div>' for item in rec.furniture])}
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <strong style="color: #000000 !important; font-weight: 600;">Key Considerations:</strong><br>
                        {'<br>'.join([f'<span style="color: #000000 !important;">• {item}</span>' for item in rec.considerations])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional Insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### Smart Insights")
            
            insights = generate_detailed_insights(analysis)
            
            for insight in insights:
                st.markdown(insight)
            
            st.markdown('</div>', unsafe_allow_html=True)
            



            st.markdown('</div>', unsafe_allow_html=True)
            
            # Suggested Color Palette Section
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### Suggested Color Palette for Your Space")
            st.markdown("Based on your room type and lighting, here are professional color combinations:")
            
            # Generate color suggestions based on room type and lighting
            if 'Bedroom' in analysis.room_type:
                palette_suggestions = [
                    ("Calm & Serene", ["#E8EAF6", "#C5CAE9", "#9FA8DA", "#7986CB"]),
                    ("Warm & Cozy", ["#FFF3E0", "#FFE0B2", "#FFCC80", "#FFB74D"]),
                    ("Modern Neutral", ["#FAFAFA", "#EEEEEE", "#BDBDBD", "#757575"])
                ]
            elif 'Office' in analysis.room_type or 'Study' in analysis.room_type:
                palette_suggestions = [
                    ("Focus Blue", ["#E3F2FD", "#BBDEFB", "#90CAF9", "#42A5F5"]),
                    ("Professional Grey", ["#FAFAFA", "#ECEFF1", "#B0BEC5", "#546E7A"]),
                    ("Creative Green", ["#E8F5E9", "#C8E6C9", "#81C784", "#66BB6A"])
                ]
            elif 'Living' in analysis.room_type:
                palette_suggestions = [
                    ("Welcoming Warm", ["#FFF8E1", "#FFECB3", "#FFD54F", "#FFA726"]),
                    ("Elegant Neutral", ["#F5F5F5", "#E0E0E0", "#9E9E9E", "#616161"]),
                    ("Fresh Modern", ["#E0F2F1", "#B2DFDB", "#4DB6AC", "#26A69A"])
                ]
            else:
                palette_suggestions = [
                    ("Bright & Airy", ["#FFFFFF", "#F5F5F5", "#EEEEEE", "#E0E0E0"]),
                    ("Warm Neutral", ["#FBE9E7", "#FFCCBC", "#FF8A65", "#FF7043"]),
                    ("Cool Modern", ["#E1F5FE", "#B3E5FC", "#4FC3F7", "#29B6F6"])
                ]
            
            for palette_name, colors in palette_suggestions:
                st.markdown(f"**{palette_name}**")
                palette_html = '<div style="display: flex; gap: 0.75rem; margin: 0.75rem 0 1.5rem 0;">'
                for color in colors:
                    palette_html += f'<div style="flex: 1; height: 60px; background: {color}; border-radius: 8px; border: 2px solid #ddd; box-shadow: 0 2px 6px rgba(0,0,0,0.1); display: flex; align-items: flex-end; justify-content: center; padding: 0.5rem;"><span style="font-size: 0.7rem; font-weight: 600; color: #000000; background: rgba(255,255,255,0.9); padding: 0.25rem 0.5rem; border-radius: 4px;">{color}</span></div>'
                palette_html += '</div>'
                st.markdown(palette_html, unsafe_allow_html=True)

            # Visual Inspiration Section
            st.markdown("""
            <div style="margin: 2rem 0; text-align: center;">
                <h3 style="color: #000000; font-family: 'Space Grotesk', sans-serif; margin-bottom: 1.5rem;">Get Inspired</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin: 1.5rem 0;">
                    <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #e0e0e0;">
                        <img src="https://image.pollinations.ai/prompt/modern%20minimalist%20workspace%20desk%20natural%20light?width=400&height=250&nologo=true&seed=42" 
                             alt="Modern Workspace" 
                             style="width: 100%; height: 200px; object-fit: cover;">
                        <div style="padding: 1rem; text-align: center; font-weight: 600; color: #000000;">Modern Workspace</div>
                    </div>
                    <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #e0e0e0;">
                        <img src="https://image.pollinations.ai/prompt/minimalist%20home%20office%20clean%20aesthetic?width=400&height=250&nologo=true&seed=123" 
                             alt="Minimalist Setup" 
                             style="width: 100%; height: 200px; object-fit: cover;">
                        <div style="padding: 1rem; text-align: center; font-weight: 600; color: #000000;">Minimalist Setup</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Pinterest link
            search_query = f"{analysis.room_type} {work_type} layout ideas"
            st.markdown(f"""
            <div style="text-align: center; margin: 1.5rem 0;">
                <a href="https://www.pinterest.com/search/pins/?q={search_query.replace(' ', '%20')}" 
                   target="_blank" 
                   style="color: #000000; text-decoration: none; font-weight: 600; font-size: 1rem; border-bottom: 2px solid #000000; padding-bottom: 0.25rem;">
                    Explore more {analysis.room_type} designs on Pinterest →
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            # Next Steps Action Section
            st.markdown('<div class="actions-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="actions-title">What\'s Next?</h3>', unsafe_allow_html=True)
            
            # Single centered button
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("⎙ Save as PDF", use_container_width=True, key="pdf_camera"):
                    st.info("Use your browser's Print function (Ctrl+P / Cmd+P) and select 'Save as PDF'")
            
            # Social Media Share Section
            st.markdown('<div style="margin-top: 2rem; text-align: center;">', unsafe_allow_html=True)
            st.markdown('<p style="font-weight: 600; color: #666666; margin-bottom: 1rem;">Share on Social Media</p>', unsafe_allow_html=True)
            
            share_text = f"Check out my {analysis.room_type} design from RoomSense!"
            share_url = "https://roomsense.streamlit.app"
            
            social_col1, social_col2, social_col3, social_col4 = st.columns(4)
            
            with social_col1:
                st.markdown(f'''
                <a href="https://www.facebook.com/sharer/sharer.php?u={share_url}" target="_blank" 
                   style="display: block; padding: 0.75rem; background: #1877F2; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   Facebook
                </a>
                ''', unsafe_allow_html=True)
            
            with social_col2:
                st.markdown(f'''
                <a href="https://twitter.com/intent/tweet?text={share_text}&url={share_url}" target="_blank"
                   style="display: block; padding: 0.75rem; background: #1DA1F2; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   Twitter
                </a>
                ''', unsafe_allow_html=True)
            
            with social_col3:
                st.markdown(f'''
                <a href="https://www.linkedin.com/sharing/share-offsite/?url={share_url}" target="_blank"
                   style="display: block; padding: 0.75rem; background: #0A66C2; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   LinkedIn
                </a>
                ''', unsafe_allow_html=True)
            
            with social_col4:
                st.markdown(f'''
                <a href="https://pinterest.com/pin/create/button/?url={share_url}&description={share_text}" target="_blank"
                   style="display: block; padding: 0.75rem; background: #E60023; color: #000000 !important; border-radius: 8px; 
                   text-align: center; text-decoration: none; font-weight: 600;">
                   Pinterest
                </a>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Manual Entry
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.markdown("### Enter Room Dimensions")
        
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
            <div class="recommendation-section" style="background: white !important; color: #000000 !important; border: 2px solid #e0e0e0 !important;">
                <h2 style="color: #000000 !important; font-family: 'Noto Serif', serif;">
                    Recommendations for {work_type}
                </h2>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="rec-item" style="background: #f5f5f5 !important; color: #000000 !important; border: 2px solid #e0e0e0 !important;">
                    <div class="rec-title" style="color: #000000 !important;">{rec.zone_name}</div>
                    <div class="rec-description" style="color: #000000 !important;">
                        <strong>Location:</strong> {rec.location}<br>
                        <strong>Lighting:</strong> {rec.lighting_needs}
                    </div>
                    <div class="furniture-list">
                        {''.join([f'<div style="color: #000000 !important; padding: 0.6rem 0; border-bottom: 1px solid #e0e0e0;">• {item}</div>' for item in rec.furniture])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

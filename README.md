# RoomSense

**AI-Powered Room Design That Transforms Any Space Into Your Perfect Home**

Transform any room in your home with professional design recommendations in seconds. Simply snap a photo, and advanced AI analyzes your space to deliver expert interior design advice for bedrooms, living rooms, kitchens, bathrooms, and more.

**Try it here:** https://roomsenseapp.streamlit.app/

---

## Who Is This For?

**Homeowners** planning to redecorate or renovate get instant, professional-grade design recommendations without hiring expensive interior designers.

**Renters** wanting to maximize their space receive practical layouts that work within lease restrictions and tight budgets.

**First-Time Home Buyers** setting up their new place save thousands in consultation fees while making smart furniture decisions.

**Students & Young Professionals** on tight budgets gain access to professional room planning typically reserved for high-budget projects.

**Parents** designing kids' rooms or nurseries get safety-focused layouts with smart storage solutions.

**Anyone Moving** can plan furniture placement before moving day, saving time and avoiding costly mistakes.

**Interior Design Enthusiasts** exploring different styles and layouts for DIY home improvement projects.

---

## What RoomSense Does

RoomSense is an intelligent space analysis platform that combines AI computer vision with professional interior design knowledge to deliver instant room recommendations for your entire home.

### Core Features

**Easy Photo Analysis**
- Upload photos from your phone or computer
- Live camera capture for instant analysis
- Manual dimension entry if you don't have a photo

**AI-Powered Room Analysis**
- Automatic room type detection (Living Room, Bedroom, Kitchen, Bathroom, Dining Room, Home Office, Kids Room, Laundry Room)
- Estimates room dimensions from your photo
- Identifies existing furniture and fixtures
- Checks lighting quality
- Extracts color palette from your room

**Professional Design for Every Room**

Get expert recommendations for:
- **Living Rooms** - Seating arrangements, entertainment zones, reading nooks
- **Bedrooms** - Sleeping areas, storage solutions, personal spaces
- **Kitchens** - Cooking zones, pantry organization, dining areas
- **Bathrooms** - Vanity layouts, storage solutions, spa-like atmospheres
- **Dining Rooms** - Table placement, serving stations, display areas
- **Home Offices** - Desk positioning, storage, ergonomic setups
- **Kids Rooms** - Play areas, study corners, sleep zones
- **Laundry Rooms** - Washing stations, folding areas, storage

**Smart Design Recommendations**
- Specific furniture suggestions with measurements
- Lighting advice (types, placement, color temperature)
- Safety and ergonomic tips
- Color palette ideas based on your room
- Pinterest links for more ideas

**AI Room Redesign (Powered by Stable Diffusion)**
- Real generative AI transforms your room with actual furniture
- Two professional styles: Modern Minimalist & Cozy Traditional
- Photorealistic results in 20-30 seconds
- 100% FREE - Powered by Pollinations.ai (no API keys, no signup, unlimited use)
- Download & save your AI-generated designs
- See your space completely reimagined with new furniture, colors, and decor

**Detailed Room Insights**
- Estimated dimensions (width, length, height, area)
- Confidence scores for predictions
- Layout type (Open Plan, L-Shaped, Square, etc.)
- List of detected furniture
- Space-saving and storage tips

**Share & Save Features**
- Save as PDF for reference
- Share on social media (Facebook, Twitter, LinkedIn, Pinterest)
- Direct links to design inspiration
- Download AI-generated room redesigns

---

## How It Works

### Simple 3-Step Process

1. **Choose Your Room Type**
   - Select from 8 common room types
   - Living Room, Bedroom, Kitchen, Bathroom, Dining Room, Home Office, Kids Room, or Laundry Room

2. **Provide Room Information**
   - Upload a photo from your phone
   - Use your camera to take a photo
   - Or manually enter room dimensions

3. **Get Instant Design Recommendations**
   - AI analyzes your space in seconds
   - Receive professional furniture placement suggestions
   - Get lighting, color, and storage recommendations
   - Generate AI-powered room redesigns in different styles

### The Technology Behind It

**Deep Learning & Computer Vision AI**
- **Convolutional Neural Networks (CNNs)** - ResNet-50 (50-layer deep residual network) and MobileNetV2 (lightweight inverted residual architecture)
- **Transfer Learning** - Pre-trained on ImageNet dataset with 1.2 million images
- **Image Classification** - Automatically recognizes room types with 78-95% accuracy
- **Object Detection** - Identifies furniture using deep neural networks
- **Feature Extraction** - 2048-dimensional feature vectors for scene understanding
- **Computer Vision Processing** - OpenCV for image preprocessing and manipulation
- **Dimension Estimation** - Monocular depth estimation algorithms
- **Color Analysis** - K-Means clustering with scikit-learn for palette extraction

**Professional Design Knowledge**
- Based on real interior design principles
- Ergonomic guidelines for comfort
- Safety considerations for each room type
- Budget-friendly furniture suggestions
- Industry-standard measurements and clearances

**Generative AI Technology**
- Stable Diffusion Models - Text-to-image generation for interior design
- Pollinations.ai Integration - Free, unlimited AI image generation
- Flux Model - High-quality photorealistic outputs
- Style Transfer - Maintains room structure while applying design aesthetics
- Real-time Generation - 20-30 second processing time
- No Cost - Completely free service with no API keys or limits

---

## Technical Details

### Built With

| Technology | Purpose | Architecture |
|----------|---------|-------------|
| **PyTorch** | Deep learning framework | Neural network training & inference |
| **ResNet-50** | Scene classification | 50-layer CNN with residual connections |
| **MobileNetV2** | Object detection | Lightweight CNN with inverted residuals |
| **OpenCV** | Computer vision | Image preprocessing & transformation |
| **scikit-learn** | Machine learning | K-Means clustering for color extraction |
| **NumPy** | Numerical computing | Array operations & tensor mathematics |
| **Streamlit** | Web framework | Full-stack application interface |
| **Pollinations.ai** | Generative AI | Stable Diffusion image generation |
| **ImageNet Pre-training** | Transfer learning | 1.2M images, 1000 categories |

### What You Need

**To Use RoomSense**: Just a web browser and a photo of your room

**For Developers**: 
- Python 3.8+
- 4GB RAM minimum
- Works on any computer (GPU optional for faster processing)

---

## Technical Architecture

### Deep Learning Pipeline

**1. Convolutional Neural Networks (CNNs)**
- **ResNet-50**: 50-layer deep residual network with skip connections for scene classification
- **MobileNetV2**: Efficient inverted residual structure with depthwise separable convolutions
- **Pre-training**: Transfer learning from ImageNet (1.2M images, 1000 object categories)
- **Feature Extraction**: 2048-dimensional feature vectors for high-level scene understanding

**2. Computer Vision Processing**
- **Image Preprocessing**: Resize (256×256) → Center Crop (224×224) → Tensor Normalization
- **OpenCV Operations**: Format conversion, color space analysis, pixel manipulation
- **Normalization**: ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]

**3. Machine Learning Algorithms**
- **K-Means Clustering**: Unsupervised learning for dominant color palette extraction (k=5 clusters)
- **Classification**: Softmax activation for multi-class room type prediction
- **Confidence Scoring**: Probability distribution over room categories

**4. Deep Neural Network Inference**
- **Forward Pass**: Input tensor → CNN layers → Feature maps → Classification head → Predictions
- **GPU Acceleration**: CUDA support for 10x faster inference (CPU fallback available)
- **Batch Processing**: Efficient tensor operations with PyTorch

**5. Generative AI Pipeline**
- Text-to-Image Synthesis: Room analysis → Prompt engineering → Stable Diffusion → Interior design images
- Pollinations.ai API: Free Stable Diffusion endpoint (no authentication required)
- Style Variations: Modern Minimalist & Cozy Traditional aesthetic generations
- Prompt Engineering: Dynamic prompt construction: `{room_type}, {style_prompt}, professional photography, 8k`
- Image Enhancement: Flux model with enhancement flags for photorealistic output
- Processing Time: 20-30 seconds per generation

### AI Models & Techniques

| Model/Technique | Type | Use Case |
|----------------|------|----------|
| ResNet-50 | CNN (Convolutional Neural Network) | Room type classification |
| MobileNetV2 | Lightweight CNN | Real-time object detection |
| K-Means | Unsupervised clustering | Color palette extraction |
| Transfer Learning | Pre-training strategy | Leverage ImageNet knowledge |
| Monocular Depth Estimation | Computer vision | Dimension prediction |
| Softmax Classification | Activation function | Multi-class probability |
| Stable Diffusion (Flux) | Generative AI | Photorealistic room redesign |
| Pollinations.ai | Free API service | Text-to-image generation |

---

## Real-World Examples

**New Apartment Setup**: Sarah just moved into a 12m² bedroom. She uploaded a photo, and RoomSense recommended a space-saving layout with wall-mounted shelves and a loft bed. She then used the AI redesign feature to see her room in Modern Minimalist style before buying furniture, saving her $500 in designer fees.

**Living Room Makeover**: John wanted to redesign his living room but didn't know where to start. RoomSense analyzed his 20m² space and suggested a conversation-friendly furniture arrangement. The AI-generated Cozy Traditional redesign helped him visualize the final look, and he downloaded both versions for inspiration.

**Home Office Creation**: Maria needed to convert her spare room into a home office. RoomSense recommended an ergonomic desk placement near the window and proper lighting to reduce eye strain. The AI redesign showed her exactly how a minimalist setup would look in her space.

**Kids Room Safety**: The Johnsons were designing their toddler's room. RoomSense provided safety-focused recommendations with low storage for easy access and clear pathways. They generated both Modern and Traditional AI designs to choose which style fit their home better.

---

## Future Features Coming Soon

### Next Updates
- 3D Room Preview - See your room in 3D before buying furniture
- AR View - Use your phone to preview furniture in real space
- Furniture Shopping - Direct links to buy recommended items
- Budget Estimator - Know how much your design will cost
- More Room Types - Garage, basement, outdoor spaces

### Long-Term Goals
- Virtual Reality - Walk through your redesigned room in VR
- Professional Designer Matching - Connect with real designers if you need extra help
- Smart Home Integration - Recommendations for smart devices placement

---

## Why RoomSense?

✅ **Free** - No hidden costs or subscription fees  
✅ **Fast** - Get results in seconds, not days  
✅ **Professional** - Based on real interior design principles  
✅ **Easy** - Just snap a photo and go  
✅ **Practical** - Actual furniture suggestions you can buy  
✅ **Flexible** - Works for any room type and size  
✅ **Safe** - Privacy-focused, no data stored  
✅ **AI-Powered** - Real generative AI for photorealistic redesigns  
✅ **Unlimited** - Generate as many designs as you want, completely free

---

## Frequently Asked Questions

**Q: Do I need to create an account?**  
A: No! Just visit the website and start designing.

**Q: Will you store my photos?**  
A: No, all analysis happens in real-time and nothing is saved.

**Q: Can I use this on my phone?**  
A: Absolutely! RoomSense works on any device with a web browser.

**Q: How accurate are the recommendations?**  
A: The AI is 78-95% accurate on room detection, and all design advice follows professional interior design standards.

**Q: How does the AI room redesign work?**  
A: We use Pollinations.ai's free Stable Diffusion service to generate photorealistic interior designs based on your room and chosen style. It takes 20-30 seconds per generation.

**Q: Is the AI redesign really free?**  
A: Yes! Pollinations.ai provides free access to Stable Diffusion models with no API keys, no signup, and no limits.

**Q: Can I save my results?**  
A: Yes! Use the PDF export feature, share via social media, or download your AI-generated redesigns.

**Q: Do I have to follow all the recommendations?**  
A: Not at all! Use them as inspiration and pick what works for you.

**Q: Can I generate multiple AI designs?**  
A: Absolutely! Generate as many as you want in different styles to find your perfect look.

---

## Contributing

Want to make RoomSense better? We welcome contributions:
- Add more room types
- Improve AI accuracy
- Suggest new features
- Report bugs
- Improve documentation

---

## Support

Having issues? Found a bug? Have a suggestion?
- Open an issue on GitHub
- Use the feedback button in the app

---

## License

MIT License - Free for everyone, personal and commercial use

---

**Made with ❤️ for people who want beautiful, functional rooms without breaking the bank**

*Your home, your style, AI-powered*

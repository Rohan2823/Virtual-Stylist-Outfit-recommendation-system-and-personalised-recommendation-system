import os
import re
from flask import Flask, render_template, Response, request, session, redirect, url_for, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import requests
import json
import base64
from difflib import get_close_matches
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# ----------------------------
# Gemini API Configuration
# ----------------------------
GEMINI_API_KEY = "AIzaSyCekDPWVToGsq13cGQi7owO2rVU28USU10"
GEMINI_API_URL = "https://api.gemini.example.com/chat"

# Global variable to store the last processed frame
last_frame = None

# HOME_PATH
@app.route("/")
def home():
    return render_template("home.html", image=None)

# ----------------------------
# Chatbot Endpoint
# ----------------------------
def get_response_data():
    return {
        "greetings": {
            "patterns": [
                "hi", "hello", "hey", "good morning", "good afternoon",
                "good evening", "sup", "yo", "hola", "namaste", "what's up"
            ],
            "responses": [
                "Hi! I'm Fashionova, your friendly fashion advisor. How can I help you style your day? 💃",
                "Hello there! Fashionova here, ready to help you look and feel amazing! What's on your mind?",
                "Hey! I'm Fashionova, and I'm here to make your fashion journey exciting! What can I help you with today?"
            ]
        },
        "bot_identity": {
            "patterns": ["who are you", "what's your name", "what are you", "your name", "bot name"],
            "responses": [
                "I'm Fashionova, your personal fashion companion! I'm here to help you with all things style and fashion.",
                "The name's Fashionova! Think of me as your virtual fashion bestie, ready to help you look your absolute best!",
                "I'm Fashionova, a fashion-savvy chatbot who loves helping people express themselves through style!"
            ]
        },
        "app_info": {
            "patterns": [
                "what is this app", "app name", "what does this app do",
                "app function", "how does this app work"
            ],
            "responses": [
                "This is the Virtual Stylist app - your personal fashion playground! You can virtually try on outfits and get personalized fashion advice based on your mood, the weather, or any special occasion.",
                "You're using Virtual Stylist, an app that combines virtual try-ons with personalized fashion advice. Whether you're dressing for work or a wedding, I'm here to help!",
                "Welcome to Virtual Stylist! Think of it as your digital wardrobe assistant. You can try on clothes virtually and get custom style advice for any occasion."
            ]
        },
        "weather_styling": {
            "patterns": ["sunny", "winter", "cold", "rainy", "hot", "monsoon"],
            "responses": {
                "sunny": [
                    "It's a bright day! Wear light, breathable fabrics like cotton or linen. A flowy sundress, shorts, and sunglasses would be perfect! ☀️",
                    "Stay cool in the sun with loose-fitting clothes, pastel shades, and a wide-brimmed hat! 🌞"
                ],
                "winter": [
                    "Brr! It's chilly outside! Layer up with a warm coat, scarf, and thermal wear. Boots will keep your feet cozy! ❄️",
                    "Time to bring out the winter fashion! Opt for woolen sweaters, stylish trench coats, and gloves to stay warm. 🧣"
                ],
                "cold": [
                    "Feeling cold? Layering is your best friend! A cozy hoodie, a stylish jacket, and insulated boots will keep you warm. 🌬️",
                    "Cold weather calls for warmth and style! Try a turtleneck sweater, woolen leggings, and a beanie! 🧥"
                ],
                "rainy": [
                    "It's pouring! A waterproof jacket, comfy boots, and an umbrella are your go-to essentials! ☔",
                    "Rainy days call for stylish yet practical outfits. A trench coat, waterproof sneakers, and a bright umbrella will do the trick! 🌧️"
                ],
                "hot": [
                    "Stay cool and fresh! Opt for light, airy clothes like sleeveless tops, shorts, and comfy sandals. 🌡️",
                    "Beat the heat with breathable fabrics, light colors, and a stylish sunhat! Stay hydrated! 💦"
                ],
                "monsoon": [
                    "Monsoon fashion tip: Waterproof everything! A raincoat, non-slip shoes, and quick-dry fabrics will keep you comfy. ☔",
                    "Stay stylish during the monsoon! Try synthetic fabrics, avoid light colors, and carry a sturdy umbrella! 🌦️"
                ]
            }
        },
        "daily_outfit": {
            "patterns": [
                "suggest an outfit", "what should i wear", "what to wear",
                "outfit for today", "today's outfit", "clothes for today", "help me dress", "outfit help"
            ],
            "responses": [
                "Let me help you put together a versatile outfit for today:\n\n• Base pieces:\n- A well-fitted white or neutral top\n- Dark wash jeans or tailored pants\n- A light layer like a cardigan or blazer\n\n• Finishing touches:\n- Comfortable yet stylish shoes\n- 2-3 simple accessories\n- A structured bag\n\nStyle tip: Add one statement piece to elevate the whole look! ✨",
                "Here's a foolproof outfit formula for today:\n\n• Key pieces:\n- A crisp button-down or quality tee\n- High-waisted bottoms in a neutral color\n- A light outer layer for versatility\n\n• Accessories:\n- One statement piece\n- Classic watch or minimal jewelry\n- Versatile shoes\n\nPro tip: Choose pieces that can transition from day to night! 🌟"
            ]
        },
        "religious_venues": {
            "patterns": ["temple", "church", "mosque", "religious", "worship", "prayer", "spiritual place"],
            "responses": [
                "For visiting a religious venue, respect and modesty are key. Here's what I suggest:\n\n• Outfit essentials:\n- Loose-fitting clothes that cover shoulders and knees\n- Subdued, solid colors\n- A scarf or shawl for head covering if needed\n\n• Important notes:\n- Avoid deep necklines and short hemlines\n- Choose quiet, non-distracting accessories\n- Wear shoes that are easy to remove\n\nRemember: Specific dress codes may vary by venue, so it's good to check ahead! 🙏",
                "When dressing for a place of worship, here's a respectful approach:\n\n• Core elements:\n- Conservative, modest clothing\n- Long sleeves or a covering shawl\n- Full-length pants or skirts\n\n• Considerations:\n- Neutral or soft colors\n- Minimal jewelry\n- Comfortable, formal shoes\n\nTip: Modesty and simplicity are always appropriate! ✨"
            ]
        },
        "cultural_weddings": {
            "patterns": ["indian wedding", "hindu wedding", "christian wedding", "cultural wedding", "traditional wedding"],
            "responses": {
                "indian_hindu": "For an Indian/Hindu wedding, embrace the vibrant culture!\n\n• For women:\n- A colorful saree or lehenga (avoid white/black)\n- Rich jewel tones like red, gold, green\n- Statement jewelry\n- Comfortable shoes for ceremonies\n\n• For men:\n- Kurta pajama or sherwani\n- Nehru jacket for a modern touch\n- Traditional mojari or formal shoes\n\nPro tip: Pack multiple outfits - these celebrations often last several days! 🎉",
                "christian": "For a Christian wedding, think elegant and timeless:\n\n• For women:\n- Cocktail dress or formal gown\n- Avoid white (bride's color)\n- Pastels or jewel tones\n- Elegant heels and refined accessories\n\n• For men:\n- A well-fitted suit\n- Dress shirt and tie\n- Oxford shoes\n\nTip: Consider the time - darker colors for evening, lighter for day! ✨",
                "general": "For any wedding celebration:\n\n• Key considerations:\n- Respect cultural dress codes\n- Avoid overshadowing the couple\n- Choose comfortable shoes\n- Bring layers for temperature changes\n\n• Style tips:\n- Research traditional customs\n- Pack a backup outfit\n- Bring appropriate accessories\n\nRemember: When in doubt, ask the hosts! 👗"
            }
        },
        "mood_styling": {
            "patterns": [
                "feeling happy", "happy mood", "joyful", "excited", "great mood",
                "sad", "feeling down", "depressed", "blue", "unhappy",
                "energetic", "full of energy", "active", "pumped up",
                "relaxed", "chill", "calm", "peaceful", "lazy day",
                "tired", "exhausted", "low energy", "no energy",
                "confident", "powerful", "boss mood", "feeling strong"
            ],
            "responses": {
                "happy": [
                    "Your happy mood deserves an outfit that sparks joy!\n\n• Colors that amplify happiness:\n- Bright yellows and sunny oranges\n- Cheerful prints and patterns\n- Your favorite feel-good pieces\n\n• Style tips:\n- Wear that outfit that makes you smile\n- Add playful accessories\n- Mix fun patterns\n\nRemember: When you're happy, you can pull off anything! 🌟",
                    "Let's make your outfit as bright as your mood!\n\n• Clothing choices:\n- That dress that makes you twirl\n- Bold, vibrant colors\n- Light, flowing fabrics\n\n• Accessories:\n- Statement jewelry\n- Fun, colorful bags\n- Shoes that make you dance\n\nStyle tip: Happy days are perfect for bold choices! ✨"
                ],
                "sad": [
                    "I hear you're feeling down. Let's boost your mood with comfort and style:\n\n• Gentle choices:\n- Soft, cozy fabrics\n- Your favorite comfort pieces\n- Protective layers\n\n• Mood-lifting elements:\n- A pop of your favorite color\n- Your lucky charm accessory\n- Comfortable shoes\n\nRemember: It's okay not to be okay. Let's find something that makes you feel good 💕",
                    "On down days, let's focus on gentle comfort:\n\n• Soothing choices:\n- Soft, relaxed fits\n- Calming blues or warm neutrals\n- Your most reliable pieces\n\n• Kind touches:\n- Extra-soft fabrics\n- Comforting layers\n- Simple accessories\n\nTip: Wear something that holds happy memories 🌸"
                ]
            }
        },
        "workout_attire": {
            "patterns": ["gym", "workout", "exercise", "fitness", "training", "sports", "athletic wear"],
            "responses": [
                "Let's get you ready for an awesome workout!\n\n• Essential gear:\n- Moisture-wicking tops and bottoms\n- Supportive athletic shoes\n- Performance socks\n- Breathable layers\n\n• Activity-specific tips:\n- Running: Lightweight, reflective gear\n- Yoga: Stretchy, fitted clothes\n- Weights: Supportive, flexible wear\n\nPro tip: Pack a fresh change of clothes! 💪",
                "Time to dress for success in your workout!\n\n• Must-haves:\n- Supportive sports bra (if needed)\n- Quick-dry, breathable top\n- Comfortable, secure bottoms\n- Proper athletic shoes\n\n• Extra tips:\n- Layer up for outdoor sessions\n- Bring a water bottle\n- Pack a small towel\n\nRemember: Comfort enables performance! 🏃‍♀️"
            ]
        },
        "color_matching": {
            "patterns": ["color combination", "what colors", "color match", "colors together", "color coordination"],
            "responses": [
                "Let's create some beautiful color combinations!\n\n• Classic pairings:\n- Navy and white\n- Black and beige\n- Gray and burgundy\n\n• Bold combinations:\n- Yellow and blue\n- Purple and green\n- Orange and navy\n\nPro tip: Use the 60-30-10 rule: 60% main color, 30% secondary, 10% accent! 🎨",
                "Here are some foolproof color combinations:\n\n• Safe bets:\n- Monochromatic looks\n- Neutral combinations\n- Complementary colors\n\n• Style tips:\n- Add texture for interest\n- Use accessories for pops of color\n- Consider the season\n\nRemember: When in doubt, keep it simple! 🌈"
            ]
        }
    }

def get_mood_response(message):
    response_data = get_response_data()
    mood_mapping = {
        "happy": ["happy", "joyful", "excited", "great"],
        "sad": ["sad", "down", "depressed", "blue", "unhappy"],
        "energetic": ["energetic", "energy", "active", "pumped"],
        "relaxed": ["relaxed", "chill", "calm", "peaceful"],
        "tired": ["tired", "exhausted", "low energy", "no energy"],
        "confident": ["confident", "powerful", "boss", "strong"]
    }
    
    for mood, keywords in mood_mapping.items():
        if any(re.search(r'\b' + re.escape(word) + r'\b', message) for word in keywords):
            mood_responses = response_data.get("mood_styling", {}).get("responses", {}).get(mood)
            if mood_responses:
                return random.choice(mood_responses)
            else:
                return "Let me help you find the perfect outfit to match your mood! How are you feeling today? 😊"
    
    return "Tell me more about how you're feeling, and I'll suggest the perfect outfit to match or lift your mood!"

def get_wedding_response(message):
    response_data = get_response_data()
    # Check for specific wedding types using regex for accuracy
    if re.search(r'\b(indian|hindu)\b', message):  
        return response_data.get("cultural_weddings", {}).get("responses", {}).get("indian_hindu", "")
    elif re.search(r'\bchristian\b', message):
        return response_data.get("cultural_weddings", {}).get("responses", {}).get("christian", "")
    return response_data.get("cultural_weddings", {}).get("responses", {}).get("general", "")

def get_response(user_message):
    response_data = get_response_data()
    message = user_message.lower().strip()
    
    # 1. Greetings
    for pattern in response_data["greetings"]["patterns"]:
        if re.search(r'\b' + re.escape(pattern) + r'\b', message):
            return random.choice(response_data["greetings"]["responses"])
    
    # 2. Bot Identity
    for pattern in response_data["bot_identity"]["patterns"]:
        if re.search(r'\b' + re.escape(pattern) + r'\b', message):
            return random.choice(response_data["bot_identity"]["responses"])
    
    # 3. App Info
    for pattern in response_data["app_info"]["patterns"]:
        if re.search(r'\b' + re.escape(pattern) + r'\b', message):
            return random.choice(response_data["app_info"]["responses"])
    
    # 4. Mood Styling queries
    if any(re.search(r'\b' + re.escape(pattern) + r'\b', message) for pattern in response_data["mood_styling"]["patterns"]):
        return get_mood_response(message)
    
    # 5. Wedding queries (cultural weddings)
    if re.search(r'\b(wedding|marriage|ceremony)\b', message):
        wedding_response = get_wedding_response(message)
        if wedding_response:
            return wedding_response
    
    # 6. Daily Outfit queries
    if any(re.search(r'\b' + re.escape(pattern) + r'\b', message) for pattern in response_data["daily_outfit"]["patterns"]):
        return random.choice(response_data["daily_outfit"]["responses"])
    
    # 7. Weather Styling queries
    if any(re.search(r'\b' + re.escape(pattern) + r'\b', message) for pattern in response_data["weather_styling"]["patterns"]):
        for weather, responses in response_data["weather_styling"]["responses"].items():
            if re.search(r'\b' + re.escape(weather) + r'\b', message):
                return random.choice(responses)
    
    # 8. Religious Venues queries
    if any(re.search(r'\b' + re.escape(pattern) + r'\b', message) for pattern in response_data["religious_venues"]["patterns"]):
        return random.choice(response_data["religious_venues"]["responses"])
    
    # 9. Workout Attire queries
    if any(re.search(r'\b' + re.escape(pattern) + r'\b', message) for pattern in response_data["workout_attire"]["patterns"]):
        return random.choice(response_data["workout_attire"]["responses"])
    
    # 10. Color Matching queries
    if any(re.search(r'\b' + re.escape(pattern) + r'\b', message) for pattern in response_data["color_matching"]["patterns"]):
        return random.choice(response_data["color_matching"]["responses"])
    
    # Default response if no match is found
    default_message = ("""I'd love to help you with your style questions! You can ask me about:

• Daily outfit suggestions
• Mood-based styling
• Religious and cultural dress codes
• Wedding guest attire
• Workout wear
• Color combinations
• Special occasions

What would you like to know more about? 💫""")
    return default_message

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    
    response = get_response(data["message"])
    return jsonify({"response": response})
# ----------------------------
# Recommendation (Survey) Part
# ----------------------------
df = pd.read_csv("dataset.csv")
if "label" in df.columns:
    df = df.drop(columns=["label"])

categorical_cols = ["gender", "mood", "weather", "color", "occasion"]
encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
for col, encoder in encoders.items():
    df[col] = encoder.transform(df[col])

scaler = StandardScaler()
df[["height", "weight"]] = scaler.fit_transform(df[["height", "weight"]])

features = ["gender", "height", "weight", "mood", "weather", "color", "occasion"]
knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
knn.fit(df[features])

def recommend_images(user_input):
    input_df = pd.DataFrame([user_input])
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform([user_input[col]])
    input_df[["height", "weight"]] = scaler.transform([[user_input["height"], user_input["weight"]]])
    _, idx = knn.kneighbors(input_df[features])
    # Return list of image paths (if dataset contains fewer than 3, list will be shorter)
    image_paths = df.iloc[idx[0]]["image_path"].tolist()
    return image_paths

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = {
            "gender": request.form["gender"],
            "height": float(request.form["height"]),
            "weight": float(request.form["weight"]),
            "mood": request.form["mood"],
            "weather": request.form["weather"],
            "color": request.form["color"],
            "occasion": request.form["occasion"]
        }
        image_paths = recommend_images(user_input)
        session["image_paths"] = image_paths
        # Also set a default selected image
        if image_paths:
            session["image_path"] = image_paths[0]
        return render_template("index.html", images=image_paths)
    return render_template("index.html", images=None)

# ----------------------------
# Select Dress Points Page
# ----------------------------
@app.route("/select_points", methods=["GET", "POST"])
def select_points():
    # Normal version: simply use session["image_path"]
    outfit_path = session.get("image_path")
    if outfit_path is None:
        return redirect(url_for("index"))
    return render_template("select_points.html", image=outfit_path)
    
@app.route("/save_points", methods=["POST"])
def save_points():
    data = request.get_json()
    # Expect exactly 4 points for: Left Shoulder, Right Shoulder, Left Hip, Right Hip.
    if data and "points" in data and len(data["points"]) == 4:
        session["dress_points"] = data["points"]
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "4 points are required."}), 400

# ----------------------------
# New route: Update selected image before moving to select points
# ----------------------------
@app.route("/update_image", methods=["GET"])
def update_image():
    selected_image = request.args.get("selected")
    if selected_image:
        session["image_path"] = selected_image
    return redirect(url_for("select_points"))

# ----------------------------
# AR Try-On (Live Webcam) Part
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def overlay_dress(frame, landmarks, dress_img, dress_points):
    """
    Overlays the dress image onto the frame using a perspective transform based on 4 points.
    dress_points: list of 4 points (in percentages) selected on the dress image corresponding to:
       [Left Shoulder, Right Shoulder, Left Hip, Right Hip]
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Obtain target landmarks from mediapipe.
    target_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value
    ]
    target_pts = []
    for idx in target_indices:
        lm = landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        target_pts.append([x, y])
    target_pts = np.float32(target_pts)
    
    # Convert the dress image points (percentages) to absolute coordinates.
    dress_h, dress_w = dress_img.shape[:2]
    src_pts = []
    for pt in dress_points:
        src_pts.append([int(pt[0] * dress_w), int(pt[1] * dress_h)])
    src_pts = np.float32(src_pts)
    
    if src_pts.shape[0] != 4 or target_pts.shape[0] != 4:
        return frame

    M = cv2.getPerspectiveTransform(src_pts, target_pts)
    warped_dress = cv2.warpPerspective(dress_img, M, (frame_w, frame_h), borderMode=cv2.BORDER_TRANSPARENT)
    
    output = frame.copy()
    if warped_dress.shape[2] == 4:
        alpha = warped_dress[:, :, 3] / 255.0
        for c in range(3):
            output[:, :, c] = (1 - alpha) * output[:, :, c] + alpha * warped_dress[:, :, c]
    else:
        output = cv2.addWeighted(output, 1, warped_dress, 1, 0)
    
    return output

def generate_frames(dress_img=None, dress_points=None):
    global last_frame
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            # Hide the skeleton by commenting out the next line.
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if dress_img is not None and dress_points is not None:
                frame = overlay_dress(frame, results.pose_landmarks.landmark, dress_img, dress_points)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        last_frame = frame  # Store the last processed frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route("/tryon")
def tryon():
    if "dress_points" not in session:
        return redirect(url_for("select_points"))
    return render_template("tryon.html")

@app.route("/video_feed")
def video_feed():
    capture = request.args.get("capture", "false").lower() == "true"
    outfit_path = session.get("image_path")
    if outfit_path is None:
        outfit_path = "static/images/default.png"
    outfit_full_path = os.path.join(app.root_path, outfit_path)
    dress_img = cv2.imread(outfit_full_path, cv2.IMREAD_UNCHANGED)
    if dress_img is not None and dress_img.shape[2] == 3:
        dress_img = cv2.cvtColor(dress_img, cv2.COLOR_BGR2BGRA)
    dress_points = session.get("dress_points")
    if capture:
        global last_frame
        if last_frame is not None:
            encoded = base64.b64encode(last_frame).decode('utf-8')
            return jsonify({"status": "captured", "image": encoded})
        else:
            return jsonify({"status": "error", "message": "No frame captured yet."}), 500
    return Response(generate_frames(dress_img, dress_points), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize the global variable
    last_frame = None
    app.run(debug=True)

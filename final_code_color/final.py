import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import webcolors
import streamlit as st
import os
import tempfile
import numpy as np
import base64
from passlib.hash import pbkdf2_sha256
loaded_model = load_model('model.h5')

import sqlite3

def home_page():
    st.title("Welcome to the Color Based Emotion Detection System")
    st.write("This application uses advanced CNN techniques to detect color in images,videos")
    st.write("Please login or signup to use the system.")

    # You can add more widgets or information as needed

def create_connection(db_file):
    """ Create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
    return conn
def create_table(conn):
    """ Create a table for storing user data """
    try:
        sql = '''CREATE TABLE IF NOT EXISTS users (
                    username text PRIMARY KEY,
                    password text NOT NULL
                 );'''
        conn.execute(sql)
    except Exception as e:
        print(e)



# Function to set the background image
def set_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background_image('basic.jpg')

# Initialize session state for user authentication
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# User database simulation (in-memory)
users = {}

# Initialize session state for user data if not already present
if 'users' not in st.session_state:
    st.session_state['users'] = {}

def signup(username, password, conn):
    """ Sign up a new user """
    try:
        hashed_password = pbkdf2_sha256.hash(password)
        sql = ''' INSERT INTO users(username,password)
                  VALUES(?,?) '''
        cur = conn.cursor()
        cur.execute(sql, (username, hashed_password))
        conn.commit()
        return True
    except Exception as e:
        print(e)
        return False

def login(username, password):
    st.text(f"Debug: Users currently in system: {st.session_state['users']}")  # For debugging
    if username in st.session_state['users'] and pbkdf2_sha256.verify(password, st.session_state['users'][username]):
        st.session_state['logged_in'] = True
        return True
    return False

# Initialize database connection
db_file = 'your_database.db'
conn = create_connection(db_file)
create_table(conn)

# # Modify your existing signup form function
# def signup_form():
#     with st.form("signup"):
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         submit = st.form_submit_button("Signup")

#         if submit:
#             if signup(username, password, conn):
#                 st.success("Signup successful!")
#             else:
#                 st.error("Username already exists or an error occurred.")
def signup_form():
    with st.form("signup"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Signup")

        if submit:
            if signup(username, password, conn):  # Assuming signup() returns True on success
                st.success("Signup successful! Please login.")
                st.session_state['page'] = 'Login'  # Redirect to login page
            else:
                st.error("Username already exists or an error occurred.")



def validate_login(username, password, conn):
    """ Validate login credentials """
    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        user_data = cur.fetchone()
        if user_data:
            stored_password = user_data[0]
            return pbkdf2_sha256.verify(password, stored_password)
        return False
    except Exception as e:
        print(e)
        return False
def login_form():
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if validate_login(username, password, conn):
                st.session_state['logged_in'] = True  # Set logged_in to True
                st.success("Logged in successfully!")
            else:
                st.error("Incorrect username or password.")


emotions = {
    'black': ['Powerful', 'sophisticated', 'edgy'],
    'firebrick': ['Passionate', 'aggressive', 'important'],
    'gold': ['Opulent', 'traditional', 'prestigious'],
    'cadetblue': ['Sleek', 'graceful', 'futuristic'],
    'turquoise': ['Refreshing', 'tranquil', 'creative'],
    'lavender': ['Delicate', 'graceful', 'nostalgic'],
    'beige': ['Simplistic', 'dependable', 'conservative'],
    'palevioletred': ['Dynamic', 'bold', 'passionate'],
    'cornflowerblue': ['Imaginative', 'spirited', 'unique'],
    'darkslategray': ['Natural', 'peaceful', 'enduring'],
    'rosybrown': ['Warm', 'inviting', 'vibrant'],
    'darkolivegreen': ['Professional', 'reliable', 'authoritative'],
    'olivedrab': ['Energetic', 'lively', 'fresh'],
    'lightgray': ['Solid', 'professional', 'mature'],
    'steelblue': ['Fresh', 'cool', 'youthful'],
    'darkgray': ['Deep', 'wise', 'thoughtful'],
    'dimgray': ['Lush', 'vibrant', 'sophisticated'],
    'cadetblue': ['Earthy', 'warm', 'enduring'],
    'gray': ['Soft', 'friendly', 'approachable'],
    'slategray': ['Refreshing', 'serene', 'youthful'],
    'seagreen': ['peaceful', 'bold', 'youthful']
}

import sqlite3


    # You can add more widgets or information as needed

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(hex_value):
    try:
        color_name = webcolors.hex_to_name(hex_value)
    except ValueError:
        rgb_value = webcolors.hex_to_rgb(hex_value)
        color_name = closest_color(rgb_value)
    return color_name

def load_and_preprocess_image(image_path, img_size=(64, 64)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def find_dominant_color(image, k=1):
    reshaped_img = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_img)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return "#{:02x}{:02x}{:02x}".format(dominant_color[0], dominant_color[1], dominant_color[2])

def process_video(uploaded_file):
    # Create the 'uploads' directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    file_path = 'uploads/' + uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Open the video file
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (64, 64)) / 255.0
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

        # Make predictions for each frame
        predictions = loaded_model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions, axis=1)

        # Find the dominant color for each frame
        dominant_hex_color = find_dominant_color(frame)
        color = get_color_name(dominant_hex_color)
        main_emotion = emotions[color]

        # Display the frame and predictions
        st.image(frame, caption="Frame", use_column_width=True)
        st.write("Predicted Class:", predicted_class[0])
        st.write("Dominant Hex Color:", dominant_hex_color)
        st.write("Predicted Color:", color)
        st.write("Emotions:", main_emotion)

    cap.release()

def main():
    st.title("Upload Color Image/Video Upload and Analysis")

    uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"])
    if st.button('Logout'):
        st.session_state['logged_in'] = False
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            # Image processing
            file_path = 'uploads/' + uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            preprocessed_image = load_and_preprocess_image(file_path)
            predictions = loaded_model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions, axis=1)

            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            dominant_hex_color = find_dominant_color(original_image)
            color = get_color_name(dominant_hex_color)
            main_emotion = emotions[color]

            st.image(original_image, caption="Uploaded Image", use_column_width=True)
            st.write("Predicted Class:", predicted_class[0])
            st.write("Dominant Hex Color:", dominant_hex_color)
            st.write("Predicted Color:", color)
            st.write("Emotions:", main_emotion)

        elif file_extension in ['mp4', 'avi', 'mkv']:
            # Video processing
            process_video(uploaded_file)

# if __name__ == "__main__":
#     main()
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'  # Default page

# App routing
if not st.session_state.get('logged_in'):  # Check if user is not logged in
    st.sidebar.title("Navigation")
    # Use session state for controlling the current page
    option = st.sidebar.radio("Choose an option", ["Home", "Login", "Signup"], index=["Home", "Login", "Signup"].index(st.session_state['page']))

    if option == "Home":
        home_page()
    elif option == "Signup":
        signup_form()
    else:  # Login
        login_form()
else:
    main() 
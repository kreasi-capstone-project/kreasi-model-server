from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessing components
model = load_model('my_model.h5')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def classify_level(score):
    if score == 0:
        return "Beginner"
    elif 1 <= score <= 3:
        return "Intermediate"
    elif 4 <= score <= 6:
        return "Mixed"
    else:
        return "Advanced"

def predict_course(score, skill):
    # Classify level based on score
    level = classify_level(score)

    # Combine level and skill into a single feature
    combined_features = f"{level} {skill}"

    # Vectorize the combined features
    combined_features_vectorized = vectorizer.transform([combined_features]).toarray()

    # Create skill presence features
    top_skills = [
        "leadership and management", "data analysis", "strategy",
        "strategy and operations", "critical thinking", "problem solving",
        "communication", "computer programming", "business analysis", "decision making"
    ]
    skill_features = np.array([1 if top_skill in skill.lower() else 0 for top_skill in top_skills]).reshape(1, -1)

    # Combine vectorized and skill presence features
    X_new = np.hstack([combined_features_vectorized, skill_features])

    # Scale the input data
    X_new_scaled = scaler.transform(X_new)

    # Predict probabilities
    predicted_probabilities = model.predict(X_new_scaled)

    # Get top predictions
    top_n_indices = np.argsort(predicted_probabilities[0])[::-1]
    predicted_courses = label_encoder.inverse_transform(top_n_indices)

    # Return results
    return level, predicted_courses[:5]  # Top 5 predictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        score = data.get('score')
        skill = data.get('skill')

        # Validate input
        if score is None or skill is None:
            return jsonify({"error": "Invalid input. 'score' and 'skill' are required."}), 400

        # Make prediction
        user_level, predicted_courses = predict_course(score, skill)

        # Return response
        return jsonify({
            "user_level": user_level,
            "predicted_courses": list(predicted_courses)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("./models/cardiovision_model_v1.keras")
print("Model loaded successfully!")
model.summary()
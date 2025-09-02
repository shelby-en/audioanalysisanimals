import librosa
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# Load the fine-tuned model and feature extractor
model_name = "ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals"
model_path = "./data/chkpts/pretrain/"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = HubertForSequenceClassification.from_pretrained(model_path)

# Prepare the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode

# Function to predict the class of an audio file
def predict_audio_class(audio_file, feature_extractor, model, device):
    # Load and preprocess the audio file
    speech, sr = librosa.load(audio_file, sr=48000)
    input_values = feature_extractor(speech, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_values).logits

    # Get the predicted class ID
    predicted_id = torch.argmax(logits, dim=-1)
    # Convert the predicted ID to the class name
    predicted_class = model.config.id2label[predicted_id.item()]
    
    return predicted_class

# Replace 'path_to_your_new_audio_file.wav' with the actual path to the new audio file
audio_file_path = "./data/augmented_mp3s/DuckA_3.wav"
predicted_class = predict_audio_class(audio_file_path, feature_extractor, model, device)
print(f"Predicted class: {predicted_class}")

model.save_pretrained(model_path)

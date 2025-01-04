import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def generate_caption(caption_model, tokenizer, feature_extractor, img, max_length):
    """
    Generate a caption for a given image.
    
    Parameters:
        caption_model: The trained captioning model.
        tokenizer: The tokenizer used during training.
        feature_extractor: The feature extractor model.
        image_path: Path to the input image.
        max_length: Maximum length of the caption.
        
    Returns:
        str: Generated caption.
    """
    # Step 1: Extract image features using the feature extractor
      # Load and resize the image
    img = img_to_array(img) / 255.0  # Convert to array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    feature = feature_extractor.predict(img, verbose=0)  # Extract features

    # Step 2: Generate the caption using the caption model
    sequence = tokenizer.texts_to_sequences(["startseq"])[0]  # Start with the start token
    for _ in range(max_length):
        padded_sequence = pad_sequences([sequence], maxlen=max_length)  # Pad the sequence
        y_pred = caption_model.predict([feature, padded_sequence], verbose=0)  # Predict the next word
        next_word_index = np.argmax(y_pred)  # Get the word index with highest probability
        next_word = tokenizer.index_word.get(next_word_index)  # Map index to word

        if next_word is None or next_word == "endseq":  # Stop if end token is generated
            break

        sequence.append(next_word_index)  # Add word index to sequence

    # Remove startseq and endseq tokens for the final caption
    caption = " ".join([tokenizer.index_word[idx] for idx in sequence 
                        if idx > 0 and idx not in [tokenizer.word_index['startseq'], tokenizer.word_index['endseq']]])
    return caption

def load_models():
    model = tf.keras.models.load_model(r'artifacts\model.keras')
    fe = tf.keras.models.load_model(r'artifacts\fe.keras')
    max_length = 34


    with open(r'artifacts\tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    
    return model,fe,max_length,tokenizer
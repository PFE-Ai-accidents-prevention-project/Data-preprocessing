import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv("your_dataset_augmented_for_DL_less.csv", encoding='latin1')
    
    # Create X and y
    y = df['Title_Incident_Report']
    X = df.drop('Title_Incident_Report', axis=1)
    
    return X, y, df

# 2. Feature Engineering
def feature_engineering(X, y, df):
    # Count samples per class
    class_counts = y.value_counts()
    min_samples = 2  # Minimum samples per class
    valid_classes = class_counts[class_counts >= min_samples].index
    
    # Filter data to keep only classes with sufficient samples
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    df = df[mask]
    
    print(f"Removed {len(class_counts) - len(valid_classes)} classes with fewer than {min_samples} samples")
    print(f"Remaining classes: {len(valid_classes)}")
    
    # Continue with existing code
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Separate categorical and numerical features
    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(exclude=['object']).columns
    
    # Handle categorical features
    categorical_data = {}
    for feature in cat_features:
        # Skip Date_fix as it requires special handling
        if feature == 'Date_fix':
            continue
            
        le = LabelEncoder()
        X[feature] = X[feature].fillna('Unknown')
        X[feature] = le.fit_transform(X[feature])
        categorical_data[feature] = {
            'encoder': le,
            'values': X[feature].values,
            'vocab_size': len(le.classes_)
        }
    
    # Handle numerical features
    scaler = StandardScaler()
    if len(num_features) > 0:
        X[num_features] = scaler.fit_transform(X[num_features].fillna(0))
    
    # Handle date feature
    # In feature_engineering function, update the date parsing:
    if 'Date_fix' in X.columns:
         X['Date_fix'] = pd.to_datetime(X['Date_fix'], 
                                  format='%d/%m/%Y',  # Specify the format explicitly
                                  errors='coerce')
         X['Year'] = X['Date_fix'].dt.year
         X['Month'] = X['Date_fix'].dt.month
         X['Day'] = X['Date_fix'].dt.day
         X = X.drop('Date_fix', axis=1)
    
    # Create text features (combine categorical text fields for NLP)
    text_columns = ['Department_Area', 'Incident_Type', 'Process', 'Operation_Mode', 
                    'Machine_Category', 'Machine_Type', 'Type_Of_Injury', 'Injured_Body_Part']
    
    text_data = []
    for _, row in df.iterrows():
        text = ' '.join([str(row[col]) for col in text_columns if col in df.columns])
        text_data.append(text)
    
    # Tokenize text
    max_words = 5000
    max_sequence_length = 100
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    text_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return X, y_encoded, categorical_data, text_sequences, label_encoder, max_words, max_sequence_length

# 3. Build VGG-inspired CNN model
def build_vgg_model(categorical_data, max_words, max_sequence_length, num_classes):
    # Text input branch (VGG-inspired)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    embedding_dim = 128
    embedding_layer = Embedding(
        input_dim=max_words,
        output_dim=embedding_dim
    )(text_input)
    
    # VGG-style: Multiple Conv1D blocks with increasing filters
    # Block 1
    x = Conv1D(64, 3, activation='relu', padding='same', name='block1_conv1')(embedding_layer)
    x = Conv1D(64, 3, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling1D(2, name='block1_pool')(x)
    
    # Block 2
    x = Conv1D(128, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv1D(128, 3, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling1D(2, name='block2_pool')(x)
    
    # Block 3
    x = Conv1D(256, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv1D(256, 3, activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv1D(256, 3, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling1D(2, name='block3_pool')(x)
    
    # Flatten text branch - updated name
    text_features = Flatten(name='text_flatten')(x)
    
    # Categorical inputs branch
    categorical_inputs = []
    categorical_embeddings = []
    
    for feature, data in categorical_data.items():
        # Create valid layer name by replacing spaces and special characters
        safe_feature_name = feature.replace(' ', '_').replace('-', '_')
        
        input_layer = Input(shape=(1,), name=f'{safe_feature_name}_input')
        categorical_inputs.append(input_layer)
        
        # Embedding for categorical variables
        embedding_size = min(50, (data['vocab_size'] + 1) // 2)
        embedding = Embedding(
            input_dim=data['vocab_size'],
            output_dim=embedding_size
        )(input_layer)
        # Updated name for categorical flatten layers with safe feature name
        embedding = Flatten(name=f'{safe_feature_name}_flatten')(embedding)
        categorical_embeddings.append(embedding)
    
    # Combine all features
    if categorical_embeddings:
        combined_categorical = Concatenate()(categorical_embeddings)
        combined_features = Concatenate()([text_features, combined_categorical])
    else:
        combined_features = text_features
    
    # Dense layers (VGG-style: Multiple dense with dropout)
    x = Dense(512, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    all_inputs = [text_input] + categorical_inputs
    model = Model(inputs=all_inputs, outputs=output)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer=Adam(learning_rate=0.001),
                 metrics=['accuracy'])
    
    return model

# 4. Train and evaluate model
def train_and_evaluate(model, X, text_sequences, categorical_data, y, label_encoder):
    # Check if we have enough samples
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        raise ValueError("Each class must have at least 2 samples for train/test split")
    
    # Split data
    X_train, X_test, text_train, text_test, y_train, y_test = train_test_split(
        X, text_sequences, y, test_size=0.2, random_state=42, stratify=y)
    
    # Prepare inputs
    train_inputs = [text_train]
    test_inputs = [text_test]
    
    for feature, data in categorical_data.items():
        train_inputs.append(X_train[feature].values.reshape(-1, 1))
        test_inputs.append(X_test[feature].values.reshape(-1, 1))
     # Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    # Updated ModelCheckpoint with .keras extension
    model_checkpoint = ModelCheckpoint(
        'best_vgg_model.keras',  # Changed from .h5 to .keras
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max'
    )
    
    # Train model
    history = model.fit(
        train_inputs, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Load the best model
    loaded_model = load_model('best_vgg_model.keras')
    
    y_train_pred_proba = loaded_model.predict(train_inputs)  # Predict probabilities for training data
    y_train_pred = np.argmax(y_train_pred_proba, axis=1) 

    y_test_pred_proba = loaded_model.predict(test_inputs)  # Predict probabilities for test data
    y_test_pred = np.argmax(y_test_pred_proba, axis=1) 
    
    
    # Convert predictions back to original labels
    y_train_pred_labels = label_encoder.inverse_transform(y_train_pred)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Training Precision: {train_precision*100:.2f}%")
    print(f"Training Recall: {train_recall*100:.2f}%")
    print(f"Training F1 Score: {train_f1*100:.2f}%")
    
    
# Testing metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
    print(f"Testing Precision: {test_precision*100:.2f}%")
    print(f"Testing Recall: {test_recall*100:.2f}%")
    print(f"Testing F1 Score: {test_f1*100:.2f}%")
   

    #print("\nClassification Report:")
    #print(classification_report(y_test_labels, y_pred_labels))
    
    return model, history, y_train_pred, y_test_pred

# 5. Main function
def main():
    # Load data
    file_path = 'your_dataset_augmented_for_DL_less.csv'  # Replace with your actual file path
    X, y, df = load_and_preprocess_data(file_path)
    
    # Feature engineering
    X, y_encoded, categorical_data, text_sequences, label_encoder, max_words, max_sequence_length = feature_engineering(X, y, df)
    
    # Build model
    num_classes = len(np.unique(y_encoded))
    model = build_vgg_model(categorical_data, max_words, max_sequence_length, num_classes)
    
    # Train and evaluate
    model, history, predictions = train_and_evaluate(model, X, text_sequences, categorical_data, y_encoded, label_encoder)
    
    # Summary
    print(f"Model trained successfully with {len(np.unique(y_encoded))} unique incident title classes")
    
    return model, label_encoder

if __name__ == "__main__":
    model, label_encoder = main()
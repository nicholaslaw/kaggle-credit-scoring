ECHO "PREPROCESSING"
python preprocess.py
ECHO "TRAINING MODEL"
python train.py
ECHO "GENERATING PREDICTIONS"
python generate_predictions.py
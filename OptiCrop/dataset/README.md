This folder contains a sample crop recommendation dataset and instructions for training the AI model.

Required columns:
- `N`
- `P`
- `K`
- `temperature`
- `humidity`
- `ph`
- `rainfall`
- `label`

The sample file `crop_recommendation_sample.csv` includes realistic training rows for common crops such as rice, maize, wheat, cotton, soybean, and sugarcane.

To train the model:

```powershell
python train_model.py
```

To use your own dataset, copy it to `dataset/crop_recommendation.csv` and ensure the same column names are present.

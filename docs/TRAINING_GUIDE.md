# Panduan Training Neural Network

## Fitur yang Telah Diimplementasikan

### 1. Backpropagation
- Implementasi lengkap algoritma backpropagation
- Mendukung berbagai activation functions (sigmoid, ReLU, linear)
- Mendukung berbagai loss functions (MSE, Binary Cross-Entropy)

### 2. Optimizer
Tersedia 2 optimizer:
- **Gradient Descent (GD)**: Full batch gradient descent
- **SGD**: Stochastic Gradient Descent dengan dukungan mini-batch

### 3. Training Method
Method `train()` pada class `NeuralNetwork` dengan parameter:
- `X`: Training features (numpy array)
- `y`: Training labels (numpy array)
- `epochs`: Jumlah epoch (default: 100)
- `learning_rate`: Learning rate (default: 0.01)
- `optimizer`: Pilih 'gd' atau 'sgd' (default: 'gd')
- `loss_function`: Pilih 'mse' atau 'binary' (default: 'mse')
- `batch_size`: Batch size untuk SGD (default: None = full batch)
- `verbose`: Print progress (default: True)

### 4. Flask API Endpoint
Endpoint baru: `POST /train`

**Request Body:**
```json
{
  "dataset": "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0",
  "epochs": 1000,
  "learning_rate": 0.5,
  "optimizer": "gd",
  "loss_function": "mse",
  "batch_size": null
}
```

**Response:**
```json
{
  "success": true,
  "message": "Training completed: 1000 epochs",
  "history": {
    "epochs": [1, 2, 3, ...],
    "loss": [0.25, 0.24, 0.23, ...]
  },
  "final_loss": 0.103,
  "accuracy": 0.75,
  "updated_weights": {
    "layer_1": [[...], [...]],
    "layer_2": [[...]]
  },
  "updated_biases": {
    "layer_1": [...],
    "layer_2": [...]
  },
  "predictions": {
    "y_true": [0, 0, 0, 1],
    "y_pred": [0.095, 0.244, 0.278, 0.484],
    "y_pred_classes": [0, 0, 0, 0]
  }
}
```

## Cara Penggunaan

### 1. Menggunakan Python Langsung

```python
import numpy as np
from ann_core import NeuralNetwork

# Buat network
nn = NeuralNetwork()
nn.add_layer(2, 'linear')   # Input
nn.add_layer(2, 'sigmoid')  # Hidden
nn.add_layer(1, 'sigmoid')  # Output

# Set connections
nn.set_connections(1, [[0, 1], [0, 1]],
                   [[0.5, -0.3], [-0.4, 0.6]],
                   [0.1, -0.2])
nn.set_connections(2, [[0, 1]],
                   [[0.8, -0.5]],
                   [0.2])

# Prepare data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Train
history = nn.train(
    X, y,
    epochs=1000,
    learning_rate=0.5,
    optimizer='gd',
    loss_function='mse'
)

# Make predictions
predictions = nn.forward(X)
print(predictions)
```

### 2. Menggunakan Flask API

Lihat file `example_api_training.py` untuk contoh lengkap.

**Langkah-langkah:**
1. Start Flask server: `python app.py`
2. Build network: `POST /build_network`
3. Train network: `POST /train`
4. Lihat hasil di response

## Output Training

Setelah training selesai, Anda akan mendapatkan:

### 1. Bobot & Bias Baru
Semua weights dan biases telah diupdate melalui backpropagation.

**Contoh:**
```
Layer 1 (Input -> Hidden):
  Node 0: W=['-1.839', '-0.745'], b=0.749
  Node 1: W=['-0.460', '-1.694'], b=0.479

Layer 2 (Hidden -> Output):
  Node 0: W=['-2.593', '-1.709'], b=0.562
```

### 2. Kurva Loss Per Epoch
History berisi loss untuk setiap epoch, bisa digunakan untuk visualisasi.

**Contoh:**
```
Epoch   Loss
1       0.198486
2       0.197875
...
1000    0.103236
```

### 3. Hasil Prediksi Baru
Prediksi model setelah training dengan akurasi yang lebih baik.

**Contoh:**
```
Input     Target  Prediksi  Class  Status
[0 0]     0       0.0950    0      BENAR
[0 1]     0       0.2445    0      BENAR
[1 0]     0       0.2783    0      BENAR
[1 1]     1       0.4839    0      SALAH
```

## File Testing & Demo

1. **test_training_simple.py**: Test sederhana untuk verifikasi implementasi
2. **demo_training.py**: Demo lengkap dengan output yang mudah dibaca
3. **example_api_training.py**: Contoh penggunaan via Flask API

## Tips Penggunaan

### Memilih Optimizer
- **Gradient Descent (GD)**: Lebih stabil, konvergen smooth, cocok untuk dataset kecil
- **SGD**: Lebih cepat, bisa escape local minima, cocok untuk dataset besar

### Memilih Learning Rate
- Terlalu besar: Training tidak konvergen, loss naik-turun
- Terlalu kecil: Training lambat
- Recommended: 0.01 - 1.0 (tergantung problem)

### Memilih Epochs
- Monitor loss curve
- Stop jika loss sudah plateau (tidak turun lagi)
- Typical: 100 - 10000 epochs

### Memilih Loss Function
- **MSE**: Untuk regression atau output continuous
- **Binary Cross-Entropy**: Untuk binary classification

## Troubleshooting

### Loss tidak turun
- Coba tingkatkan learning rate
- Coba tambah epochs
- Periksa apakah data sudah dinormalisasi

### Loss naik-turun drastis
- Kurangi learning rate
- Coba optimizer yang berbeda

### Accuracy rendah
- Tambah hidden layers/nodes
- Tambah epochs
- Coba learning rate berbeda
- Periksa apakah problem bisa dipelajari (e.g., XOR butuh hidden layer)

## Contoh Dataset

### AND Gate
```csv
x1,x2,y
0,0,0
0,1,0
1,0,0
1,1,1
```

### OR Gate
```csv
x1,x2,y
0,0,0
0,1,1
1,0,1
1,1,1
```

### XOR Gate (butuh hidden layer!)
```csv
x1,x2,y
0,0,0
0,1,1
1,0,1
1,1,0
```

## Roadmap Future Improvements

- [ ] Momentum optimizer
- [ ] Adam optimizer
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Validation split
- [ ] Regularization (L1, L2)
- [ ] Dropout
- [ ] Batch normalization

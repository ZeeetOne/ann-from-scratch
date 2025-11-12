# 🧠 ANN from Scratch - Interactive Neural Network Builder & Trainer

Program interaktif berbasis web untuk membangun, melatih, dan menguji Artificial Neural Network (ANN) dari awal menggunakan Python dengan **Backpropagation Training** lengkap!

## ✨ Fitur Utama

### Network Configuration
- **Konfigurasi Network Fleksibel**: Tentukan jumlah layer, jumlah node per layer
- **Custom Connections**: Tentukan koneksi antar node secara manual
- **Custom Weights & Biases**: Set bobot dan bias untuk setiap koneksi
- **Multiple Activation Functions**:
  - Sigmoid
  - ReLU
  - Threshold
  - Linear

### Prediction & Evaluation
- **Multiple Loss Functions**:
  - Binary Cross-Entropy
  - Mean Squared Error (MSE)
- **Dataset Upload**: Input dataset dalam format CSV
- **Real-time Predictions**: Lihat hasil prediksi dan metrik performa

### ⭐ Training dengan Backpropagation (NEW!)
- **Backpropagation Algorithm**: Implementasi lengkap dari nol
- **Multiple Optimizers**:
  - **Gradient Descent (GD)**: Full batch, konvergen stabil
  - **Stochastic Gradient Descent (SGD)**: Mini-batch support, lebih cepat
- **Hyperparameter Control**:
  - Learning Rate (0.001 - 10.0)
  - Epochs (10 - 10000)
  - Batch Size (untuk SGD)
- **Training Visualization**:
  - Loss curve (grafik SVG interaktif)
  - Training metrics (time, final loss, accuracy)
  - Updated weights & biases setelah training
  - Predictions comparison before/after training

### User Interface
- **Interactive Web UI**: Antarmuka web yang modern dan user-friendly
- **Real-time Results**: Lihat hasil training secara langsung
- **Responsive Design**: Mobile-friendly

## Instalasi

### 1. Clone atau Download Repository

```bash
cd ann-from-scratch
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

1. Jalankan aplikasi Flask:

```bash
python app.py
```

2. Buka browser dan akses:

```
http://localhost:5000
```

## Cara Menggunakan

### Quick Start (Cepat)

1. Klik tombol **"Load Example Network (3-4-2)"** untuk memuat network contoh
   - 3 input neurons: mewakili 3 fitur (suhu, tekanan, kelembapan)
   - 1 hidden layer dengan 4 neurons
   - 2 output neurons: mewakili 2 prediksi (hujan, cerah)
2. Klik tombol **"Load Example Dataset"** untuk memuat dataset contoh (10 samples dengan 3 features dan 2 outputs)
3. Klik **"Make Predictions"** untuk melihat hasil

### Custom Network (Manual)

#### Step 1: Konfigurasi Network

1. Masukkan jumlah layer (minimal 2)
2. Klik **"Generate Layer Configuration"**
3. Untuk setiap layer, tentukan:
   - Jumlah nodes
   - Activation function

#### Step 2: Set Connections & Weights

Untuk setiap layer (kecuali input layer):
- Tentukan node mana yang terhubung dari layer sebelumnya
- Set weight untuk setiap koneksi
- Set bias untuk setiap node

Contoh:
```
Connect to nodes: 0,1        (node ini menerima dari node 0 dan 1 di layer sebelumnya)
Weights: 0.5,0.3            (weight dari node 0 = 0.5, dari node 1 = 0.3)
Bias: 0.1                   (bias untuk node ini)
```

#### Step 3: Build Network

Klik **"Build Neural Network"** untuk membuat network dengan konfigurasi yang telah ditentukan.

#### Step 4: Input Dataset

1. Input dataset dalam format CSV dengan format:
   ```csv
   feature1,feature2,...,target
   2,8,1
   5,8,1
   3,10,1
   ```
   **Penting**: Kolom terakhir harus berisi target/label (y)

2. Pilih loss function (Binary atau MSE)
3. Set classification threshold (default: 0.5)

#### Step 5: Make Predictions

Klik **"Make Predictions"** untuk mendapatkan hasil prediksi.

#### Step 6: Train Neural Network ⭐ (NEW!)

Setelah membuild network dan input dataset, Anda bisa melatih network untuk meningkatkan akurasi:

1. **Pilih Optimizer**:
   - **Gradient Descent (GD)**: Konvergen stabil, cocok untuk dataset kecil
   - **SGD**: Update lebih sering, bisa escape local minima

2. **Set Hyperparameters**:
   - **Learning Rate**: 0.001 - 10.0 (recommended: 0.5 - 1.0)
   - **Epochs**: 10 - 10000 (recommended: 500 - 2000)
   - **Batch Size** (untuk SGD): Kosongkan untuk full batch, atau set angka (e.g., 2, 4, 8)

3. Klik **"Start Training"**

4. **Hasil Training**:
   - Training time
   - Final loss (setelah training)
   - Accuracy (% prediksi benar)
   - **Loss Curve**: Grafik SVG menunjukkan penurunan loss per epoch
   - **Updated Weights & Biases**: Bobot dan bias baru hasil training
   - **Predictions After Training**: Prediksi baru dengan model yang sudah dilatih

### Tips Hyperparameter Tuning

**Learning Rate:**
- Terlalu besar (>2.0): Training tidak stabil, loss naik-turun
- Optimal (0.1 - 1.0): Konvergen cepat dan stabil
- Terlalu kecil (<0.01): Training sangat lambat

**Epochs:**
- Monitor loss curve
- Stop jika loss sudah tidak turun lagi (plateau)
- Typical: 500 - 2000 epochs

**Optimizer:**
- **GD**: Lebih stabil, smooth convergence
- **SGD**: Lebih cepat, cocok untuk dataset besar

## Contoh Dataset

Dataset contoh yang sudah disediakan (prediksi cuaca):

```csv
suhu,tekanan,kelembapan,hujan,cerah
25,1010,85,1,0
30,1015,45,0,1
22,1005,90,1,0
28,1012,50,0,1
20,1000,95,1,0
32,1018,40,0,1
24,1008,80,1,0
29,1014,48,0,1
21,1003,88,1,0
31,1016,42,0,1
```

**Penjelasan:**
- **3 Features (Input)**: suhu (°C), tekanan (hPa), kelembapan (%)
- **2 Outputs**: hujan (1=ya, 0=tidak), cerah (1=ya, 0=tidak)
- **10 Samples**: Dataset untuk training dan testing

## Penggunaan via Python Script

Selain web interface, Anda juga bisa menggunakan `ann_core.py` langsung dari Python:

```python
import numpy as np
from ann_core import NeuralNetwork

# Buat network
nn = NeuralNetwork()
nn.add_layer(2, 'linear')   # Input layer
nn.add_layer(2, 'sigmoid')  # Hidden layer
nn.add_layer(1, 'sigmoid')  # Output layer

# Set connections
nn.set_connections(1, [[0, 1], [0, 1]],
                   [[0.5, -0.3], [-0.4, 0.6]],
                   [0.1, -0.2])
nn.set_connections(2, [[0, 1]],
                   [[0.8, -0.5]],
                   [0.2])

# Prepare data (AND gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Train dengan Gradient Descent
history = nn.train(
    X, y,
    epochs=1000,
    learning_rate=0.5,
    optimizer='gd',
    loss_function='mse'
)

# Predict
predictions = nn.forward(X)
print(predictions)

# Check loss history
print(f"Initial Loss: {history['loss'][0]}")
print(f"Final Loss: {history['loss'][-1]}")
```

Untuk contoh lengkap, jalankan:
```bash
python demo_training.py
```

## Struktur Project

```
ann-from-scratch/
│
├── app.py                    # Flask web application
├── ann_core.py              # Core ANN library (Network, Optimizers, Backprop)
├── requirements.txt         # Python dependencies
├── README.md               # Dokumentasi utama
├── TRAINING_GUIDE.md       # Panduan training lengkap
│
├── templates/
│   └── index.html          # Web UI (dengan training section)
│
├── static/
│   ├── css/
│   │   └── style.css       # Styling (dengan training styles)
│   └── js/
│       └── app.js          # Frontend JS (dengan training logic)
│
└── tests & demos/
    ├── demo_training.py          # Demo training interaktif
    ├── test_training_simple.py   # Test backpropagation
    ├── visualize_training.py     # Visualisasi loss curve
    └── example_api_training.py   # Contoh penggunaan API
```

## Cara Kerja

### Forward Propagation

1. Input data masuk ke input layer
2. Untuk setiap layer berikutnya:
   - Hitung weighted sum dari koneksi yang masuk
   - Tambahkan bias
   - Aplikasikan activation function
3. Output dari layer terakhir adalah prediksi (y-hat)

### Activation Functions

- **Sigmoid**: `f(x) = 1 / (1 + e^-x)` - Output antara 0 dan 1
- **ReLU**: `f(x) = max(0, x)` - Output 0 jika negatif, x jika positif
- **Threshold**: `f(x) = 1 if x > threshold else 0` - Binary output
- **Linear**: `f(x) = x` - Tidak ada transformasi (untuk input layer)

### Loss Functions

- **Binary Cross-Entropy**: Untuk klasifikasi binary
- **MSE (Mean Squared Error)**: Untuk regression atau klasifikasi

### Backpropagation & Training ⭐

1. **Forward Pass**: Hitung prediksi (ŷ) dari input
2. **Calculate Loss**: Hitung error antara prediksi dan actual
3. **Backward Pass** (Backpropagation):
   - Hitung gradient loss terhadap output layer
   - Propagate gradient ke hidden layers menggunakan chain rule
   - Hitung gradient untuk setiap weight dan bias
4. **Update Parameters**:
   - **Gradient Descent**: `w_new = w_old - learning_rate * gradient`
   - **SGD**: Update per mini-batch
5. **Repeat**: Ulangi untuk beberapa epoch hingga konvergen

**Key Concepts:**
- **Learning Rate**: Mengontrol seberapa besar update parameter
- **Epochs**: Jumlah iterasi training through seluruh dataset
- **Batch Size**: Jumlah samples per update (SGD)
- **Loss Curve**: Visualisasi penurunan loss selama training

## Teknologi yang Digunakan

- **Backend**: Python, Flask
- **Computation**: NumPy
- **Data Processing**: Pandas
- **Frontend**: HTML, CSS, JavaScript
- **UI**: Custom responsive design

## 📚 Resources Tambahan

- **TRAINING_GUIDE.md**: Panduan lengkap training, tips, troubleshooting
- **demo_training.py**: Demo interaktif dengan output lengkap
- **visualize_training.py**: Visualisasi loss curve dengan ASCII/matplotlib
- **example_api_training.py**: Contoh penggunaan API training

## 🧪 Testing

Jalankan demo untuk test implementasi:

```bash
# Demo training AND gate
python demo_training.py

# Visualisasi training
python visualize_training.py

# Test backpropagation
python test_training_simple.py
```

## 🎯 Contoh Hasil Training

### AND Gate dengan SGD
```
Initial Loss: 0.306300
Final Loss: 0.045181
Improvement: 85.25%
Accuracy: 100.00%

Predictions After Training:
[0 0] -> 0.0324 (target: 0) ✓
[0 1] -> 0.1890 (target: 0) ✓
[1 0] -> 0.2061 (target: 0) ✓
[1 1] -> 0.6866 (target: 1) ✓
```

## 🚀 Next Steps

Setelah berhasil menjalankan training, coba:

1. **Experiment dengan hyperparameters** - Ubah learning rate, epochs, batch size
2. **Dataset berbeda** - Coba OR gate, XOR gate, atau dataset custom
3. **Arsitektur berbeda** - Tambah hidden layers atau nodes
4. **Compare optimizers** - Bandingkan hasil GD vs SGD
5. **Visualize results** - Gunakan `visualize_training.py`

## 💡 Tips & Best Practices

1. **Start simple**: Mulai dengan network kecil (2-2-1) dan dataset sederhana
2. **Monitor loss curve**: Loss harus turun konsisten, jika naik-turun coba kurangi learning rate
3. **Patience**: Neural network butuh banyak epochs untuk konvergen
4. **Experiment**: Try different hyperparameters untuk hasil optimal
5. **Understand**: Baca TRAINING_GUIDE.md untuk pemahaman lebih dalam

## 📝 Lisensi

MIT License - Free to use for educational purposes.

## 🤝 Kontribusi

Contributions welcome! Areas untuk improvement:
- [ ] Adam optimizer
- [ ] Momentum
- [ ] Learning rate decay
- [ ] Early stopping
- [ ] Regularization (L1, L2)
- [ ] Dropout
- [ ] More activation functions (tanh, leaky ReLU)
- [ ] Save/Load model

Silakan buat issue atau pull request untuk perbaikan atau penambahan fitur.

---

**Built with ❤️ for Deep Learning Education**

Untuk pertanyaan atau feedback, silakan buat issue di repository ini.

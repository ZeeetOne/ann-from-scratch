"""
Script untuk visualisasi training results
Jika matplotlib terinstal, akan membuat grafik loss curve
Jika tidak, akan menampilkan ASCII art visualization
"""

import numpy as np
from ann_core import NeuralNetwork

def plot_loss_curve_ascii(history, title="Loss Curve"):
    """Create ASCII art plot of loss curve"""
    epochs = history['epoch']
    losses = history['loss']

    # Normalize losses to 0-20 range for plotting
    min_loss = min(losses)
    max_loss = max(losses)
    range_loss = max_loss - min_loss if max_loss > min_loss else 1

    height = 20
    width = 60

    # Create plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for i, (epoch, loss) in enumerate(zip(epochs, losses)):
        if i % (len(epochs) // width + 1) == 0:  # Sample points to fit width
            x = min(int((i / len(epochs)) * (width - 1)), width - 1)
            y = height - 1 - int(((loss - min_loss) / range_loss) * (height - 1))
            y = max(0, min(y, height - 1))
            plot[y][x] = '*'

    # Print plot
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    print(f"\nLoss")
    print(f"{max_loss:.4f} |")

    for row in plot:
        print("        |" + ''.join(row))

    print(f"{min_loss:.4f} |" + "_" * width)
    print(f"        0{' ' * (width-10)}Epochs{' ' * 5}{len(epochs)}")
    print("\nFinal Loss: {:.6f}".format(losses[-1]))
    print("="*70)

def try_matplotlib_plot(histories, labels, title="Training Comparison"):
    """Try to plot with matplotlib, fallback to ASCII if not available"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        for history, label in zip(histories, labels):
            plt.plot(history['epoch'], history['loss'], linewidth=2, label=label)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        filename = 'training_results.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n[SUCCESS] Plot saved to '{filename}'")
        plt.show()

        return True
    except ImportError:
        return False

# Main demo
print("="*70)
print(" " * 20 + "TRAINING VISUALIZATION")
print("="*70)

# Prepare dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])
y_or = np.array([[0], [1], [1], [1]])

# Train network 1: AND gate with GD
print("\n[1] Training Network untuk AND gate (Gradient Descent)...")
nn1 = NeuralNetwork()
nn1.add_layer(2, 'linear')
nn1.add_layer(2, 'sigmoid')
nn1.add_layer(1, 'sigmoid')

np.random.seed(42)
nn1.set_connections(1, [[0, 1], [0, 1]],
                    [[0.5, -0.3], [-0.4, 0.6]],
                    [0.1, -0.2])
nn1.set_connections(2, [[0, 1]],
                    [[0.8, -0.5]],
                    [0.2])

history_gd = nn1.train(X, y_and, epochs=500, learning_rate=0.5,
                       optimizer='gd', verbose=False)

print(f"[OK] Selesai - Final Loss: {history_gd['loss'][-1]:.6f}")

# Train network 2: AND gate with SGD
print("\n[2] Training Network untuk AND gate (SGD)...")
nn2 = NeuralNetwork()
nn2.add_layer(2, 'linear')
nn2.add_layer(2, 'sigmoid')
nn2.add_layer(1, 'sigmoid')

np.random.seed(42)
nn2.set_connections(1, [[0, 1], [0, 1]],
                    [[0.5, -0.3], [-0.4, 0.6]],
                    [0.1, -0.2])
nn2.set_connections(2, [[0, 1]],
                    [[0.8, -0.5]],
                    [0.2])

history_sgd = nn2.train(X, y_and, epochs=500, learning_rate=0.5,
                        optimizer='sgd', batch_size=2, verbose=False)

print(f"[OK] Selesai - Final Loss: {history_sgd['loss'][-1]:.6f}")

# Train network 3: OR gate with GD
print("\n[3] Training Network untuk OR gate (Gradient Descent)...")
nn3 = NeuralNetwork()
nn3.add_layer(2, 'linear')
nn3.add_layer(2, 'sigmoid')
nn3.add_layer(1, 'sigmoid')

np.random.seed(123)
nn3.set_connections(1, [[0, 1], [0, 1]],
                    [[0.3, 0.4], [0.5, -0.2]],
                    [0.1, 0.1])
nn3.set_connections(2, [[0, 1]],
                    [[0.6, 0.7]],
                    [0.1])

history_or = nn3.train(X, y_or, epochs=500, learning_rate=0.5,
                       optimizer='gd', verbose=False)

print(f"[OK] Selesai - Final Loss: {history_or['loss'][-1]:.6f}")

# Try matplotlib visualization first
print("\n[4] Membuat Visualisasi...")

has_matplotlib = try_matplotlib_plot(
    [history_gd, history_sgd, history_or],
    ['AND (GD)', 'AND (SGD)', 'OR (GD)'],
    'Training Loss Comparison'
)

# Fallback to ASCII if matplotlib not available
if not has_matplotlib:
    print("\n[INFO] Matplotlib tidak tersedia. Menggunakan ASCII visualization...")
    plot_loss_curve_ascii(history_gd, "AND Gate - Gradient Descent")
    plot_loss_curve_ascii(history_sgd, "AND Gate - SGD")
    plot_loss_curve_ascii(history_or, "OR Gate - Gradient Descent")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nAND Gate (Gradient Descent):")
pred = nn1.forward(X)
for i in range(len(X)):
    print(f"  {X[i]} -> {pred[i][0]:.4f} (target: {y_and[i][0]})")
print(f"  Final Loss: {history_gd['loss'][-1]:.6f}")

print("\nAND Gate (SGD):")
pred = nn2.forward(X)
for i in range(len(X)):
    print(f"  {X[i]} -> {pred[i][0]:.4f} (target: {y_and[i][0]})")
print(f"  Final Loss: {history_sgd['loss'][-1]:.6f}")

print("\nOR Gate (Gradient Descent):")
pred = nn3.forward(X)
for i in range(len(X)):
    print(f"  {X[i]} -> {pred[i][0]:.4f} (target: {y_or[i][0]})")
print(f"  Final Loss: {history_or['loss'][-1]:.6f}")

print("\n" + "="*70)
print("[DONE] Visualisasi selesai!")
print("="*70)

# Installation tip
if not has_matplotlib:
    print("\n[TIP] Install matplotlib untuk visualisasi grafik:")
    print("  pip install matplotlib")

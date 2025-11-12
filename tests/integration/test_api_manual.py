"""
Manual API Test

Quick test to verify API endpoints work correctly.

Usage:
    1. Run the app: python run.py
    2. In another terminal, run: python tests/integration/test_api_manual.py

Author: ANN from Scratch Team
"""

import requests
import json

BASE_URL = "http://localhost:5000"


def test_build_network():
    """Test build network endpoint"""
    print("\n=== Testing Build Network ===")

    payload = {
        "layers": [
            {"num_nodes": 2, "activation": "linear"},
            {"num_nodes": 2, "activation": "sigmoid"},
            {"num_nodes": 1, "activation": "sigmoid"}
        ],
        "connections": [
            {
                "layer_idx": 1,
                "connections": [[0, 1], [0, 1]],
                "weights": [[0.5, -0.3], [-0.4, 0.6]],
                "biases": [0.1, -0.2]
            },
            {
                "layer_idx": 2,
                "connections": [[0, 1]],
                "weights": [[0.8, -0.5]],
                "biases": [0.2]
            }
        ]
    }

    response = requests.post(f"{BASE_URL}/build_network", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        print(f"Message: {data.get('message')}")
        print(f"Classification Type: {data.get('classification_type')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_quick_start():
    """Test quick start endpoint"""
    print("\n=== Testing Quick Start ===")

    response = requests.post(f"{BASE_URL}/quick_start_binary")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        print(f"Message: {data.get('message')}")
        return data.get('example_dataset')
    else:
        print(f"Error: {response.text}")
        return None


def test_predict(dataset):
    """Test prediction endpoint"""
    print("\n=== Testing Prediction ===")

    payload = {
        "dataset": dataset,
        "loss_function": "binary",
        "threshold": 0.5
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        print(f"Accuracy: {data.get('accuracy') * 100:.2f}%")
        print(f"Loss: {data.get('loss'):.6f}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_training(dataset):
    """Test training endpoint"""
    print("\n=== Testing Training ===")

    payload = {
        "dataset": dataset,
        "epochs": 100,
        "learning_rate": 0.5,
        "optimizer": "gd",
        "loss_function": "binary"
    }

    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        print(f"Final Loss: {data.get('final_loss'):.6f}")
        print(f"Accuracy: {data.get('accuracy') * 100:.2f}%")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print(" API Manual Test")
    print("=" * 60)
    print(" Make sure the server is running: python run.py")
    print("=" * 60)

    try:
        # Test health endpoint
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("\nError: Server is not responding!")
            print("Please run: python run.py")
            return
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to server!")
        print("Please run: python run.py")
        return

    # Run tests
    success = test_build_network()
    if not success:
        print("\nBuild network test failed!")
        return

    dataset = test_quick_start()
    if not dataset:
        print("\nQuick start test failed!")
        return

    success = test_predict(dataset)
    if not success:
        print("\nPrediction test failed!")
        return

    success = test_training(dataset)
    if not success:
        print("\nTraining test failed!")
        return

    print("\n" + "=" * 60)
    print(" All API tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()

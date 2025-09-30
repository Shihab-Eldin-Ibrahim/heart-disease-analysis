"""
Quick test to verify confusion matrices are working properly
"""

from heart_disease_analysis import HeartDiseaseAnalyzer
import matplotlib.pyplot as plt

def test_confusion_matrices():
    print("🧪 Testing Confusion Matrix Visualization...")
    
    # Initialize analyzer
    analyzer = HeartDiseaseAnalyzer()
    
    # Load data and run basic pipeline
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.train_supervised_models()
    
    print("✅ Data loaded and models trained")
    print(f"✅ Models available: {list(analyzer.results.keys())}")
    
    # Test model comparison
    print("📈 Creating model comparison visualization...")
    fig1, fig2 = analyzer.create_model_comparison_plot()
    
    if fig2 is not None:
        print("✅ Confusion matrices created successfully!")
        plt.figure(fig2.number)
        plt.title("✅ FIXED: Confusion Matrices Visualization")
        plt.show()
    else:
        print("❌ Confusion matrices failed to create")
    
    # Show accuracy comparison
    plt.figure(fig1.number)
    plt.title("✅ Model Accuracy Comparison")
    plt.show()
    
    print("\n🎉 Confusion matrix test completed!")

if __name__ == "__main__":
    test_confusion_matrices()

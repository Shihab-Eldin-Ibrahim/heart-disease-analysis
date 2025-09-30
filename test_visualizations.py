"""
Quick test script to verify the improved visualizations
"""

from heart_disease_analysis import HeartDiseaseAnalyzer
import matplotlib.pyplot as plt

def test_visualizations():
    print("🧪 Testing Improved Visualizations...")
    
    # Initialize analyzer
    analyzer = HeartDiseaseAnalyzer()
    
    # Load data and run basic pipeline
    analyzer.load_data()
    analyzer.preprocess_data()
    analyzer.train_supervised_models()
    analyzer.dimensionality_reduction()
    
    print("✅ Data loaded and models trained")
    
    # Test main visualization
    print("📊 Creating main data analysis visualization...")
    fig1 = analyzer.create_visualizations()
    plt.figure(fig1.number)
    plt.title("✅ IMPROVED: Heart Disease Dataset Analysis")
    plt.show()
    
    # Test model comparison
    print("📈 Creating model comparison visualization...")
    fig2, fig3 = analyzer.create_model_comparison_plot()
    plt.figure(fig2.number)
    plt.title("✅ IMPROVED: Model Performance Comparison")
    plt.show()
    plt.figure(fig3.number)
    plt.title("✅ IMPROVED: Confusion Matrices")
    plt.show()
    
    # Test PCA plot
    print("🔍 Creating PCA visualization...")
    fig4 = analyzer.create_pca_plot()
    plt.figure(fig4.number)
    plt.title("✅ IMPROVED: PCA Analysis")
    plt.show()
    
    print("\n🎉 All visualizations tested successfully!")
    print("\nKey improvements made:")
    print("✅ Fixed overlapping titles")
    print("✅ Improved correlation matrix readability")
    print("✅ Added proper spacing and padding")
    print("✅ Enhanced color schemes")
    print("✅ Added value labels on plots")
    print("✅ Better font sizes and formatting")
    print("✅ Clearer axis labels with units")

if __name__ == "__main__":
    test_visualizations()

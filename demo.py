"""
Demo script to test the Heart Disease Analysis Pipeline
Run this to quickly test the complete pipeline functionality
"""

from heart_disease_analysis import HeartDiseaseAnalyzer
import matplotlib.pyplot as plt

def main():
    print("🫀 Heart Disease Analysis Pipeline Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = HeartDiseaseAnalyzer()
    
    # Run complete pipeline
    analyzer.run_complete_pipeline()
    
    # Create and display visualizations
    print("\n📊 Creating visualizations...")
    
    # Data analysis plot
    fig1 = analyzer.create_visualizations()
    plt.figure(fig1.number)
    plt.title("Data Analysis Visualization")
    plt.show()
    
    # Model comparison plot
    if analyzer.results:
        fig2, fig3 = analyzer.create_model_comparison_plot()
        plt.figure(fig2.number)
        plt.title("Model Comparison Visualization")
        plt.show()
        if fig3 is not None:  # Check if confusion matrix figure exists
            plt.figure(fig3.number)
            plt.title("Confusion Matrices Visualization")
            plt.show()
        else:
            print("⚠️ Confusion matrices not available - no models trained yet")
    
    # PCA analysis plot
    if hasattr(analyzer, 'pca_results'):
        fig4 = analyzer.create_pca_plot()
        plt.figure(fig4.number)
        plt.title("PCA Analysis Visualization")
        plt.show()
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the Streamlit web app:")
    print("streamlit run heart_disease_analysis.py")

if __name__ == "__main__":
    main()

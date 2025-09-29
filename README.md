# Text Translation and Sentiment Analysis using Transformers

A comprehensive machine learning project that performs multilingual text translation and sentiment analysis on movie reviews using state-of-the-art Transformer models from Hugging Face.

## 🎯 Project Overview

This project analyzes the sentiment of movie reviews written in three different languages (English, French, and Spanish). The pipeline includes:

1. **Text Translation**: Converting French and Spanish reviews and synopses to English using pre-trained MarianMT models
2. **Data Preprocessing**: Standardizing and cleaning the multilingual dataset
3. **Sentiment Analysis**: Analyzing the sentiment of movie reviews using pre-trained sentiment analysis models
4. **Output Generation**: Creating a unified dataset with sentiment scores

## 📊 Dataset

The project works with 30 movies (10 in each language) with their reviews and synopses stored in separate CSV files:

- `data/movie_reviews_eng.csv` - English movie reviews
- `data/movie_reviews_fr.csv` - French movie reviews  
- `data/movie_reviews_sp.csv` - Spanish movie reviews

### Data Structure
Each CSV file contains:
- **Title/Título/Titre**: Movie title
- **Year/Año/Année**: Release year
- **Synopsis/Sinopsis**: Movie synopsis
- **Review/Críticas/Critiques**: Movie review
- **Original Language**: Language indicator (added during processing)

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch 2.8.0+** (with CUDA support)
- **Transformers 4.56+** (Hugging Face)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Key Models
- **Helsinki-NLP/opus-mt-fr-en**: French to English translation
- **Helsinki-NLP/opus-mt-es-en**: Spanish to English translation
- **Sentiment Analysis Model**: Pre-trained model for emotion detection

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Conda or pip package manager
- CUDA-compatible GPU (recommended)

### Key Functions

#### Data Preprocessing
```python
def preprocess_data() -> pd.DataFrame:
    """
    Reads movie data from CSV files, standardizes column names,
    adds language indicators, and concatenates into unified DataFrame.
    """
```

#### Text Translation
```python
def translate(text: str, model, tokenizer) -> str:
    """
    Translates text using MarianMT model and tokenizer.
    
    Args:
        text: Input text to translate
        model: Pre-trained MarianMT model
        tokenizer: Corresponding tokenizer
    
    Returns:
        Translated text in English
    """
```

#### Sentiment Analysis
```python
def analyze_sentiment(text, classifier):
    """
    Performs sentiment analysis on text using pre-trained classifier.
    
    Args:
        text: Input text for analysis
        classifier: Pre-trained sentiment analysis model
    
    Returns:
        Sentiment label (Positive/Negative)
    """
```

## 📁 Project Structure

```
transformers/
├── README.md                                                    # Project documentation
├── TextTranslationAndSentimentAnalysisUsingTransformers.ipynb  # Main notebook
├── data/                                                        # Input datasets
│   ├── movie_reviews_eng.csv                                   # English reviews
│   ├── movie_reviews_fr.csv                                    # French reviews
│   └── movie_reviews_sp.csv                                    # Spanish reviews
├── result/                                                      # Output directory
│   └── reviews_with_sentiment.csv                              # Final processed dataset
└── .vscode/                                                     # VS Code configuration
```

## 🔄 Workflow

1. **Data Loading**: Read CSV files for each language
2. **Column Standardization**: Map different column names to standard format
3. **Language Tagging**: Add original language indicators
4. **Model Loading**: Load pre-trained translation and sentiment models
5. **Translation**: Convert French and Spanish text to English
6. **Sentiment Analysis**: Analyze sentiment of all reviews
7. **Export**: Save final dataset with sentiment scores

## 📈 Output

The final output is a CSV file (`result/reviews_with_sentiment.csv`) containing:

| Column | Description |
|--------|-------------|
| Title | Movie title |
| Year | Release year |
| Synopsis | Movie synopsis (translated to English) |
| Review | Movie review (translated to English) |
| Review Sentiment | Sentiment analysis result (Positive/Negative) |
| Original Language | Original language of review (English/French/Spanish) |

## ⚠️ Troubleshooting

### Common Issues

1. **PyTorch Version Error**
   ```
   ValueError: torch.load vulnerability issue
   ```
   **Solution**: Ensure PyTorch >= 2.6.0 is installed
   ```bash
   pip install "torch>=2.6.0" --upgrade
   ```

2. **Kernel Corruption**
   - Restart Jupyter kernel: `Kernel > Restart`
   - Or restart VS Code notebook kernel: `Ctrl+Shift+P > Jupyter: Restart Kernel`

3. **CUDA Memory Issues**
   - Reduce batch size or use CPU-only mode
   - Clear GPU cache: `torch.cuda.empty_cache()`

4. **Model Download Issues**
   - Ensure stable internet connection
   - Models are downloaded automatically on first use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for translation models
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Transformers](https://github.com/huggingface/transformers) library

## 📧 Contact

For questions or support, please open an issue in the repository.


# Text Decoding from GPT-2 Using Beam Search

This repository contains a Jupyter Notebook (`PA4_CSE256_FA24.ipynb`) that explores text decoding from a pretrained GPT-2 model using beam search. The notebook is designed as an assignment (PA4) for CSE256 (Fall 2024) and includes implementations of beam search, visualization tools, and custom score processors to handle repetition in generated text.

## Project Overview

The notebook demonstrates the following:
1. **Beam Search Implementation**: Decoding text from GPT-2 with configurable beam sizes and decoding steps.
2. **Visualization**: Plotting token probabilities and displaying them with colored text to analyze model behavior.
3. **Repetition Mitigation**: 
   - `WordBlock`: Prevents repeated unigrams by penalizing previously seen words.
   - `BeamBlock`: Prevents repeated n-grams (e.g., trigrams) using beam blocking techniques.
4. **Exercises**: Includes questions and coding tasks to explore beam search dynamics.

The code uses Hugging Face's Transformers library and runs on Google Colab with GPU acceleration.

## Prerequisites

To run this notebook, you'll need:
- **Python 3.7+**
- **Google Colab** (recommended) or a local environment with GPU support
- **Dependencies**:
  ```bash
  pip install -q sentence-transformers==2.2.2 transformers==4.17.0 torch matplotlib rich numpy
  ```

## Setup Instructions

### Running on Google Colab
1. **Upload to Google Drive**:
   - Copy the notebook to your Google Drive.
   - Open it in Google Colab via the link: [https://colab.research.google.com/drive/1lr1__w7G6rIaS-GpxUoF9PZQ_G9p6iuk](https://colab.research.google.com/drive/1lr1__w7G6rIaS-GpxUoF9PZQ_G9p6iuk).
2. **Enable GPU**:
   - Go to `Edit > Notebook Settings > Hardware accelerator` and select `GPU`.
3. **Install Dependencies**:
   - Run the cell under "Installing Hugging Face's Transformers and Additional Libraries" to install required packages.
4. **Execute the Notebook**:
   - Run all cells sequentially to see the output of beam search, visualizations, and custom score processors.

### Running Locally
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` file with the dependencies listed above.)
3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook PA4_CSE256_FA24.ipynb
   ```
4. Ensure you have a GPU and PyTorch configured with CUDA support for optimal performance.

## Notebook Structure

- **Introduction**: Setup for GPU and library installation.
- **Part 1 - Beam Search**:
  - Utility functions for beam search decoding.
  - Visualization of token probabilities with plots and colored text.
- **Questions**:
  - **1.1**: Generate and print the third most probable sequence with a beam size of 4.
  - **1.2**: Analyze why later tokens have higher probabilities.
  - **1.3**: Compare beam search with and without `WordBlock`.
  - **1.4**: Implement `BeamBlock` to prevent tri-gram repetition.

## Example Output

### Beam Search (No Score Processor)
```
Once upon a time, in a barn near a farm house, a young boy was playing with a stick. He was playing with a stick, and the boy was playing with a stick...
```

### Beam Search with WordBlock
```
Once upon a time, in a barn near a farm house, the young girl was playing with her father's dog. She had been told that she would be given to him...
```

### Beam Search with BeamBlock (n=2)
```
Once upon a time, in a barn near a farm house, a young boy was playing with a stick. He was about to be killed when he was hit by a bullet...
```

## Notes
- The notebook uses GPT-2 (`gpt2` model) from Hugging Face. Ensure an internet connection for model downloads.
- `BeamBlock` is implemented for n-grams (default: bigrams), but you can adjust `n_gram_size` for trigrams or other sizes.
- Visualizations require `matplotlib` and `rich` for plotting and colored text output.

## References
- Hugging Face Transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- Beam Blocking Paper: [https://arxiv.org/pdf/1705.04304.pdf](https://arxiv.org/pdf/1705.04304.pdf) (Section 2.5)



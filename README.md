# 🖼️ Image Captioning using Transformers

- This project implements an **image captioning model** using a **Transformer-based architecture**. It takes an image as input and generates a meaningful natural language description.\
- “The model integrates a pre-trained CNN (VGG16) for visual feature extraction, followed by a Transformer-based encoder-decoder architecture for sequential caption generation.”

> Each image in the dataset is annotated with 5 captions, helping the model learn diverse sentence structures.

---

## 🧠 Key Components

### 📊 Dataset
- **Flickr30k Dataset**: Contains images with five textual descriptions each.
- Format: CSV file `captions.txt` mapping image file names to their respective captions.

---

## 🔧 Tools & Technologies Used

| Tool/Library       | Purpose |
|--------------------|---------|
| **Python**         | Main programming language |
| **TensorFlow / Keras** | Model creation, training, and evaluation |
| **VGG16**          | Pre-trained CNN for image feature extraction |
| **Transformer Architecture** | For sequence modeling (attention mechanism) |
| **Pandas, NumPy**  | Data manipulation |
| **Matplotlib, Seaborn** | Visualization |
| **NLTK**           | BLEU score for evaluating generated captions |
| **Tqdm**           | Progress bar for loops |
| **PIL (Pillow)**   | Image loading and processing |

---

## 📦 Project Structure

```
📁 Image-Captioning-using-Transformers
│
├── 📁 research
│   └── image-captioning-using-transformers.ipynb     # Main notebook with all steps
│
├── 📁 models
│   ├── model.keras                                   # Saved Transformer model
│   └── tokenizer.pkl                                 # Tokenizer object for captions
│
└── README.md                                         # Project documentation

```

---

## 🔍 Model Architecture

1) **CNN Feature Extractor** *(Encoder) – Visual Feature Encoder* 
   - Backbone: Pre-trained VGG16 model from ImageNet.

   - Role: Extracts high-level features from input images. The fully connected layers of VGG16 are discarded, and the output from the final convolutional layer is retained.

   - Output: A fixed-length vector (or feature map) that encodes spatial and semantic information about the image.

   - Why VGG16? \
      Its deep architecture effectively captures hierarchical representations and it is lightweight compared to newer CNNs.

2) **Transformer Encoder** – *Visual Feature Refiner* 
   - Takes the VGG16 image feature vector as input.

   - Applies Layer Normalization to stabilize training.

   - Passes features through a Dense Layer to project them into the same dimensional space as the caption embeddings.

   - Processes the features using a Multi-Head Attention layer:-

      Helps the model focus on different spatial parts of the image.

      Enhances representation by attending to multiple areas simultaneously.

      Another Layer Normalization follows to refine output.

   - Key Advantage: This enables the model to "look" at different parts of the image while generating each word of the caption.

3) **Transformer Decoder** – *Caption Generator* 
   - Inputs are the tokenized and embedded captions (with positional encoding added to preserve word order).

   - The decoder:

      Uses Masked Multi-Head Self-Attention to prevent the model from "cheating" (i.e., looking ahead at future words).

      Applies Cross-Attention between the encoded image features and embedded caption tokens.

      Finally, passes through a Dense Softmax layer to generate a probability distribution over the vocabulary at each timestep.

   - Output: A sequence of predicted words forming a caption, starting with <start> and ending with <end> token.

---

## ⚙️ Training Configuration

| Hyperparameter     | Value        |
|--------------------|--------------|
| `MAX_LENGTH`       | 30           |
| `MAX_TOKENS`       | 10,000       |
| `BATCH_SIZE`       | 128          |
| `EMBEDDING_DIM`    | 512          |
| `UNITS`            | 512          |
| `EPOCHS`           | 20           |

---

## 📈 Evaluation Metric

- **BLEU Score** from `nltk.translate.bleu_score` is used to evaluate the quality of generated captions against the ground truth captions.
- These scores measure the similarity between the candidate caption and the reference captions using n-grams (sequences of n words).
- and obtained these scores over Flickr30k dataset :- \
   `BLEU-1`: 0.7140 \
   `BLEU-2`: 0.4990 \
   `BLEU-3`: 0.4026 \
   `BLEU-4`: 0.4408 

---

## 📌 How to Run

1. Clone the repository
2. Ensure dataset (Flickr30k) and `captions.txt` are in place
3. Open the notebook in Jupyter or any compatible environment
4. Run all cells to:
   - Load and clean data
   - Extract image features
   - Train the Transformer model
   - Generate and evaluate captions

---

## 🎯 Sample Image & Caption

Image:

![example](https://storage.googleapis.com/kagglesdsdata/datasets/623329/1111749/Images/flickr30k_images/1000092795.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250625%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250625T072346Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=3766215501091e69460fd909430f3e59e16bff7112352104f2d0677d404f8e8f249061ad03dd7032bebf13ac4d109243e3e7c62bc71dc948a075ce8e34b3371be06ca8739d19f465d81a11b48e4336ee818165a3aa82289438e52d0910e67d23cb40c46bb5964a763436bc2fdfd83a2f1fee0919bd2b6384232dabd59dc5fce316185d96e7920ff65017fe4058c029089c68d5bd52d424bf90bbfcd6fe3448b2ebfeacd8f735a9b352055fcf253c3bf0d8219192584b02ae44e3e80f1bfacb65f227d7302a44013f739f96da5c8c3e03ec2176dcf18b448a70300cd3870e31d16fe10d9c05b131c48d595601da1851887d007fb3b93116fc01e42d70de23e003)  
Caption: `"Two friends enjoy time spent together."`

---

## 📚 References

- [Flickr30k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k)
- [Attention is All You Need (Vaswani et al., 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- TensorFlow, Keras documentation

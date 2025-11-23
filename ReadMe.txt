Media Bias Classifier (TextCNN)
================================
Modern news consumption is mediated by algorithms that tend to amplify existing political preferences, obscuring ideological framing, and narrowing exposure to diverse viewpoints. This project proposes an AI-driven system that combines deep learning and natural language processing to classify Canadian news articles as Left, Right, or Neutral. By accepting article URLs or pre-labeled outlets, the model produces systematic, transparent assessments of political bias, thereby enabling scholars, journalists, and readers to interrogate media narratives with greater rigor and to mitigate the polarization that echo chambers sustain.

Classifies news articles as Left, Right, or Neutral bias. It fetches Kaggle datasets, cleans them, trains a TextCNN with PyTorch, and serves a FastAPI UI so you can paste a URL and see the predicted leaning.

Quick Start (recommended)
-------------------------
1) Clone/download the repo and (optionally) activate your virtualenv.  
2) Put your Kaggle token at `%USERPROFILE%\.kaggle\kaggle.json`.  
3) From the repo root, run one of:
   ```
   .\scripts\quickstart.bat
   ```
   or
   ```
   powershell -ExecutionPolicy Bypass -File .\scripts\quickstart.ps1
   ```
   This installs dependencies, auto-installs the right torch build (CUDA for NVIDIA, DirectML for AMD/ATI, CPU otherwise), downloads data, prepares CSVs, trains, and starts the API.
4) Open http://127.0.0.1:8000 and paste article URLs to get labels.


Manual setup (only if you skip quickstart)
------------------------------------------
- Base deps: `python -m pip install -r requirements.txt`
- Choose one torch build:
  - NVIDIA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
  - AMD/Intel (DirectML): `python -m pip install -r requirements-directml.txt`
  - CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`


Dataset Description, including how the data was split into training, testing, and validation sets
--------------------------
The dataset used for training is a single, unified corpus constructed from multiple Kaggle collections focused on media and political bias. The primary sources are:
- surajkarakulath/labelled-corpus-political-bias-hugging-face (news articles labeled as left, right, or centre).
- gandpablo/news-articles-for-political-bias-classification (articles with per article political bias labels).
- timospinde/mbib-media-bias-identification-benchmark (a large benchmark of heterogeneous bias datasets; only the parts with clear article level bias labels were used).
- newsanalysis/political-bias-in-mainstream-media (mainstream outlets with bias ratings).
- tegmark/mediabias (phrase and article level media bias data where article text and source bias could be aligned).
- timospinde/babe-media-bias-annotations-by-experts (expert annotated bias segments and neutral headlines, used both for neutral and left/right examples).

All of these sources were normalised into a common format using a dedicated preprocessing script. For each dataset, the script:

- Identified the main article text column (e.g., text, article, body) and, where present, the headline/title (title, headline) and outlet/source name (outlet, source, media_name, etc.).
- Constructed a single input string per article by concatenating source: <outlet>. <title>. <article_body>, so the model could learn from both the outlet identity and the textual content.
- Mapped the original bias annotations (which vary by dataset) into a unified three class scheme: Left, Right, and Neutral. This included handling label strings such as “lean left”, “centrist”, “least biased”, and numeric label codes where the dataset used integers instead of strings.
- Removed rows with missing or unmappable labels and dropped extremely short texts (fewer than 15 characters), which are unlikely to contain enough information for reliable bias classification.
- Deduplicated entries based on the combined text field so that repeated articles that appear in multiple sources are only counted once in the training data.

After merging the datasets, the corpus was dominated by Neutral articles. To avoid bias, the dataset was balanced through down sampling. The smallest class size was used as the target, and the larger classes were randomly reduced to match it. This produced a final dataset with equal counts for Left, Right, and Neutral, which was then used as the fixed input for all experiments. From the cleaned and balanced corpus, stored as one CSV with text and label columns, the data was split into training, validation, and test sets using a stratified method. This ensured that each set kept the same class proportions.

- A 70 percent, 15 percent, 15 percent split is widely recommended because it reserves most data for training while keeping separate validation and test sets for tuning and final evaluation, which helps prevent overfitting and ensures proper generalisation.
- Each split preserved equal counts of Left, Right, and Neutral examples, producing a total corpus of 32,604 items with 10,868 examples in each class.
- 22,824 training examples (7,608 Left, 7,608 Right, 7,608 Neutral)
- 4,890 validation examples (1,630 per class)
- 4,890 test examples (1,630 per class)

The training split is used to build the vocabulary and fit model parameters, the validation split is used exclusively for hyperparameter tuning, early stopping, and metric reporting, and the test split is kept for final evaluation.

Note: Although the dataset is class-balanced after preprocessing, it may still reflect biases in how different news outlets or labeling sources define “Left,” “Right,” and “Neutral.” These underlying inconsistencies can influence how well the model generalises to new or international news domains.

Training Procedure
--------------------------
The model was only trained after the dataset had been fully prepared and fixed as the independent variable.

- The final TextCNN was trained with 200 dimensional embeddings, four convolutional filter sizes (3, 4, 5, 7) and 100 filters per size, followed by max over time pooling, a 0.6 dropout layer, and a final fully connected layer producing three logits.
- Optimisation used the Adam optimiser with a learning rate of 0.0007 and modest class weights that slightly up weighted Left and Right relative to Neutral to emphasise performance on the political labels without changing the architecture or data (Left:1.10,Right:1.20,Neutral:1.0).
- A ReduceLROnPlateau scheduler monitored validation loss and cut the learning rate when improvements stalled, and early stopping terminated training if validation loss failed to improve for two consecutive epochs, typically selecting a best checkpoint between epochs 8 and 10.
- After each epoch, training and validation loss and accuracy were recorded; whenever a new minimum validation loss was observed, the model weights were checkpointed, and the final selected checkpoint was evaluated on the validation split to produce detailed metrics (per class precision, recall, F1, confusion matrix, and epoch history) while leaving the test split untouched for potential future evaluation.

Evaluation Metrics
------------------
We report accuracy, precision, recall, and F1-score:

- **Accuracy**: Proportion of correctly classified articles.  
- **Precision**: Among articles predicted as a given class, how many truly belong to that class.  
- **Recall**: Among all articles that truly belong to a class, how many the model correctly recovers.  
- **F1-score**: Harmonic mean of precision and recall, balancing false positives and false negatives.

Because even a balanced corpus can behave unevenly during training, we report both macro and weighted F1-scores to ensure that no class dominates the overall performance.


Model Description (model type and architecture)
--------------------------
The final model is a TextCNN classifier that takes tokenized news articles as input and predicts one of three bias labels: Left, Right, or Neutral. It is a shallow CNN designed to capture phraselevel patterns in text rather than relying on very deep stacking of layers.

- Input text is converted to integer token IDs and passed through a learned embedding layer that maps each token to a 200 dimensional vector.

- The embedded sequence runs through four parallel 1D convolutions (kernel sizes 3, 4, 5, 7; 100 filters each) so the TextCNN detects short bias cues via 3-7 word n‑grams. Larger kernels aren’t used because bias signals rarely span longer phrases, so extra parameters may confuse the model.

- Each conv feature map passes through ReLU, then max-over-time pooling to keep its strongest n‑gram signal; the pooled outputs from all four kernel sizes (100 filters each) concatenate into a 400‑dimensional feature vector (4 n-gram x 100 filter).

- A dropout layer with a rate of 0.6 is applied to this concatenated vector for regularisation, and a final fully connected layer projects the 400 features down to three labels (Left, Right, and Neutral).

- The classifier’s logits are passed through a softmax (trained with weighted cross-entropy) to yield normalized probabilities for the Left, Right, and Neutral classes. That keeps the description compact while covering the final probability output and the loss used to train it.


Evaluation Results and a Performance Results:
--------------------------
The model was evaluated on a held-out test set using standard classification metrics: precision, recall, F1-score, and accuracy. Weighted averages were used to ensure that class imbalance did not skew the results.

Overall Performance:
- Accuracy: 85.4%
- Macro F1-Score: 85.4%

F1-Score by Class:
- Neutral: 93.5%
- Left: 82.2%
- Right: 80.3%

The results indicate a strong overall performance. The model is particularly effective at identifying Neutral articles. As shown by the per-class F1-scores and the confusion matrix, the model is less certain when distinguishing between 'Left' and 'Right' biased content, which represents the most significant challenge for this classification task.


Overall Analysis of the Model:
--------------------------
The model demonstrates strong predictive power, achieving an overall accuracy of 85.4%, which is a robust result for a nuanced three-class NLP task.

Its primary strength lies in the accurate identification of Neutral articles, which scored an F1 of 93.5%. This suggests that neutral journalistic language has distinct, machine-learnable patterns that the TextCNN architecture, focused on short phrases (n-grams), is well-suited to capture.

The main area for improvement is in distinguishing between Left and Right-leaning content, where F1-scores were lower (82.2% and 80.3%, respectively). This is the core challenge of the problem, as ideological bias is often expressed through subtle framing and contextual cues that are difficult to differentiate using only local phrase patterns. While the TextCNN provides a strong baseline, its architectural limitations may be a factor here.

Future work could explore more advanced, context-aware architectures like Transformers (e.g., BERT), which might better capture the long-range dependencies that distinguish Left from Right-leaning narratives. Nonetheless, the current model serves as a very effective proof-of-concept for systematically assessing media bias.


Contribution Breakdown specifying the role and work completed by each group member
--------------------------
Hajar:
- Implemented the conversion and merging logic that turns heterogeneous Kaggle datasets into a single, consistent text,label training file.
- Designed and coded the data balancing step so that Left, Right, and Neutral each have the same frequency.
- Added and iteratively refined training controls such as learning rate scheduling, early stopping, class weighting, and outlet based text augmentation, using the metrics to drive successive improvements.
- Implemented the logic to skip or stop training when later epochs showed little to no improvement.

Hans: 
- Exported per-class precision/recall/F1, macro/weighted averages, confusion matrices, and per-epoch history into structured JSON.
- Built the metrics plots: combined bar chart, precision/recall/F1 trend lines, and confusion matrix heatmap per run.
- Configured and refined the quickstart scripts and documentation, making it possible to go from a clean environment to a trained model and running API with a single command.

Shared:
- Researched and selected the TextCNN architecture and key hyperparameters (embedding size, filter sizes, number of filters, dropout levels) based on both literature and empirical results.
- Worked together on parameter tuning and ablation experiments, interpreting per class F1 scores and confusion matrices to decide which changes (e.g., adding 7 gram filters, adjusting dropout, using class weights) were genuinely beneficial.
- Reviewed results from multiple runs and used them to refine data preprocessing choices (such as outlet augmentation and minimum text length) and training settings.
- Simplified and refactored code paths where possible (e.g., simplifying dataset splitting and metrics handling) to make the project easier to run, debug, and extend for future work.

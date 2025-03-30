# AI-Enabled Choreography: Dance Beyond Music



- **Name:** Yash Ogale  
- **University:** Veermata Jijabai Institute of Technology Mumbai (VJTI)  
- **Location:** Mumbai, Maharashtra, India  
- **Time Zone:** Indian Standard Time (UTC+5:30)  
- **Email:**  
  - Primary: [ynogale@gmail.com](mailto:ynogale@gmail.com)  
  - University: [ynogale_b23@et.vjti.ac.in](mailto:ynogale_b23@et.vjti.ac.in)  
- **GitHub:** [yashogale30](https://github.com/yashogale30)  
- **Resume:** [Resume Link](https://drive.google.com/file/d/12D3VHWiCak4SaKstke8DVEtj2pn-o8bj/view?usp=sharing) 
- **LinkedIn:** [yashogale](https://www.linkedin.com/in/yash-ogale-03a30b2aa/)  


<p align="center">
  <img src="images\intro1.png" width="30%"/>
  <img src="images\intro2.png" width="30%"/>
  <img src="images\intro3.png" width="30%"/>
</p>


## Background
AI-generated choreography has recently gained attention, with models trained on motion capture data to generate dance sequences. . In GSoC 2024, contributors developed projects analyzing improvisational dance duets using neural networks such as Graph Neural Networks (GNNs) and Transformers.

However, dance is more than just movement; it is influenced by various factors beyond music, including speech, imagery, writing, touch, architecture, proprioception, and sculpture. This project seeks to create a more inclusive AI approach to dance that embraces these diverse influences instead of reducing dance to mere movement prediction.

## Project Duration
- **Total length:** 175 hours

## Objectives
1. **Dataset Creation:** Develop a dataset of dynamic point-cloud data extracted from motion capture videos of dancers.
2. **Multimodal Pairing:** Design a (semi-)supervised approach to pair dance movements with multiple modalities, such as:
   - Natural language descriptions
   - Spoken word
   - Music
   - Visual art
   - 3D architectural renderings
3. **Embedding Framework:** Establish a multimodal embedding method that integrates these diverse modalities.
4. **Generative Model:** Develop an any-to-any generative model capable of translating one modality into another, including 3D movement generation from non-movement inputs.
5. **Artist-Driven Design:** Collaborate closely with dance artists to guide the model’s development.

## Expected Outcomes
- A dataset containing motion capture data for dance.
- A structured framework for representing dance across multiple modalities.
- A generative AI model that translates between different artistic inputs and dance movements.
- A model designed in partnership with dance artists to maintain artistic integrity.

---

## Tasks

### Task 1: Visualizing Dance Sequences
The first task is to develop a 3D plotting function that visualizes animated dance sequences using motion capture data. The motion capture data is stored in `.npy` format, containing joint positions over time. Each file represents a different motion capture session from the same dancer.

- Display the dancer as either a cloud of points or with connected lines for better clarity.
- Ensure that the visualization enables movement playback over a selected number of timesteps.

The npy files are in format of (no_of_joints , time_steps , 3 )
i.e we can imagine this as a 3D cuboid 
the rows indicate the number of joints, the coloumns represent timeframes, and similarly for all three axes i.e the depth , 


#### Approach 1: Using Matplotlib
- Used `mpl_toolkits.mplot3d` to plot `.npy` files in a 3D space.
- Encountered issues due to lack of joint labeling, causing ambiguity.
- **Took help from last year’s labels** to attempt structuring the visualization.
  <p align="center">
  <img src="images\last_yr_label.png" width="70%"/>
  
  </p>
- Despite this, ambiguity in joint connections remained, making this approach unsuitable.
  <p align="center">
  <img src="images\image1.png" width="40%"/>
  <img src="images\gif1.gif" width="47%"/>

  
  </p>


#### Approach 2: Visualization with Plotly
- Used **Plotly Express (px)** and **Plotly Graph Objects (go)** to visualize a single frame.
- **Scatter Plot:** Plotted the first frame’s joint coordinates (x, y, z) in a 3D scatter plot.
<p align="center">
  <img src="images\gif2.gif" width="65%"/>

  
  </p>
- **Understanding the Human Skeleton:** I then gained a clear understanding of the human structure and manually labeled the joints, identifying which points correspond to each body part. This helped in structuring a meaningful skeleton.
- <p align="center">
  <img src="images\image2.png" width="100%"/>
  
  </p>
- **Connecting Joints:** After identifying key points, created a set of joint connections to represent the human skeleton properly.
- <p align="center">
  <img src="images\gif3.gif" width="70%"/>
  
  </p>

##### Rendering Movement Across Frames
- Used **kaleido** and **plotly.io**:
  - Converted Plotly figures into images (RGB arrays).
- Used **OpenCV**:
  - Each frame visualized and compiled into a 30 FPS video.


### Challenges Faced: Hardware Constraints  

While working with `mariel_betternot_and_retrograde.npy`, which contained **10,925 frames**, processing the entire dataset was computationally expensive. To optimize performance, I **skipped every 5 frames** (i.e., 0th, 5th, 10th, ...) to speed up processing.  

However, this skipping resulted in an output video that played **5× faster** than the original motion. To correct this and restore natural timing, I used the **FFmpeg** library to slow down the playback by a factor of **0.2×** before saving the final output video.


**Final Output:** A correctly timed, high-quality visualization of dance motion.
<p align="center">
  <img src="images\gif4.gif" width="50%"/>
  
  </p>
<p align="center">
  <a href="https://drive.google.com/file/d/1-2U855bCS0xD1RE_ncmGQ6YJ9yfMiH7g/view?usp=sharing">OUTPUT VIDEO</a>
</p>

---

### Task 2: Multimodal Dance & Text Embedding
The second task involves training a neural network using contrastive learning to embed short dance sequences and natural language descriptions into a shared representation space.

#### Key Components:
- Selecting and implementing a contrastive learning method.
- Generating semi-supervised natural language labels for dance sequences (~1% manual, rest automated).
- Using **K-Means clustering** for movement categorization.
- Applying data augmentation where relevant.
- Evaluating model performance in:
  1. Generating dance sequences from text inputs.
  2. Generating text descriptions from dance inputs.

---

## Approach & Implementation

### Step 1: Data Preparation and Manual Labeling
- Segmented `.npy` files into **40-frame chunks**.
- Manually labeled **30 segments (1200 frames total)** with the following labels:
  - Stand, Walk, Lie_down, Flip_horizontally, Raise_hand, Sleep_dance, Stand_dance.
- **Checked last year’s labels** to maintain consistency in movement categorization.
- Saved labeled data in JSON format.

### Step 2: Auto-labeling Using Clustering
Used **PCA (Principal Component Analysis)** to reduce dimensionality and **K-Means Clustering** to group similar movements.

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
```
- **PCA:** Preserved variance while structuring data.
- **K-Means Clustering:**
  - Created **10 clusters** and assigned labels based on similarity with manually labeled data.

---

### Step 3: Dance-Text Contrastive Model
Designed a contrastive learning model to embed dance sequences and textual descriptions into a shared space.

#### Model Architecture
```python
self.dance_encoder = nn.Sequential(
    nn.Linear(55 * 40 * 3, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
)

self.text_embedding = nn.Embedding(1000, 256)
self.text_encoder = nn.LSTM(256, 256, batch_first=True)
```

- **Dance Encoder:**
  - Input: 55 joints × 40 frames × 3D coordinates → Flattened.
  - Two fully connected layers to extract 256-dimensional embeddings.
- **Text Encoder:**
  - Tokenized text input processed through an **LSTM**.
  - Output: 256-dimensional vector representation.

#### Contrastive Loss Function
```python
def contrastive_loss(dance_emb, text_emb):
    return -torch.cosine_similarity(dance_emb, text_emb).mean()
```
Encourages dance and text embeddings to be similar when paired.

---

### Step 4: Training Process
- **Input:** Dance sequences (as tensors) and their corresponding text labels.
- **Repeatedly referenced last year’s labels** to improve training accuracy.
- **20 epochs** of training using **Adam optimizer (lr=0.001)**.
- **Final Loss:** `-0.9792491793632507` (indicating strong alignment of embeddings).

---

### Step 5: Testing & Dance Recognition
To evaluate the model, I tested it on new dance sequences:
#### Text Prediction from Dance
1. Load the trained model and label dictionary.
2. Extract the first 40 frames from a test dance sequence.
3. Convert to tensor and pass through the model to get the dance embedding.
4. Compare the dance embedding with all stored text embeddings using cosine similarity.
5. Select the text label with the highest similarity score as the predicted description.

---

### Step 6: Generating Dance from Text
After testing dance recognition, I also generated a dance sequence from a given text label.
#### Process:
1. Check if the text exists in the vocabulary (word_to_index).
2. Convert text to tensor and get its embedding using the model.
3. Decode the text embedding into a dance sequence using a linear layer:
4. `decoder = nn.Linear(text_embedding.shape[1], 55 * 40 * 3)`
5. `dance_embedding = decoder(text_embedding)`
6. Reshape output to match (55 joints, 40 frames, 3D).
7. Save the generated dance sequence as `"generated_dance.npy"`.

---

### Final Results
- The trained contrastive model successfully linked dance motion and text descriptions.
- Recognized dance sequences based on motion input.
- Generated dance movements corresponding to given text labels.

<p >
  <a href="https://colab.research.google.com/drive/1WxubTdg3GvI_2-Dv_gmC-hihNcMiCQkk?usp=sharing">Task 1 file</a>
</p>

<p >
  <a href="https://colab.research.google.com/drive/1e9LQjeML2KpVCEbyFEWkdvjEmONNVaHp?usp=sharing">Task 2 file</a>
</p>
# CHD-TOF-Network-Diffusion-Model

This project uses publicly available genomic data to build a machine learning model that identifies genes most likely associated with Tetralogy of Fallot (TOF).

  *TOF is a congenital heart defect characterized by four structural abnormalities of the heart.

The goal of this project is to demonstrate how artificial intelligence can be applied to public biological datasets to prioritize candidate genes involved in congenital heart disease.

This is a computational research project and does not involve clinical or patient data.*


# AI Model Approach

**1) Feature Engineering**

Each gene is converted into structured features such as:

- Known association with congenital heart disease

- Participation in cardiac development pathways

- Expression in embryonic heart tissue

- Involvement in Notch signaling or transcription regulation

- Number of literature mentions related to TOF

**2️) Labeling**

Genes previously associated with TOF are labeled as:

- 1 = Known TOF-associated gene

- 0 = Not currently associated

**3️) Machine Learning Model**

We are using a classification model (Random Walk Routing) to predict the probability that a gene contributes to Tetralogy of Fallot.

# The model produces:

- Probability scores for each gene

- Feature importance rankings

- A ranked list of candidate genes

  # IMPORTANT LIMITS

This model does not prove causation.

Predictions depend entirely on public data quality.

Some genes may be underrepresented due to publication bias.

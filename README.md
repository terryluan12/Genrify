# Genrify

## Overview

**Genrify** is a Convolutional Neural Network (CNN)-based ensemble learning model designed for the classification of three-second audio clips into one of ten genres:

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

![Genrify Illustration](https://github.com/terryluan12/Genrify/assets/56266904/e7dc474e-a43d-47a4-933f-a761594d2f76)

## Model Architecture

Our model employs ensemble learning, where four parallel CNNs work in a majority vote scheme to make the final prediction. Each weak learner CNN adopts a different preprocessing approach for the three-second WAV file input. The four preprocessing methods used are:

- Spectrogram
- Mel-spectrogram
- Chroma
- MFCC

## Resources

For more detailed information about the project, please refer to our presentation:

- [Genrify Presentation](https://docs.google.com/presentation/d/1SXDWecciwypmPmyzwSq1_Rx65wBvp7c6l4Pt9K0WopQ/edit?usp=sharing)

For our comprehensive final report, you can access it through this link:

- [Genrify Final Report](https://docs.google.com/document/d/1x_qe4E25_nmzUQsxAx4JxU7f4pMhfjB14XLgYC40kng/edit?usp=sharing)

Feel free to explore these resources for a more in-depth understanding of the Genrify project.

---

# Multimodal Sarcasm Detection  

## Overview  

This project addresses the **Multimodal Sarcasm Detection** task, which involves classifying sarcasm in content that combines both text and images. The goal is to categorize inputs into one of four classes:  
- **MULTI-SARCASM** (sarcasm present in both text and image)  
- **IMAGE-SARCASM** (sarcasm present only in the image)  
- **TEXT-SARCASM** (sarcasm present only in the text)  
- **NON-SARCASM** (no sarcasm present)  

This repository provides an implementation of the **Multiview-CLIP model**, adapted from the official code of [MMSD2.0](https://github.com/JoeYing1019/MMSD2.0?tab=readme-ov-file#mmsd20-towards-a-reliable-multi-modal-sarcasm-detection-system) as presented in the paper **"MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System."**  

Our model was evaluated in the **UIT Data Science Challenge B**, and our ranking and results can be found on the [official leaderboard](https://codalab.lisn.upsaclay.fr/competitions/20563#results) under team **79JHotaru**.  

## Architecture  

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ad4fa12-401a-4516-a73f-888abd9468ac" alt="Model Architecture">
</p>

import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['GPT-2', 'GPT-2 Finetuned', 'Mistral-7b-Instruct', 'Look-up Table']
accuracy = [0.0, 0.49, 0.66, 1.0]

# Create subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Accuracy and ROUGE Scores
ax[0].bar(models, accuracy, color='blue', label='Accuracy')
ax[0].set_title('Model Performance – Accuracy')
ax[0].set_ylabel('Scores')
ax[0].legend()

plt.tight_layout()
plt.show()

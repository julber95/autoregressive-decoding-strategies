import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

os.makedirs('utils', exist_ok=True)

# Data for two prompts
ai_results = {
    'Greedy':      {'BLEU': 0.1109, 'ROUGE-2': 0.3333, 'ROUGE-L': 0.5455},
    'Top-k (k=10)':{'BLEU': 0.2575, 'ROUGE-2': 0.3200, 'ROUGE-L': 0.3846},
    'Top-k (k=5)': {'BLEU': 0.1263, 'ROUGE-2': 0.2500, 'ROUGE-L': 0.3871},
    'Nucleus (p=0.9)':{'BLEU': 0.1545, 'ROUGE-2': 0.3333, 'ROUGE-L': 0.5455},
    'Beam (width=5)': {'BLEU': 0.1779, 'ROUGE-2': 0.2963, 'ROUGE-L': 0.3704},
    'Beam (width=10)':{'BLEU': 0.1929, 'ROUGE-2': 0.3333, 'ROUGE-L': 0.3750}
}

climate_results = {
    'Greedy':      {'BLEU': 0.3479, 'ROUGE-2': 0.4545, 'ROUGE-L': 0.5455},
    'Top-k (k=10)':{'BLEU': 0.2889, 'ROUGE-2': 0.4286, 'ROUGE-L': 0.5385},
    'Top-k (k=5)': {'BLEU': 0.1912, 'ROUGE-2': 0.3030, 'ROUGE-L': 0.3636},
    'Nucleus (p=0.9)':{'BLEU': 0.3479, 'ROUGE-2': 0.4545, 'ROUGE-L': 0.5455},
    'Beam (width=5)': {'BLEU': 0.3490, 'ROUGE-2': 0.5000, 'ROUGE-L': 0.5833},
    'Beam (width=10)':{'BLEU': 0.4053, 'ROUGE-2': 0.5000, 'ROUGE-L': 0.5833}
}

ai_df = pd.DataFrame.from_dict(ai_results, orient='index')
climate_df = pd.DataFrame.from_dict(climate_results, orient='index')

# 1. Bar charts for each prompt
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ai_df.plot(kind='bar', ax=axes[0])
axes[0].set_title('AI Prompt: "The development of artificial intelligence"')
axes[0].set_ylim(0, 1)
climate_df.plot(kind='bar', ax=axes[1], color=['orange','green','blue'])
axes[1].set_title('Climate Prompt: "Climate change remains one"')
axes[1].set_ylim(0, 1)
for ax in axes:
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Score')
    ax.legend(title='Metrics')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
# Save figure
plt.savefig('utils/metrics_bar_comparison.jpg', format='jpeg', dpi=300)
plt.show()

# 2. Scatter plot BLEU vs ROUGE-L
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(ai_df['BLEU'], ai_df['ROUGE-L'], label='AI Prompt', marker='o')
ax.scatter(climate_df['BLEU'], climate_df['ROUGE-L'], label='Climate Prompt', marker='s')
for df, prompt in [(ai_df, 'AI'), (climate_df, 'Climate')]:
    for strategy in df.index:
        ax.text(df.loc[strategy,'BLEU']+0.005, df.loc[strategy,'ROUGE-L']+0.005, 
                strategy, fontsize=8)
ax.set_xlabel('BLEU Score')
ax.set_ylabel('ROUGE-L F1 Score')
ax.set_title('Precision (BLEU) vs Fluency (ROUGE-L)')
ax.set_xlim(0, 0.5)
ax.set_ylim(0.3, 0.65)
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
# Save figure
plt.savefig('utils/bleu_vs_rougeL_scatter.jpg', format='jpeg', dpi=300)
plt.show()

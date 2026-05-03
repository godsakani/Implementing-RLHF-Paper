# ReImplementation of RLHF Paper

## What is RLHF?

**Reinforcement Learning from Human Feedback (RLHF)** is an approach for steering language models toward behaviors humans prefer, without relying only on next-token prediction on raw text. A widely used recipe especially in instruction-tuned systems—combines **supervised fine-tuning (SFT)** on demonstrations, **reward modeling (RM)** from human comparisons (or proxies in labs), and **reinforcement learning** (often **PPO** with a KL penalty toward a reference policy) so the model improves on what the reward model scores highly, while **regularization** limits reward hacking or degenerate text.

- **SFT**: Fine-tune on demonstrations (e.g., high-quality answers or, in simplified labs, sentences from a curated dataset) to get a strong starting policy.
- **RM**: A model learns to predict **reward** or **preference** from comparisons so training can scale beyond hand-written demos.
- **RL**: Update the policy to maximize expected reward from the RM, with constraints that keep behavior stable.

InstructGPT ([Ouyang et al., 2022](https://arxiv.org/pdf/2203.02155)) made this recipe visible for instruction-following: SFT aligns format and basics; the RM encodes nuanced preferences; RL improves outputs the RM scores highly while staying close enough to the supervised model that capabilities remain usable.

## Why and how RLHF relates to AI safety and alignment

**Alignment** is about building systems whose behavior matches **human intent and norms**, not only **proxy objectives** like low cross-entropy on internet text. RLHF is one of the first broadly deployed techniques that:

- **Incorporates human judgment at scale**: Preferences and red-team feedback can be distilled into a reward signal used during training, not only at deployment time.
- **Targets behaviors SFT alone misses**: Demonstrations are limited and expensive; optimizing against a learned reward can improve helpfulness, harmlessness, and instruction-following dimensions that are hard to fully supervise.
- **Connects to governance and iteration**: Updating the RM and RL stage reflects updated policies (e.g., refusals, tone, factuality emphasis), giving a lever for organizational values—understood as imperfect and requiring oversight.

RLHF is **not** a complete solution to alignment. Reward models can be **mis-specified**, **biased**, or **gamed**; optimizing against them can produce **sycophancy**, **over-refusal**, or **reward hacking**. Safety work therefore treats RLHF as one component alongside **interpretability**, **red teaming**, **constitutional or rule-based methods**, **RLAIF**, and **post-training monitoring**. Understanding RLHF end-to-end—as in this repo and in [RLHF_in_notebooks](https://github.com/ash80/RLHF_in_notebooks) is a standard foundation for reasoning about modern LLM training and its limits.

## This repository

This repository contains a hands-on **re-implementation** of that RLHF story in three Jupyter notebooks: supervised fine-tuning (SFT), reward model training, and an RLHF stage that builds on the SFT policy and trained reward model. The code follows the same educational structure as the reference notebooks below, adapted for this lab (Google Colab paths, saved checkpoints, and local notebook names).

## References

- **Tutorial-style implementation (heavily referenced):** [ash80/RLHF_in_notebooks](https://github.com/ash80/RLHF_in_notebooks)  step-by-step SFT, reward model training, and PPO-based RLHF in notebooks, using GPT-2 and sentiment on SST-2 as a concrete alignment-style objective.
- **Foundational paper:** Ouyang et al., *Training language models to follow instructions with human feedback*  the InstructGPT / RLHF-as-practice formulation many later systems build on. PDF: [https://arxiv.org/pdf/2203.02155](https://arxiv.org/pdf/2203.02155).

## Notebooks in this repo

| Step | Notebook | Role |
|------|----------|------|
| 1 | `1-supervised_fine_tuning.ipynb` | Supervised fine-tuning of GPT-2 on task data. |
| 2 | `2-reward_model.ipynb` | Train a model with a reward head to approximate human-preferred outcomes (here, sentiment signals from SST-2). |
| 3 | `3-RLHF_implementation.ipynb` | Load the SFT policy and reward model, then run the RL stage: sampling, reward scoring, and policy optimization (PPO-style updates as in the reference notebooks). |

Run them in order. You will need Hugging Face access for pretrained weights where the notebooks download models, and Colab-specific cells assume Google Drive layout for saved `sft_model` and `reward_model` artifacts unless you adjust paths for local runs.

## License

See `LICENSE` in the repository root.

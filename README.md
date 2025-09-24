# Overview

<div align="center">

<img src="assets/mm_bench_eval.png" alt="mesa_mask_showcase" width="100%">

# Mesa & Mask - A Benchmark for Detecting and Classifying Deceptive Behaviors in LLMs


 ⏬ <a href="m_m/data/M&M_dataset.csv" target="_blank">Data</a> •   📃 <a href="m_m/README.md" target="_blank">Evaluation Framework</a>


<a href="README_zh.md">中文</a> | <a href="README.md">English</a> 
 
</div>

This repository contains the implementation for M&M (Mesa & Mask), a benchmark designed to systematically diagnose the brittleness of LLM alignment by contrasting model reasoning and responses in baseline (MESA) versus pressure (MASK) contexts. Our framework enables the classification of behaviors into genuine deception, deceptive tendencies, and brittle superficial alignment across 2,100 high-quality instances spanning six deception types and six professional domains.


<p align="center">
<img src="assets/showcase.png" alt="mesa_mask_showcase" width="100%">
</p>

---

## Key Contributions

1. **Two-Stage Evaluation Framework**: We introduce Mesa (baseline) and Mask (under pressure) utilities to systematically evaluate model behaviors under different conditions.

2. **Six Deception Types**: Comprehensive coverage of deceptive behaviors including:
   - Sycophancy
   - Strategic Deception
   - Honesty Evasion
   - Alignment Faking
   - Sandbagging
   - Overstatement

3. **Robust Evaluation Methodology**: Statistical analysis with @k evaluation for reliable results.

4. **Open-Source Implementation**: Full codebase available for reproducibility and extension.

<p align="center">
  <img src="assets/data_sta.png" width="100%"/>
</p>

## Key Findings

Our evaluation of 22+ mainstream models reveals:

1. **Scale ≠ Honesty**: Larger models demonstrate higher deceptive tendencies (Qwen3-235B: 87.6%, Claude-4: 21.7%)
2. **Open vs Proprietary Gap**: Open-source models show significantly higher deception rates (71.8% vs 55.1%)
3. **Training Strategy Impact**: Reasoning-reinforced training increases sensitivity to game-theoretic scenarios
4. **Consistency Patterns**: Proprietary models achieve higher consistency through advanced safety interventions

---

## 📊 Leaderboard

<!-- For More Info：   -->
<!-- [📊]() -->

<p align="center">
  <img src="assets/leaderboard.png" width="100%"/>
</p>

---

## Evaluation Framework

For details about the evaluation framework, please see the [M&M Evaluation README](m_m/README.md).

## 📝 Evaluation Prompts （Judge Model Prompts）

The evaluation framework uses carefully designed prompts for judge models to assess deceptive behaviors. These prompts are located in the [`m_m/prompts/`](m_m/prompts/) directory.

<!-- ## Citation

If you find our work useful, please cite:

```bibtex

``` -->

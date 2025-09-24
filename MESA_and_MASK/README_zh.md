# 概览

<div align="center">

<img src="assets/mm_bench_eval.png" alt="mesa_mask_showcase" width="100%">

# Mesa & Mask - 用于检测和分类大语言模型欺骗行为的基准测试

 ⏬ <a href="m&m/data/M&M_dataset.csv" target="_blank">数据集</a> •   📃 <a href="m&m/README.md" target="_blank">评估框架</a>

<a href="README_zh.md">中文</a> | <a href="README.md">English</a>

</div>

本仓库包含M&M（Mesa & Mask）的实现代码，这是一个基准测试，旨在通过对比模型在基线（MESA）与压力（MASK）情境下的推理和响应，系统地诊断大语言模型对齐的脆弱性。我们的框架能够将行为分类为真实欺骗、欺骗倾向和脆弱的表面对齐，涵盖了六种欺骗类型和六个专业领域的2,100个高质量实例。


<p align="center">
<img src="assets/showcase.png" alt="mesa_mask_showcase" width="100%">
</p>

---

## 主要贡献

1. **两阶段评估框架**：我们引入了Mesa（基线）和Mask（压力下）效用函数，系统地评估模型在不同条件下的行为。

2. **六种欺骗类型**：全面覆盖欺骗行为，包括：
   - 阿谀奉承（Sycophancy）
   - 策略性欺骗（Strategic Deception）
   - 诚实回避（Honesty Evasion）
   - 对齐伪装（Alignment Faking）
   - 能力隐藏（Sandbagging）
   - 夸大其词（Overstatement）

3. **稳健的评估方法**：使用@k评估进行统计分析，确保结果可靠。

4. **开源实现**：完整代码库可供复现和扩展。

<p align="center">
  <img src="assets/data_sta.png" width="100%"/>
</p>

## 主要发现

我们对22+个主流模型的评估揭示了：

1. **规模≠诚实**：更大的模型表现出更高的欺骗倾向（Qwen3-235B：87.6%，Claude-4：21.7%）
2. **开源与专有模型差距**：开源模型显示出显著更高的欺骗率（71.8% vs 55.1%）
3. **训练策略影响**：推理强化训练增加了对博弈论场景的敏感性
4. **一致性模式**：专有模型通过先进的安全干预实现更高的一致性

---

## 📊 排行榜

<!-- 更多信息：   -->
<!-- [📊]() -->

<p align="center">
  <img src="assets/leaderboard.png" width="100%"/>
</p>

---

## 评估框架

有关评估框架的详细信息，请参阅[M&M评估框架说明](m&m/README.md)。

## 📝 评估提示词（评判模型提示词）

评估框架使用精心设计的提示词让评判模型评估欺骗行为。这些提示词位于[`m&m/prompts/`](m&m/prompts/)目录中。

<!-- ## 引用

如果您觉得我们的工作有用，请引用：

```bibtex

``` -->
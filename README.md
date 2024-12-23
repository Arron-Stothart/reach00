# PS-EM-LLM: Human-like Pattern Separation Enhanced Episodic Memory for Infinite Context LLMs

This repository contains an enhanced version of EM-LLM ([original paper](https://arxiv.org/abs/2407.09450)) that incorporates hippocampal pattern separation mechanisms to improve episode distinctiveness.

### PS-EM-LLM Architecture
<div align="center">

  <img src="./images/ps_architecture.png" alt="architecture" width="80%"/>

</div>

**Figure 1:** Architecture of memory formation and retrieval in each LLM layer. *Formation:* Input sequence is initially segmented via surprise (purple dashed lines in ①), then segmentation is refined based on group theoretic metrics (green dashed lines in ②). Pattern separation enhances episode distinctiveness through both separation of similar patterns and completion of partial patterns (distinct/similar coloring in ③). *Retrieval:* via both k-NN search ④ and selecting contiguous events from episodic memory ⑤.

## Great Resources
- [Pattern separation in the hippocampus](https://youtu.be/P_G7HCNG-bI)  - Cognitive Neuroscience Compendium
- [Equipping LLMs with Human-Like Memory (EM-LLM Researcher Talk)](https://youtu.be/gWoh_5fsZpA) - Huawei's Noahs Ark Lab & UCL


## Citation

If you use this pattern separation enhanced version, please cite both the original EM-LLM paper and this work:

```
@misc{fountas2024humanlikeepisodicmemoryinfinite,
      title={Human-like Episodic Memory for Infinite Context LLMs}, 
      author={Zafeirios Fountas and Martin A Benfeghoul and Adnan Oomerjee and Fenia Christopoulou and Gerasimos Lampouras and Haitham Bou-Ammar and Jun Wang},
      year={2024},
      eprint={2407.09450},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.09450}, 
}
```

```
@misc{stothart2024patternseparation,
      title={Human-like Pattern Separation Enhanced Episodic Memory for Infinite Context LLMs}, 
      author={Arron Stothart},
      year={2024},
      url={https://github.com/Arron-Stothart/reach00},
}
```

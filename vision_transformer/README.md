
# Introduction

This repo implemented the *ViT-base model using **PyTorch** * from the paper of *[An Image is Worth 16 * 16 Words: Transformers for Image Recognition at Scale.](https://arxiv.org/abs/2010.11929)*

# Code Structure


![1709751711286](https://github.com/GuilinXie/Paper_Replicating/blob/main/vision_transformer/results/vision_transformer.png)


# Training Result

* Results from training my own ViT model **from scratch**

![1709753282695](image/README/1709753282695.png)        ![1709753292112](image/README/1709753292112.png)

Firstly, this result shows that the losses are decreasing, and this indicates the model has learned something.

Secondly, the losses are converging. This means that the training works well.

But the accuracies functuate near the probability of 0.35. This means that our model is too large, and our dataset is to small for it to train from scratch.


* Results from  **transfer learning** using

```
torchvision.models.ViT_B_16_Weights.DEFAULT
```

![1709753324773](image/README/1709753324773.png)         ![1709753337975](image/README/1709753337975.png)

After using transfer learning with pretrained ViT_B_16 base model, the accuracy can be as high as 0.95, and the loss are converging faster.

# Prediction Result

```
python predict.py --model ./models/vit_3cls.py  -img_path ./data/img.jpg
```

Result using ViT model training from scratch

![1709756476316](image/README/1709756476316.png)    ![1709756494056](image/README/1709756494056.png)   ![1709756502741](image/README/1709756502741.png)

Results using ViT model training by transfer learning

![1709756544677](image/README/1709756544677.png)  



# Reference

[1]	Paper: [An Image is Worth 16 * 16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929): https://arxiv.org/abs/2010.11929

[2]    Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762): https://arxiv.org/abs/1706.03762

[3]	Online Tutorial: [The IIlustrated Transformer](https://jalammar.github.io/illustrated-transformer/): https://jalammar.github.io/illustrated-transformer/

[4]	Youtube Tutorial: [Let&#39;s build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).https://www.youtube.com/watch?v=kCc8FmEb1nY

[5]    Online Tutorial:  [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/): https://www.learnpytorch.io/

[6]	Bilibili Tutorial: [Deep Learning Knowledge](https://space.bilibili.com/94779326/channel/collectiondetail?sid=1621163): https://space.bilibili.com/94779326/channel/collectiondetail?sid=1621163

[7]	PyTorch documentation: https://pytorch.org/docs/stable/index.html

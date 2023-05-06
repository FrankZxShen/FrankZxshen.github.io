---
title: gradio参数
tags: gradio
categories: 代码阅读
cover: url(/img/Pandora.jpg)
abbrlink: 54f622a3
date: 2023-04-22 16:20:10
---

用于防止端口被占用（windows）

```
netstat -ano | findstr "7860"
```

```text
taskkill /pid 7860 /f
```



- `gradio.inputs.Image`: represents an image input.
- `gradio.inputs.Slider`: represents a numerical input with a slider widget.
- `gradio.inputs.Checkbox`: represents a boolean input with a checkbox widget.
- `gradio.outputs.Image`: represents an image output.
- `gradio.outputs.Textbox`: represents a text output with a text box widget.
- `gradio.outputs.Audio`: represents an audio output.
- `gradio.outputs.Video`: represents a video output.
- `gradio.Interface`: represents the entire user interface, composed of inputs and outputs.
- `gradio.State`: represents a persistent state variable that can be used to store information between function calls.

These parameters can be used to create a wide range of user interfaces for various types of machine learning models and other applications.



```python
txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(bot, chatbot, chatbot)
```

then方法设置要调用的bot函数，chatbot元素作为其输入和输出。chatbot元素传递三次以设置其大小和位置。
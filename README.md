# SmallLaTeXFormulaOCR

This is a small dataset consisting of images of mathematical formulas along with their corresponding LaTeX representations in text format. The dataset is intended for evaluating different LaTeX OCR (Optical Character Recognition) tools.

## Dataset Description

The dataset contains images of LaTeX formulas and their corresponding LaTeX code as text. It is a subset of the larger LaTeX_OCR dataset, which is available on Hugging Face [here](https://huggingface.co/datasets/linxy/LaTeX_OCR/viewer/default/train?p=4).

## Purpose

This dataset can be used to benchmark or test LaTeX OCR models and tools, particularly for recognizing and converting mathematical formulas from images back into LaTeX format.

## Acknowledgments

This dataset is a small part of the original [LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR/viewer/default/train?p=4) dataset. All credits for the larger dataset go to the creators and contributors of LaTeX_OCR.

## Evaluation

Evaluation will be done on Google Colab(T4 GPU). Results and Code can be found in the Evaluation folder. 
The following LaTeX OCR Models will be evaluated:
- [*] LaTeXOCR [Here](https://github.com/lukas-blecher/LaTeX-OCR)
- [] im2latex[Here](https://github.com/d-gurgurov/im2latex)
- [] Sumen [Here](https://github.com/hoang-quoc-trung/sumen)
- [] TexTeller [Here](https://github.com/OleehyO/TexTeller/tree/main)
- [] Pix2Text [Here](https://github.com/breezedeus/pix2text)

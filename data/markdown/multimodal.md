public:: true

- #Public page automatically published
- # OpenAI ChatGPT-4o (omni)
  id:: 66446c0e-93be-431d-93d4-1e5fa36848c5
	- Free to use, for everyone! Not private by default.
	- True multi modality across video, images, and audio.
	- The first of the true publicly accessible models trained without compromise for multi-modality.
	- Multi-lingual across 50 languages, supporting image input and output, real time video input, text to 3D.
	- Empathetic voice to voice with very low latency.
	- [Min Choi on X: "I used GPT-4o to create STL file for 3D model in ~ 20 seconds on my phone. Pretty remarkable what you can generate with AI and simple prompt now. https://t.co/2fbObrpPol" / X (twitter.com)](https://twitter.com/minchoi/status/1790396782200987662)
	- {{twitter https://twitter.com/minchoi/status/1790396782200987662}}
- # Google DeepMind Gemini
	- Gemini is a multimodal LLM capable of inputting and outputting text, understanding images, and generating images.
	- While specific architecture details are scarce, it represents a leap in LLMs interacting with multiple data types.
- ### Multi-Modal Large Language Models (LLMs)
	- **Introduction:**
		- [[Large language models]] are adept at generating coherent text sequences, predicting word probabilities and co-occurrences.
		- Multimodal models extend LLMs capabilities to not just output text, but images and understand multimodal inputs.
	- **Core Concepts:**
		- **LLMs for Text:**
			- LLMs process prompts and generate replies one token at a time, acting as a multiclass classifier.
		- **Image Generation:**
			- Traditional pixel-by-pixel image generation is intractable; hence, a different approach is needed.
			- The solution is treating image generation as a language generation problem, akin to ancient hieroglyphics.
	- **Techniques in Multi-Modal LLMs:**
		- **Autoencoders:**
			- Compress images into a lower-dimensional latent space and then regenerate them, learning crucial properties.
		- **[[Variational Autoencoders]] (VAE) & VQ-VAE:**
			- VAEs add a generative aspect by allowing for new image generation from random latent embeddings.
			- VQ-VAE further discretizes this process, creating a vocabulary of image "words" or tokens.
	- **Implementation:**
		- **Vector Quantization:**
			- Creates a discrete set of embedding vectors forming the vocabulary for our image-based language.
		- **Encoding and Decoding:**
			- Images are encoded to these discrete codes and decoded back to form new or reconstructed images.
	- **Training and Inference:**
		- A mixed sequence of embeddings (words and image tokens) is created for training.
		- The model learns to generate image tokens, forming a coherent sequence with the text, allowing for the generation of images corresponding to text descriptions.
	- **Challenges and Developments:**
		- The importance of quality data over quantity, especially for large, complex models.
		- Ongoing efforts focus on refining data quality, applying safety measures, and improving model transparency.
-
- ```mermaid
  flowchart LR
  A[Text Input] -->|Processed by LLM| B[Text Tokens]
  B -->|Alongside Image Tokens| D[Mixed Embeddings]
  C[Image Input] -->|Encoded via VQ-VAE| E[Image Tokens]
  E --> D
  D -->|Next Token Prediction| F[Generated Sequence]
  F -->|Decoded| G[Output Image & Text]
  ```
-
- Some random links
	- Apple Ferret is a [[Multimodal]] [[Large language models]] from [[Apple]] that can understand and ground anything at any granularity [apple/ml-ferret (github.com)](https://github.com/apple/ml-ferret)
	- [THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型 (github.com)](https://github.com/THUDM/CogVLM)
	- [moondream
		- a tiny vision language model](https://moondream.ai/)
	- Key Papers
		- [Variational Autoencoder (VAE)](https://arxiv.org/abs/1312.6114)
		- [Vector Quantized Variational Autoencoder (VQ-VAE)](https://arxiv.org/abs/1711.00937)
		- [Vector Quantized Generative Adversarial Network (VQ-GAN)](https://compvis.github.io/taming-transformers/)
		- [Gemini](https://assets.bwbx.io/documents/users/iqjWHBFdfxIU/rJRW6x8V4P0g/v0)
		- [Parti](https://sites.research.google/parti/)
		- [DallE](https://arxiv.org/pdf/2102.12092.pdf)
		- [2304.06939.pdf (arxiv.org)](https://arxiv.org/pdf/2304.06939.pdf) C4 model
		- [huggingface/OBELICS: Code used for the creation of OBELICS, an open, massive and curated collection of interleaved image-text web documents, containing 141M documents, 115B text tokens and 353M images. (github.com)](https://github.com/huggingface/OBELICS?tab=readme-ov-file)
		-
		-
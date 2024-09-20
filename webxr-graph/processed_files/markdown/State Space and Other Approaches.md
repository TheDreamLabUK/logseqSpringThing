public:: true

	- # Mamba: Linear-Time Sequence Modelling with Selective State Spaces
		- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752.pdf)
		- **Research Design & Rationale**: The study introduces a new architecture, Mamba, which incorporates a selection mechanism and hardware-aware computation into structured state space models.
		- **Significance**: Addresses the inefficiency of Transformer models with a novel architecture that scales linearly and achieves superior performance.
		- **Real-world Implications**: Potential to improve a wide range of applications in natural language processing, bioinformatics, and other areas where sequence data is prevalent.
		- **Takeaways**: Mamba architecture improves upon structured state space models (SSMs) by adding selectivity and hardware-aware algorithms, achieving linear-time modeling with high-quality performance across several modalities.
		- **Practical Implications**: Provides a more efficient alternative to Transformers, especially beneficial for long sequence data.
		- **Potential Impact**: Could influence future developments in sequence modeling and foundational models across various domains.
		- **Abstract in a nutshell**: Mamba is a novel architecture for sequence modeling that enhances structured state space models (SSMs) with selective mechanisms and hardware-aware algorithms, achieving superior performance and efficiency.
		- **Gap/Need**: Traditional Transformer models have significant computational inefficiency, especially for long sequences. Mamba addresses this by incorporating a selection mechanism and hardware-aware computation in SSMs.
		- **Innovation**: Introduces a selection mechanism in SSMs, allowing input-dependent parameterization and a simplified architecture without attention or MLP blocks, enabling linear-time computation with maintained or enhanced performance.
		- ### Key Quotes
		- "Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length and its performance improves on real data up to million-length sequences."
		- **"This class of models can be computed very efficiently as either a recurrence or convolution with linear or near-linear scaling in sequence length."**
		- "Selective SSMs and by extension the Mamba architecture are fully recurrent models with key properties that make them suitable as the backbone of general foundation models operating on sequences."
		- **How does Mamba achieve linear-time modeling?**: By introducing a selection mechanism in structured state space models and designing a hardware-aware algorithm that avoids materializing expanded states, thereby enhancing computational efficiency.
		- **What improvements does Mamba offer over traditional models?**: It achieves faster inference, linear scaling with sequence length, and competitive or superior performance across various data modalities including language, audio, and genomics.
		- **Can Mamba handle long sequences efficiently?**: Yes, it is specifically designed to address the computational inefficiencies of traditional models like Transformers in handling long sequences, offering linear scaling and improved performance.
		- Mamba outperforms Transformers of the same size in language modelling and matches Transformers twice its size both in pretraining and downstream evaluation, demonstrating its efficiency and effectiveness.
		- **Authors' Views**: The authors propose Mamba as a significant step forward in sequence modeling, addressing the inefficiency of Transformers while maintaining or improving performance.
		- **Comparative Analysis**: Mamba is positioned as superior to existing models, particularly Transformers, in terms of efficiency and scalability
- ## Quick lit survey.  (based on [Mamba-Palooza](https://www.youtube.com/watch?v=Bg1LQ_jWliU))
	- Over 30 new papers and projects since original Mamba paper, with new ones coming out every 1-2 days
	- 60% of papers address vision/image processing, 25% natural language, and the rest cover various applications
	- 80% of papers modified the original Mamba architecture in some way
	- 73% of papers reported state-of-the-art results (though not independently verified yet)
	- Handling of state in Mamba architecture is not always clear in papers, more research needed
	- Swapping out selective state space portion of Mamba in MoE architectures could be an interesting avenue to explore
	- Infrastructure and capital advantages for big tech incumbents in deploying large-scale MoE models
	- ## Mamba Learning Theory and Interpretability
		- ### In-Context Learning (Is Mamba Capable of In-Context Learning?)
			- Mamba can do in-context learning, gradually optimizing internal representations through layers
			- Similar pattern to Transformers: elevation of concepts and more accurate activations through layers, then collapse at final prediction
		- ### Learning Rules from Sequences (Othello Mamba)
			- Mamba learns rules of Othello from just a sequence of moves, achieving higher board accuracy than OthelloGPT
			- More data efficient, but becomes less accurate as game progresses and requires longer training than same-sized Transformer
		- ### Strengths and Weaknesses (Can Mamba Learn How to Learn?)
			- Mamba outperforms Transformers on tasks with irrelevance and noise
			- Mamba struggles with high-precision memory recall compared to Transformers
			- Hybrid MambaFormer model outperforms both Transformer and Mamba on various evaluation tasks
	- ## Mixture of Experts (MoE) Architectures
		- Major force driving frontier models (e.g., GPT-4, Gemini 1.5)
		- MoE-Mamba (MoE-Mamba) achieved same loss as original Mamba with 2.2x less training steps, scaling up to 32 experts
		- BlackMamba scaled up to 2.8B parameters and 8 experts, with generation latency well below Transformer, Transformer MoE, and Mamba
	- # Vision Mamba
		- Majority of Mamba papers (over 60%) address vision/image processing, especially biomedical image segmentation
		- Key themes:
			- Representing data as sequences is crucial
			- Images are not inherently sequential like language, music, or DNA
			- Multi-scan approaches enable handling non-sequential data
			- Hybrid architectures leverage strengths of different models
		- Open questions and challenges:
			- Scaling to larger models and datasets
			- Developing state regularization methods
			- Integrating Mamba with other architectural advances (e.g., memory tokens)
		- Potential for transformative impact, especially in biology and vision applications
		- Approaches:
			- U-Mamba (U-Mamba): Hybrid CNN-SSM architecture outperforming CNN and Transformers in biomedical image segmentation
			- Swin-UMamba (Swin-UMamba): Combines Mamba with ImageNet pre-training, outperforms U-Mamba
			- Vision Mamba: Bidirectional (forward and backward) scanning for learning visual representations
			- VMamba: Four-way "cross-scan" starting at each corner, combining representations
			- VM-UNet (VM-UNet): Applies VMamba's four-way scan to medical image segmentation
			- Mamba-ND: Multi-dimensional sequencing for video and weather data, using sequential SSMs
			- SegMamba: 3D image segmentation with three-way scan, handling long sequences
			- Vivim: Three-way scan for video
			- MambaMorph: Aligns two input images by generating deformation field
		- Key insights:
			- Turning images into sequences is crucial, can be done through multi-scan approaches
			- Combining scans sequentially may be more effective than parallel
			- Mamba enables memory-efficient processing of high-resolution images, promising for edge applications (e.g., robotics)
	- # Mixture of Experts (MoE) Architectures
		- Major force in frontier models (GPT-4, Gemini 1.5)
		- MoE-Mamba and BlackMamba demonstrate MoE's effectiveness with Mamba
		- Open questions around scaling and infrastructure requirements for large-scale MoE-Mamba models
		- [Introducing Jamba: AI21's Groundbreaking SSM-Transformer Model](https://www.ai21.com/blog/announcing-jamba)
	- # Long Context Modeling
		- LongMamba (LongMamba): Generalizes to 40k tokens after training on 16k, nearly perfect on "needle in a haystack" task
		- Evo: Models long DNA sequences, learning a "cell model" analogous to language models' "world model"
		- Potential challenges:
			- Eventual "rotting" of internal states with extreme context lengths
			- Need for state regularization or "pruning" to maintain performance
		- Implications for biology: Foundation models could revolutionize drug discovery and biological research
	- # Potential Applications
		- ## Climate Science?
			- To approach the problem of using Mamba to analyze space-acquired time series image-based climate data from multi-spectrum sensors, we can draw upon several techniques and architectures discussed in the Mamba literature. Here's a proposed approach:
				- Data Preprocessing:
					- Combine multi-spectrum sensor data:
						- Use techniques like MambaMorph to align and merge data from different sensors
						- Generate deformation fields to spatially align images from various sources
				- Normalize data to ensure consistent scales and ranges across sensors
					- Temporal alignment:
						- Use techniques like dynamic time warping (DTW) to align time series data temporally
						- Handle missing data points and inconsistent sampling frequencies
				- Create a unified temporal grid for all data sources
					- Incorporate historical data:
						- Preprocess and align historical records (e.g., weather station data, satellite imagery) with the space-acquired data
						- Use techniques like transfer learning or domain adaptation to handle differences in data modalities and distributions
				- Mamba Architecture:
					- Multi-dimensional sequencing (Mamba-ND):
						- Treat the aligned and preprocessed data as a multi-dimensional sequence (e.g., spatial dimensions, time, and spectral channels)
						- Apply Mamba-ND to capture dependencies across all dimensions
					- Cross-scanning (VMamba or SegMamba):
						- Perform multi-directional scans (e.g., four-way or three-way) to capture spatial dependencies
						- Combine the representations from different scans to obtain a comprehensive understanding of the data
					- Hybrid architectures (U-Mamba or Swin-UMamba):
						- Incorporate convolutional layers (CNN) for local feature extraction
						- Use Mamba layers for capturing long-range dependencies and global context
			- Leverage pre-training on large-scale datasets (e.g., ImageNet) to improve performance
			  high-level diagram of the proposed approach using Mermaid:
			- ```mermaid
			  graph TD
			      A[Multi-spectrum Sensor Data] --> B{Data Preprocessing}
			      C[Historical Data] --> B
			      
			      subgraph Data Preprocessing
			          B --> D[Combine Multi-spectrum Data]
			          B --> E[Temporal Alignment]
			          B --> F[Incorporate Historical Data]
			          
			          D --> G[MambaMorph for Alignment]
			          D --> H[Generate Deformation Fields]
			          D --> I[Normalize Data]
			          
			          E --> J[Dynamic Time Warping]
			          E --> K[Handle Missing Data]
			          E --> L[Create Unified Temporal Grid]
			          
			          F --> M[Align Historical Records]
			          F --> N[Transfer Learning/Domain Adaptation]
			      end
			      
			      B --> O{Mamba Architecture}
			      
			      subgraph Mamba Architecture
			          O --> P[Multi-dimensional Sequencing]
			          O --> Q[Cross-scanning]
			          O --> R[Hybrid Architectures]
			          
			          P --> S[Mamba-ND]
			          P --> T[Capture Dependencies Across Dimensions]
			          
			          Q --> U[VMamba/SegMamba]
			          Q --> V[Multi-directional Scans]
			          Q --> W[Combine Scan Representations]
			          
			          R --> X[U-Mamba/Swin-UMamba]
			          R --> Y[Convolutional Layers for Local Features]
			          R --> Z[Mamba Layers for Long-range Dependencies]
			          R --> AA[Pre-training on Large-scale Datasets]
			      end
			      
			      O --> AB[Comprehensive Data Representation]
			      AB --> AC[Predictive Component]
			      AB --> AD[Up-to-date Data vs. Historical Context]
			      
			      subgraph Additional Considerations
			          AE[Handle Data Quality Issues]
			          AF[Incorporate Domain Knowledge]
			          AG[Leverage Transfer Learning]
			          AH[Evaluate Model Performance]
			          AI[Interpret and Visualize Representations]
			      end
			      
			      AC --> AJ{Output Formats}
			      AJ --> AK[Short-term Forecasts]
			      AJ --> AL[Long-term Projections]
			      
			      AE --> AB
			      AF --> AB
			      AG --> AB
			      AH --> AB
			      AI --> AB
			  ```
		-
			- # Mamba2: A Deep Dive into Enhanced Efficiency and Scalability
				- [Mamba2](https://arxiv.org/abs/2405.21060) represents a significant advancement in the realm of structured state space models (SSMs), cleverly marrying theoretical elegance with practical efficiency. Building upon the foundation laid by its predecessor, Mamba, this architecture tackles the inherent computational challenges of sequence modelling, particularly for long sequences, by introducing a series of innovations that leverage the power of modern hardware. This section delves into the technical intricacies of Mamba2, guided by the insights from the paper "Transformers are SSMs: Generalised Models and Efficient Algorithms Through Structured State Space Duality".
				- ## Bridging the Gap: SSMs, Attention, and Structured Matrices
					- Before diving into the specifics of Mamba2, the paper establishes a crucial theoretical framework that connects three seemingly disparate concepts: structured state space models (SSMs), attention mechanisms, and structured matrices. This framework, termed "structured state space duality" (SSD), lays the groundwork for understanding the core innovations of Mamba2 and its relationship to existing sequence modelling paradigms.
					- The paper reveals a fundamental truth about SSMs: they can be viewed as matrix transformations operating on input sequences. This insight reframes the computation of SSMs as a matrix multiplication problem, where the matrix itself exhibits a specific structure known as semiseparability. Semiseparable matrices are characterized by low-rank submatrices, a property that Mamba2 cleverly exploits for efficiency gains.
					- A particularly important class of semiseparable matrices, the 1-semiseparable matrices, take center stage in understanding the efficiency of autoregressive sequence modelling. These matrices correspond to simple scalar recurrences, the building blocks of many efficient SSM algorithms. The paper demonstrates that the efficient computation of autoregressive attention, a crucial capability for language modelling, is intrinsically linked to the properties of semiseparable matrices.
				- ## Mamba2's Core: The State Space Dual (SSD) Layer
					- The heart of Mamba2 lies in the State Space Dual (SSD) layer, a refined core SSM layer that replaces the S6 layer from Mamba. This layer embodies the principle of sacrificing some expressivity for substantial gains in computational efficiency, achieved by:
					- Unlike Mamba, which allows each element in the state vector to decay independently, Mamba2 groups these elements into chunks and applies the same decay factor to each chunk. This seemingly minor modification unlocks significant speedups by allowing the algorithm to leverage matrix multiplications, operations highly optimised on modern hardware like GPUs with their specialised tensor cores.
					- The chunking strategy in the SSD layer allows Mamba2 to embrace the computational power of matrix multiplications. This shift away from element-wise operations, which are less efficient on modern hardware, results in substantial reductions in training time. The paper's benchmarks show the SSD algorithm to be 2-8x faster than Mamba's optimised selective scan implementation.
					- While the reduced granularity of state decay in the SSD layer might appear to limit the model's expressivity, the paper argues that this can also be viewed as a form of inductive bias. By constraining the model's flexibility, the chunking strategy might actually guide it towards learning more generalisable representations, potentially improving performance on certain tasks.
				- ## Beyond the SSD Layer: Architectural Enhancements
					- Mamba2 goes beyond the SSD layer, incorporating architectural refinements inspired by the established techniques and understanding of attention mechanisms in Transformers:
						- **Sequence Parallelism:** Mamba2 enables splitting the input sequence into smaller chunks and processing them concurrently across multiple devices. This technique is a natural extension of the block decomposition strategy used in the SSD algorithm, further enhancing efficiency for long sequences.
						- **Tensor Parallelism:** The computationally intensive matrix multiplications within Mamba2 are distributed across multiple devices, accelerating training and paving the way for training even larger models. The Mamba2 block is specifically designed to minimise synchronisation points, reducing communication overhead and maximising parallel efficiency.
						- **Parallel Parameter Projections:** Mamba2 streamlines the computation of SSM parameters (A, B, C, X) by computing them in parallel at the beginning of the block, a departure from Mamba's sequential approach. This simplification contributes to faster training and aligns better with parallelisation strategies commonly used in Transformers.
						- **Extra Normalisation:** An additional normalisation layer, such as LayerNorm, GroupNorm, or RMSNorm, is strategically placed before the final output projection to enhance training stability. This addition proves especially beneficial for larger Mamba2 models, mitigating potential instabilities that can arise during training.
				- ## Multi-head Patterns: Learning from Transformers
					- Mamba2 leverages the concept of multi-head attention, a cornerstone of Transformer architectures, to create multiple "heads" of its sequence transformation. Each head operates on a portion of the input sequence with its own set of parameters, allowing the model to capture diverse aspects of the sequential data.
					- The paper reveals a key difference between SSMs and Transformers in how they implement multi-head computations. Mamba2, like its predecessor, adopts a "multi-value attention" (MVA) pattern, where the expansion and contraction matrices (B and C) are shared across all heads. This pattern, a natural outcome of the SSM formulation, stands in contrast to the multi-query attention (MQA) pattern commonly used in Transformers. Empirical evidence suggests that the MVA pattern, unique to SSMs, might contribute to Mamba2's strong performance.
				- ## Trade-offs, Potentials, and Open Questions
					- Despite its advancements, Mamba2 is not without its trade-offs and areas for further exploration.
					- While Mamba2 excels in training efficiency, its impact on inference speed remains an open question. This aspect is particularly crucial for real-world applications where fast inference is paramount. Further research is needed to investigate how to optimise Mamba2 specifically for inference, potentially through state size adjustments or by leveraging specialised hardware.
					- The reduced expressivity of the SSD layer, a consequence of its chunking strategy, is an area that demands deeper understanding. While the paper offers a compelling argument for this as a form of inductive bias, a more detailed analysis is needed to ascertain its impact on specific tasks. Exploring whether this trade-off consistently yields benefits across diverse sequence modelling problems is a crucial direction for future research.
					- The paper hints at the potential of hybrid architectures that combine the strengths of both SSD layers and attention mechanisms. Initial results suggest that a judicious blend of these two paradigms can lead to models that outperform both pure Transformer and pure Mamba2 architectures. Further research into the optimal balance and placement of these layers within a hybrid architecture is a promising avenue for achieving even better performance and efficiency.
			-
			- # Things to do
				- TODO What is the spatial and temporal resolution of the space-acquired data?
				- TODO How far back does the historical data go, and at what resolution?
				- TODO Are there any specific climate variables or phenomena of interest (e.g., temperature, precipitation, extreme events)?
				- TODO What is the desired output format for the predictive component (e.g., short-term forecasts, long-term projections)?
				- TODO Handling data quality issues (e.g., cloud cover, sensor noise)
				- TODO Incorporating domain knowledge (e.g., physical constraints, climate models)
				- TODO Leveraging transfer learning from pre-trained models on similar datasets
				- TODO Evaluating the model's performance using appropriate metrics and validation techniques
				- TODO Interpreting and visualizing the learned representations for stakeholder communication
		- ## Ontological Layer for Metaverse
			- To approach the problem of using Mamba to analyze formal ontological graphs as used by the W3C, we can draw upon several techniques and architectures discussed in the Mamba literature. Here's a proposed approach:
			- Data Preprocessing:
				- Graph Normalization:
					- Ensure consistent formatting and structure of the ontological graphs
					- Handle missing or inconsistent data
					- Normalize node and edge labels
				- Graph Merging:
					- Combine multiple ontological graphs into a single unified graph
					- Resolve conflicts and inconsistencies between different ontologies
					- Establish mappings between equivalent concepts across ontologies
				- Graph Embedding:
					- Generate low-dimensional vector representations of nodes and edges
					- Preserve the semantic relationships and structure of the ontological graphs
					- Use techniques like RDF2Vec, TransE, or Graph Convolutional Networks (GCNs)
			- Mamba Architecture:
				- Graph-Mamba:
					- Adapt the Mamba architecture to handle graph-structured data
					- Utilize the selective state space mechanism to capture long-range dependencies in the ontological graphs
					- Achieve efficient memory usage and reduced computational complexity compared to traditional graph neural networks (GNNs)
				- Multi-dimensional Sequencing (Mamba-ND):
					- Treat the ontological graphs as multi-dimensional sequences (e.g., node features, edge types, and graph structure)
					- Apply Mamba-ND to capture dependencies across all dimensions
			- Hybrid Architectures:
				- Combine Graph-Mamba with other graph neural network architectures (e.g., GCNs, GraphSAGE)
				- Leverage the strengths of both approaches to capture local and global patterns in the ontological graphs
				-
				-
			- ```mermaid
			  graph TD
			   A[Ontological Graphs] --> B(Data Preprocessing)
			   B --> C{Mamba Architecture}
			   C --> D[Graph-Mamba]
			   C --> E[Multi-dimensional Sequencing Mamba-ND]
			   C --> F[Hybrid Architectures]
			   D --> G[Comprehensive Graph Representation]
			   E --> G
			   F --> G
			   G --> H[Ontology Alignment]
			   G --> I[Knowledge Graph Completion]
			   G --> J[Semantic Similarity]
			   ```
				-
				- TODO What are the specific ontologies being used (e.g., RDF, OWL)?
				- TODO Are there any domain-specific requirements or constraints to consider?
				- TODO What are the desired output tasks (e.g., ontology alignment, knowledge graph completion, semantic similarity)?
				- TODO How large and complex are the ontological graphs being analyzed?
				- TODO Handling scalability issues for large-scale ontological graphs
				- TODO Incorporating domain knowledge and ontology-specific constraints
				- TODO Leveraging transfer learning from pre-trained models on similar ontological graphs
				- TODO Evaluating the model's performance using appropriate graph-based metrics and validation techniques
				- TODO Interpreting and visualizing the learned graph representations for ontology engineers and domain experts
- ## Implementations
	- https://www.statespace.info/  Lots of tracked projects here. This is the best place.
	- Can use the https://huggingface.co/docs/accelerate/usage_guides/fsdp for sharded training
	- [[2401.04081] MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts (arxiv.org)](https://arxiv.org/abs/2401.04081)
	- [havenhq/mamba-chat: Mamba-Chat: A chat LLM based on the state-space model architecture 🐍 (github.com)](https://github.com/havenhq/mamba-chat)
	- [[2401.13660] MambaByte: Token-free Selective State Space Model (arxiv.org)](https://arxiv.org/abs/2401.13660) doesn't need tokens!!
-
- # Kolmogorov-Arnold Networks (KANs)
	- A novel type of neural network architecture that has recently gained attention in the field of artificial intelligence. Here are the key points about KANs:
	- Inspiration and design:
		- KANs are inspired by the Kolmogorov-Arnold representation theorem from the 1950s[](https://theaiinsider.tech/2024/05/27/what-are-kolmogorov-arnold-networks-a-guide-to-what-might-be-the-next-big-thing-in-artificial-intelligence/)[](https://arxiv.org/html/2404.19756v4).
		- Unlike traditional Multi-Layer Perceptrons (MLPs) that have fixed activation functions on nodes, KANs have learnable activation functions on edges[](https://theaiinsider.tech/2024/05/27/what-are-kolmogorov-arnold-networks-a-guide-to-what-might-be-the-next-big-thing-in-artificial-intelligence/)[](https://arxiv.org/html/2404.19756v4).
	- Key differences from MLPs:
		- KANs replace linear weights with univariate functions parametrized as splines[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/).
		- They have no linear weights at all - every weight parameter is a learnable univariate function[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/).
	- Potential advantages:
		- Improved accuracy: Smaller KANs can achieve comparable or better accuracy than larger MLPs in data fitting and PDE solving[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/).
		- Better interpretability: KANs can be intuitively visualized and easily interact with human users[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/).
		- Faster neural scaling laws: KANs potentially scale better than traditional neural networks[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/)[](https://arxiv.org/html/2404.19756v4).
	- Applications:
		- KANs show promise in image processing, speech recognition, and financial modeling[](https://theaiinsider.tech/2024/05/27/what-are-kolmogorov-arnold-networks-a-guide-to-what-might-be-the-next-big-thing-in-artificial-intelligence/).
		- They have been used to help scientists (re)discover mathematical and physical laws[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/).
	- Challenges:
		- KANs are more complex to design and implement than traditional neural networks[](https://theaiinsider.tech/2024/05/27/what-are-kolmogorov-arnold-networks-a-guide-to-what-might-be-the-next-big-thing-in-artificial-intelligence/).
		- They require specialized knowledge and are not yet widely adopted[](https://theaiinsider.tech/2024/05/27/what-are-kolmogorov-arnold-networks-a-guide-to-what-might-be-the-next-big-thing-in-artificial-intelligence/).
		- Training speed is currently slower (about 10 times) compared to MLPs of the same size[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/).
	- Implementation:
		- A Python library called "pykan" is available for implementing KANs[](https://www.reddit.com/r/MachineLearning/comments/1chrafb/r_kan_kolmogorovarnold_networks/)
		  
		  4
		  
		  .
		  
		  While KANs show promising results and potential advantages over traditional neural networks, they are still in the early stages of development and research. Further studies and real-world applications are needed to fully understand their capabilities and limitations compared to established neural network architectures.
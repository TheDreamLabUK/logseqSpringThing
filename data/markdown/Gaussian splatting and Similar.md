public:: true

- #Public page automatically published
- {{video https://www.youtube.com/watch?v=lG3g8mYKfqU}}
- # Gaussian Splatting
	- [Instantsplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds](https://instantsplat.github.io/)
	- [The Rise Of 3D Gaussian Splatting: What Is It And How Is It Changing The Immersive Media Industry? — Magnopus](https://www.magnopus.com/blog/the-rise-of-3d-gaussian-splatting)
	- 4D [[Gaussian splatting and Similar]] [with time domain](https://github.com/hustvl/4DGaussians)
	- [[Gaussian splatting and Similar]] [gsgen](https://github.com/gsgen3d/gsgen)
	- Room scale [[Gaussian splatting and Similar]] technique for single lens (#SLAM) [[Scene Capture and Reconstruction]]  [Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting (vladimiryugay.github.io)](https://vladimiryugay.github.io/gaussian_slam/)
	- [Mip-Splatting (niujinshuchong.github.io)](https://niujinshuchong.github.io/mip-splatting/) reduced artifacts in [[Gaussian splatting and Similar]]
	- [Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting (vladimiryugay.github.io)](https://vladimiryugay.github.io/gaussian_slam/)
	- GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise
		- logseq://graph/researchpapers?block-id=6579a51f-5e6d-4570-903f-9458f84e845f
	- Gaussian [[SLAM]] rooms scale scanning
		- logseq://graph/researchpapers?block-id=6579a880-ce7f-4a79-b3d3-9135ff4348b3
		- [Gaussian Splatting SLAM (rmurai.co.uk)](https://rmurai.co.uk/projects/GaussianSplattingSLAM/)  is near real-time
	- [Paper page TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering (huggingface.co)](https://huggingface.co/papers/2401.06003)
	- [Deblurring 3D Gaussian Splatting (benhenryl.github.io)](https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/)
	- [huggingface/gsplat.js: JavaScript Gaussian Splatting library. (github.com)](https://github.com/huggingface/gsplat.js/)
	- [[Gaussian splatting and Similar]] in Houdini
	- [Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers (zouzx.github.io)](https://zouzx.github.io/TriplaneGaussian/) understandable [[Text to 3D and 4D]] from [[Gaussian splatting and Similar]]
	- [dynamic3dgaussians.github.io](https://dynamic3dgaussians.github.io/) using a multi [[Motion Capture]] dome and [[Gaussian splatting and Similar]] for 6DOF [[Human tracking and SLAM capture]]
	- [LangSplat: 3D Language Gaussian Splatting](https://langsplat.github.io/)
- # NeRFs
	- MobileNeRF: This approach adapts NeRFs for mobile devices by exploiting the polygon rasterization pipeline for efficient neural field rendering. It achieves very fast rendering times (0.016-0.017s) but requires long training times[](https://spectrum.ieee.org/ai-graphics-neural-rendering).
	- MobileR2L: This method uses a full CNN-based neural light field model with a super-resolution model in its second stage. It achieves real-time inference on mobile devices while maintaining high image quality, rendering a 1008x756 image of real 3D scenes in 18.04ms on an iPhone 13[](https://spectrum.ieee.org/ai-graphics-neural-rendering).
	- Instant NGP (Neural Graphics Primitives): Developed by NVIDIA, this technique significantly speeds up the training and rendering of NeRFs, allowing for near-instantaneous scene reconstruction[](https://github.com/weihaox/awesome-neural-rendering/blob/master/docs/INTRODUCTION-AND-SURVEY.md).
	- Plenoxels (Plenoptic Voxels): This method replaces neural networks with a sparse 3D grid of spherical harmonics, enabling faster training and competitive quality compared to NeRFs[](https://github.com/weihaox/awesome-neural-rendering/blob/master/docs/INTRODUCTION-AND-SURVEY.md).
	- NGLOD (Neural Geometric Level of Detail): This approach combines neural implicit representations with explicit geometric representations, allowing for multi-resolution rendering and faster training[](https://arxiv.org/abs/2402.00028).
	- NeRF-MAE (Masked AutoEncoders for NeRFs): This technique applies the concept of masked autoencoders to NeRFs for self-supervised 3D representation learning, potentially improving generalization and efficiency[](https://ideas-ncbr.pl/en/research/neural-rendering/).
	- ## NeRFs vs Hardware Acceleration
		- old page, needs [[Update Cycle]]
		- [Neural Rendering and Its Hardware Acceleration: A Review (arxiv.org)](https://arxiv.org/html/2402.00028v1)
		- | Paper                                  | Neural Network Type | Residual Layer | Concatenation Layer | Suitability for Low-end Mobile Hardware |
		  |----------------------------------------|---------------------|----------------|---------------------|----------------------------------------|
		  | GIRAFFE                                | MLP, CNN            | Required       | Required            | 7                                      |
		  | Render Net                             | MLP, CNN            | Not Required   | Required            | 6                                      |
		  | Neural Voxel Renderer                  | MLP, CNN            | Not Required   | Required            | 5                                      |
		  | Neural Volumes                         | MLP, CNN            | Not Required   | Required            | 5                                      |
		  | NeRF                                   | MLP                 | Not Required   | Required            | 8                                      |
		  | NeRF in the Wild                       | MLP                 | Not Required   | Required            | 7                                      |
		  | KiloNeRF                               | MLP                 | Not Required   | Required            | 8                                      |
		  | FastNeRF                               | MLP                 | Not Required   | Required            | 9                                      |
		  | Plenoctrees                            | MLP                 | Not Required   | Required            | 8                                      |
		  | Instant Neural Graphics Primitives     | MLP                 | Not Required   | Required            | 9                                      |
		  | Scene Representation Networks          | MLP                 | Not Required   | Required            | 7                                      |
		  | Extracting Motion and Appearance       | MLP, CNN, Transformer | Required   | Required            | 6                                      |
		  | Instant 3D                             | MLP                 | Not Required   | Required            | 8                                      |
		  | Neural Point Cloud Rendering           | CNN, U-Net          | Not Required   | Required            | 6                                      |
		  | Deep Shading                           | CNN                 | Not Required   | Required            | 6                                      |
		  | Neural Reflectance Fields              | CNN                 | Required       | Not Required        | 7                                      |
		  | Deep Illumination                      | GAN, U-Net          | Not Required   | Required            | 5                                      |
		  | Common Objects in 3D                   | MLP, Transformer    | Required       | Required            | 7                                      |
		  | GeoNeRF                                | Transformer         | Required       | Required            | 7                                      |
		  | Gen-NeRF                               | Transformer         | Required       | Required            | 7                                      |
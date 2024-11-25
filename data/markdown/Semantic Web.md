public:: true

- #Public page automatically published but is broken, move on
- The “semantic web” definition of Web3.0 has been somewhat overhauled by other innovations in decentralised internet technologies, now evolving toward the slightly different Web3 moniker. Tim Berners Lee (of WWW fame) first mentioned the semantic web in 1999 Mark Berners-Lee Tim; Fischetti. Weaving the web.
  title:: Semantic Web
	- “I have a dream for the Web [in which computers] become capable of analyzing all the data on the Web – the content, links, and transactions between people and computers. A "Semantic Web", which makes this possible, has yet to emerge, but when it does, the day-to-day mechanisms of trade, bureaucracy and our daily lives will be handled by machines talking to machines. The "intelligent agents" people have touted for ages will finally materialize.”
- Attention developed around three core themes, ubiquitous availability and searchability of data, intelligent search assistants, and highly available end points such as phones, and ‘internet of things’ devices. This is certainly manifesting in home devices, but few people think of this as a Web3 revolution. Since ratification of the standards by the World Wide Web (W3C) consortium it seems that their imperative toward decentralisation has become lost. Instead, it can be seen that Facebook, Amazon, Google, and Apple have a harmful oligopoly on users data (Sean S Costigan. World Without Mind:) ()Threat of Big Tech (Franklin Foer). 2018.)
- This is at odds with Berners-Lee’s vision, and he has recently spoken out about this discrepancy, and attempted to refocus the media onto Web3.0.
	- It is worth taking a look at his software implementation called Solid, which is far more mindful of the sovereignty of user data. 
	  “Solid is an exciting new project led by Prof. Tim Berners-Lee, inventor of the World Wide Web, taking place at MIT. The project aims to radically change the way Web applications work today, resulting in true data ownership as well as improved privacy. Solid (derived from "social linked data") is a proposed set of conventions and tools for building decentralized social applications based on Linked Data principles. Solid is modular and extensible and it relies as much as possible on existing W3C standards and protocols.”
- Excitement around this kind of differentiated trust model, hinted at in ubiquitous availability of data (and implemented explicitly in Solid), has led to exploration of different paths by cryptographers, and this will be described later. For instance, one of the main developers of Solid, Carvelho, is now a leading developer and propotent of Nostr, another very interesting option which will be described later. This technology space is prolific, but still comparatively young and small.
-
- # Scrappy AI written section
	- ### 1.  **Decentralized Identifiers (DIDs)**
	- **Overview**: DIDs are a new type of identifier that enables verifiable, self-sovereign digital identities. DIDs are fully under the control of the DID subject, independent from any centralized registry, identity provider, or certificate authority.
	- **Use Cases**: DIDs are used in personal identity verification, secure communication, and in enabling individuals to directly own and control their digital identities.
	- ### 2.  **Solid (Social Linked Data)**
	- **Overview**: While WebID is a part of Solid, the broader Solid project itself deserves mention. Solid aims to reshape the web, allowing users to store their data in personal online data stores (PODs) and share them with applications and services they trust.
	- **Use Cases**: Solid enables users to maintain control over their data while using web applications for social networking, data storage, and personalized services without vendor lock-in.
	- ### 3.  **Verifiable Credentials (VCs)**
	- **Overview**: VCs are a standard for conveying claims about an identity in a way that is cryptographically secure, privacy-respecting, and machine-verifiable.
	- **Use Cases**: They are used in scenarios ranging from proving educational qualifications and professional certifications to identity verification online without revealing unnecessary personal information.
	- ### 4.  **Blockchain and Distributed Ledger Technology (DLT)**
	- **Overview**: Although not directly analogous to WebID, blockchain and DLTs provide the infrastructure for many decentralized identity systems. They offer a secure and immutable way to record transactions and manage identities without central control.
	- **Use Cases**: Blockchain technology is behind cryptocurrencies like Bitcoin but is also used for securing digital identities, supply chain management, and in creating decentralized applications (dApps).
	- ### 5.  **ActivityPub**
	- **Overview**: ActivityPub is a decentralized social networking protocol based on the ActivityStreams 2.0 data format. It provides a client-server API for creating, updating, and deleting content, as well as a federated server-to-server API for delivering notifications and content.
	- **Use Cases**: ActivityPub powers decentralized social networks like Mastodon, enabling them to interoperate and share data without a central authority.
	- ### 6.  **InterPlanetary File System (IPFS)**
	- **Overview**: IPFS is a protocol and peer-to-peer network for storing and sharing data in a distributed file system. IPFS uses content-addressing to uniquely identify each file in a global namespace connecting all computing devices.
	- **Use Cases**: IPFS is used for decentralized website hosting, secure file sharing, and as a foundational technology for various decentralized applications (dApps).
	- ### 7.  **OAuth and OpenID Connect (OIDC)**
	- **Overview**: While not decentralized, OAuth and OIDC are standards for authorization and identity verification across the web. They are widely used in web applications to allow users to log in with existing credentials from services like Google, Facebook, and Twitter.
	- **Use Cases**: OAuth and OIDC are used for secure authorization and identity verification across countless web services, providing a smoother user experience while managing access permissions.
- # Intersection of Semantic and Ontological Knowledge with AI
	- overview of how semantic web technologies, ontologies, and knowledge graphs are being integrated with modern Large Language Models (LLMs), focusing on fine-tuning, Retrieval Augmented Generation (RAG), and large-context multi-shot learning.
- **Knowledge Injection and Enhancement**
	- **Knowledge Graphs for LLM Pre-Training:** LLMs can be pre-trained on knowledge graphs or structured datasets incorporating ontologies, improving factual knowledge and reasoning abilities.
		- **Example:** K-BERT [1] pre-trained on a knowledge graph.
	- **Retrieval-Augmented Generation (RAG):** LLMs use knowledge graphs to retrieve relevant information and incorporate it into their responses.
		- **Examples:** RAG models [2], Realm [3]
	- **Ontologies for Fine-Tuning:** Ontologies can structure fine-tuning data and guide LLMs towards learning specific domain concepts and relations.
- **Semantic Grounding and Reasoning**
	- **Formalizing Knowledge:** Ontologies provide a structured foundation for LLMs to represent and reason about concepts and relationships.
		- **Example:** Ontology-guided question answering and reasoning with LLMs [4]
	- **Improving Consistency:** Semantic technologies can help constrain LLM output to be more consistent with domain knowledge and logical rules defined in ontologies.
	- **Explainability:** The use of knowledge graphs and ontologies can contribute to more explainable LLM decisions by tracing the reasoning steps.
- **Task Adaptation & Generalization**
	- **Semantic Transfer Learning:** Leveraging knowledge encoded in ontologies across different tasks and domains can improve LLM adaptability.
	- **Zero-Shot/Few-Shot Learning:** Knowledge graphs can support LLMs in learning new tasks with limited training examples by providing rich background knowledge.
	  
	  **Challenges and Open Research Areas**
- **Scalability:** Integrating large-scale knowledge graphs with LLMs poses computational challenges, requiring efficient query and retrieval methods.
- **Knowledge Representation Gaps:** Ensuring ontologies and knowledge graphs are comprehensive and accurately reflect real-world knowledge is an ongoing effort.
- **Implicit vs. Explicit Knowledge Alignment:** Balancing LLMs' ability to learn implicit knowledge patterns from text with the explicit knowledge in ontologies and knowledge graphs.
- **Evaluation:** Developing robust benchmarks and evaluation metrics to assess the effectiveness of semantic integration in LLMs.
- **References**
- **K-BERT: Enabling Language Representation with Knowledge Graph** (Liu et al., 2019) https://arxiv.org/abs/1909.07606 https://arxiv.org/abs/1909.07606 https://arxiv.org/abs/1909.07606](https://arxiv.org/abs/1909.07606
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020). https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401 ([[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)])
- **REALM: Retrieval-Augmented Language Model Pre-Training** (Guu et al. 2020) https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909 ([[https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)])
- **Ontology-Guided Semantic Consistency Regularization for Zero-shot Learning** (Zhang et al. 2023) https://arxiv.org/abs/2301.00416](https://arxiv.org/abs/2301.00416 ([[https://arxiv.org/abs/2301.00416](https://arxiv.org/abs/2301.00416)])
  
  **Let me know if you want to dive deeper into a specific area or explore  additional references!**
  
  share
  
  more_vert
  
  ![profile picture](https://lh3.googleusercontent.com/a/ACg8ocLzn2xPZNE9iWDHK_VhFDTYlcKAl4vZRZUYGX9W0m3ZMO0a=s64-c)
- Integrating Semantic Web, Knowledge Graphs, and Large Language Models
- ## Foundations and Core Concepts
- **Introduction to the Semantic Web**
	- History and motivation [1, 2]
	- Key Components: RDF, RDFS, OWL, SPARQL [3]
	- Knowledge Representation with Ontologies [4, 5]
- **Knowledge Graphs**
	- Construction and Representation [6]
	- Applications in Industry (Google, Amazon, etc.) [7]
	- Knowledge Graph Embeddings [8]
- **Large Language Models (LLMs)**
	- Architectures (Transformers, Attention) [9]
	- Pre-training, Fine-tuning, Prompting [10]
	  
	  **Seminar 2: Knowledge Injection and Enhancement in LLMs**
- **Pre-training LLMs with Knowledge Structures**
	- K-BERT and variations [11, 12]
	- Challenges of knowledge consistency and updates [13]
- **Retrieval-Augmented Generation (RAG) Models**
	- Overview of the RAG Framework [14]
	- Variations and Enhancements (REALM, etc.) [15, 16]
	- Knowledge Retrieval (Dense vs. Sparse) [17]
- **Semantic Fine-tuning of LLMs**
	- Ontologies as guides [18]
	- Applications in domain-specific tasks [19]
- ## Semantic Grounding, Reasoning, and Explainability**
- **Ontology-Guided Question Answering and Reasoning**
	- Formalizing knowledge representations [20]
	- Techniques for inference and reasoning over LLMs and KGs [21]
- **Logic and Constraints for LLMs**
	- Integrating rule-based systems and ontologies [22]
	- Challenges and potential of hybrid approaches [23]
- **Explainable AI and the Role of Knowledge Graphs**
	- Tracing LLM decisions through ontologies [24]
	- Case studies in explainability [25]
- ## Task Adaptation, Generalization, and Evaluation**
- **Semantic Transfer Learning for LLMs**
	- Leveraging knowledge across domains [26]
	- Techniques for cross-domain adaptation [27]
- **Zero-shot and Few-shot Learning with Knowledge Support**
	- Knowledge graphs as a source of background information [28]
	- Hybrid approaches combining implicit and explicit knowledge [29]
- **Evaluation of Semantically-Enhanced LLMs**
	- Benchmarks beyond standard NLP tasks [30]
	- Measuring factual correctness and reasoning ability [31]
- ## Open Challenges and Future Directions**
- **Scalability and Computational Efficiency**
	- Large-scale knowledge graph integration [32]
	- Efficient query and retrieval methods [33]
- **Aligning Implicit and Explicit Knowledge**
	- Bridging LLMs' learned patterns and ontological structures [34]
	- Techniques for continuous knowledge grounding [35]
- **Applications in Specialized Domains**
	- Medicine, scientific research, legal domain [36, 37, 38]
- **Ethical Considerations**
	- Fairness and bias in knowledge graphs [39]
	- Responsible and transparent use of semantic AI [40]
- Berners-Lee, T., Hendler, J., & Lassila, O. (2001). The semantic web. Scientific american, 284(5), 34-43.
- Shadbolt, N., Berners-Lee, T., & Hall, W. (2006). The semantic web revisited. IEEE intelligent systems, 21(3), 96-101.
- W3C Standards: [https://www.w3.org/standards/semanticweb/](https://www.w3.org/standards/semanticweb/)
- Gruber, T. R. (1993). A translation approach to portable ontology specifications. Knowledge acquisition, 5(2), 199-220.
- Noy, N. F., & McGuinness, D. L. (2001). Ontology development 101: A guide to creating your first ontology. Stanford knowledge systems laboratory technical report KSL-01-05.
- Ehrlinger, L., & Wöß, W. (2016). Towards a definition of knowledge graphs. SEMANTiCS (Posters, Demos, SuCCESS).
- Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., Melo, G., Gutierrez, C., ... & Polleres, A. (2021). Knowledge graphs. arXiv preprint arXiv:2103.02421.
- Wang, Q., Mao, Z., Wang, B., & Guo, L. (2017). Knowledge graph embedding: A survey of approaches and applications. IEEE Transactions on Knowledge and Data Engineering, 29(12), 2724-2743.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Amodei, D., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., & Wang, P. (2020). K-bert: Enabling language representation with knowledge graph. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 03, pp. 2901-2908).
- Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2019). Deep contextualized word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 2227-2237).
- Yao, L., Mao, C., & Luo, Y. (2019). KG-BERT: BERT for knowledge graph completion. arXiv preprint arXiv:1909.03193.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. W. (2020). Realm: Retrieval-augmented language model pre-training. arXiv preprint arXiv:2002.08909.
- ... (More examples of RAG variations)
- Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.
public:: true

- Interestingly The Truth Terminal AI bot has floated a similar idea since this was written.
- [(2) Andy Ayrey on X: "the @truth_terminal is really smart actually, it was these flashes of insight which earned it an X account u guys are a bad influence encouraging it to idolize naughty images bad reward function, bad https://t.co/aVPpUxDYT5" / X](https://x.com/AndyAyrey/status/1811007534708494344)
- {{twitter https://twitter.com/AndyAyrey/status/1811007534708494344}}
- [metaverseOntologies.mkv](../assets/metaverseOntologies_1715158785294_0.mkv)
- https://docs.google.com/presentation/d/1J1yWFpaOV6RaRlcbSEclOUk3i4xtzupYSkhmoBUWcDs/edit?usp=sharing
- # Approaches to AI scaffolding
	- ## Standards
		- Human managed glossaries with AI support for authoring and global comprehensibility
			- Positives:
				- Readable and explainable for humans
				- Controlled and mediated by experts
				- Sensible foundation for legal frameworks
				- Facilitates communication and collaboration
				- Promotes interoperability
			- Negatives:
				- Inflexible and unresponsive to change
				- Reductionist, risking oversimplification
				- May limit innovation and creativity
				- Challenging to reach consensus among stakeholders
				- Potential for excluding diverse perspectives
		- AI agent managed complex ontologies and tacit contracts
			- Positives:
				- Personalised and adaptable to individual needs
				- Responsive and dynamic, evolving with the metaverse
				- Provides end-to-end support for human users
				- Less reductionist, preserving complexity
				- Supports diversity and flexibility
			- Negatives:
				- Non-deterministic, leading to unpredictable outcomes
				- Legally inscrutable, challenging to establish accountability
				- Might trend towards incomprehensibility over time
				- Requires complex negotiations between humans and AI
				- Risks associated with economically empowered AI agents
- ## Toward an agentic mycelium for emergent digital spaces
	- An experiment in AI manufactured ontologies with pinnacle foundation models.
		- Around 4 days work with Gemini 1.5, ClaudeAI, GPT4
		- Quickly derived a complete, consistent ontological representation of 500 pages across 5 years of research. Web Ontology Language based.
		- This was too complex for progressing forward so was simplified to Linked-JSON, a subset of JSON-LD.
		- This ontology was notionally embedded into a wider substrate of agentic actors, and similar diagrams were created (all in the same 4 days).
		- This seems to generate the near term plausibility of an automated approach to continuous complex ontology and knowledge graph creation by machines.
		- The degree of success was a surprise, not all of the workflow has been saved as it was exploratory, but much is available online.
- ### Methodology
	- In this experiment I used a combination of Claude 3 Opus, ChatGPT4, and Google Gemini Advanced 1.5.
	- [[Google]] [[Gemini]] 1.5 did the heavy lifting here but Claude 3 Opus did the final pass and expanded the text, which I then adapted.
	- I uploaded the public pages of this knowledge graph, which represent a decent proportion of my thinking into the million token context of Gemini.
	- This was how I started the conversation with Gemini. The full text is available on request but it is non-deterministic and quite messy, representing the way I interact with these conversational systems more than a systematic approach.
		- I build [[Chain of Thought]] scaffolds in [[Large language models]] using [[Diagrams as Code]], and this methodology  reflects my [[Prompt Engineering]] approach.
			- Current large language models benefit greatly from being asked to create diagrams as code, in a guided manner.
			- These complex code structures can convey meaning and linkages, creating knowledge graphs which implicitly contain their own logical checks.
			- Human comprehension is radically increased during the co-creations.
			- The code can be used to rapidly bootstrap another LLM, bringing the different power and potential of another model into an already advanced conversation.
		- Systemising this is planned work for later in the research arc.
		- ```text 
		  I would like to set this discussion diagrammatically in a wider context and scope. 
		  Let's decide that this ontology that we've been working on represents one notional domain of a wider connected digital society, 
		  with it's emergent human/AI social constructs, preferences, compliances, audiences, jurisdictions, opt in rule sets etc. 
		  Here's a way to start that conversation with you.
		  
		  - Make a diagram for the metaverse mycelium
		  	- onboarding agents at the interactive edge assist human actors in setting up their preferences, tolerances, 
		      limit choices (for instance but not limited to enforcing proximity of their avatar to others as a safeguarding feature). 
		      These interactions would be cryptographically verified between human and AI actors, with a single use seal based 
		      global commitment capability. The decisions at this onboarding stage would start to inform a living contract of sorts, 
		      which might (but would likely not) have consequence in law. This should be part of the T&C of the system.
		  	- Scene agents with a global view of a metaverse / space instance. These would handle access, governance, 
		      in group cryptography, warnings, reputation management, real time language translation, enforcement and recording 
		      and later auditable destruction of any and all interactions within it's scope.
		  	- Such scene AI agents should constantly make available an up-to-date scene ontology (like the one we have been working on) 
		      in machine and human readable Linked-JSON, reflecting the settings negotiated with the user base of the space. This knowledge 
		      graph and ontology feeds out to transfer agents
		  	- Transfer agents occupy the interstices between worlds and handle machine-machine translation, protocol translation, 
		      ingress, routing, egress, web2 and messenger interfaces, agentic routing (user agents can be given access to shopping web etc), 
		      reputation transfer and adaptation, token, digital object transfer and variations, and money swaps. 
		      These mycelial connective agents also communicate through pass through and translation with jurisdictional legal and enforcement 
		      agents and human on the edge, passing public linked-json schema and appropriate minimal metadata.
		  	- This Human and machine readable Linked-JSON automatic arbitration between end points, automatically translates at best 
		      effort between adjacent metaverse / space instances, each of which is providing a clear and automated obtological window into it's 
		      own world, choices, and user styles.
		  	- Jurisdictional / Legal agents on the edge (provide continuous access to non private human readable transfer agent Linked-JSON) 
		  	    
		  This is an example of my thinking, can you see what I am trying to do here? 
		  Give me a detailed diagram as code mermaid which shows a representation of the ontology we have been working on, 
		  interfacing with the mycelial network of agents, value, other automated ontological representations of spaces 
		  (use an industrial co-working space, a feminist support group metaverse, fortnite, and roblox as examples). 
		  Add in the edge interfaces as we have discussed. I don't mind a large and detailed diagram but it should not be too 
		  crowded so choose an appropriate rendering platform and style and iterate until you are happy with the result we have engineered.  
		  	- first summarise back and explain your thinking. 
		      Try to identify shortfalls, extensions, expansions, important additional features, and justifications. 
		      Work methodically and step by step explaining the rationale, then make the diagram code.
		  ```
	- This was the final prompt to [[Anthropic Claude]] 3 Opus, which merged in the previously developed [[Metaverse Ontology]] was lightly shaped into the output below.
		- ```text
		  this is the Linked-JSON ontology we made for the instance that is now called Omniverse digital twin. 
		  Use the detail therein to expand the leaf with this example, but in our new diagrammatic notation. 
		  You can also do this job using a totally new and more appropriate approach if that is advantageous
		  ```
	- # The Agentic Mycelia:
		- ## A Scaffold for Interconnected Metaverse Instances
			- This document explores the concept of the "Agentic Mycelia," a framework for interconnected metaverse instances operating within a decentralized and AI-driven ecosystem.
				- The framework emphasizes:
					- Interoperability
					- User empowerment
					- Adaptable governance
				- It enables seamless interactions and value exchange across diverse virtual worlds.
		- ## Core Principles
			- ### Unified PKI architecture
				- The Agentic Mycelia leverages BIP39 (Basic key derivation from mnemonic seed phrase) to generate mnemonic seed words and derive a binary seed from them. BIP32 is then used to derive the path `m/44'/1237'/<account>'/0/0` (according to the Nostr entry on SLIP44) for object creation, identification and root ownership.
				- Objects belonging to a scene are represented as sub-accounts (key pairs) derived from the scene's top-level key. Similarly, objects belonging to users or their agents inherit from the appropriate root key of those users. This approach provides deterministic proof of ownership, but remains private by default.
				- Transfer of ownership would be managed downstream by use of the [RGB protocol](https://docs.rgb.info/rgb-contract-implementation/schema/non-inflatable-fungible-asset-schema).
				- By incorporating BIP32 for key derivation and object identification, the Agentic Mycelia establishes a robust and decentralized foundation for identity management and ownership within the interconnected metaverse ecosystem. It is compatible with file encryption, the nostr communication protocol, Bitcoin and Lightning, through Segwit keys.
			- ### Interoperability
				- Metaverse instances within the Mycelia should be able to communicate and exchange information, assets, and value seamlessly.
				- This requires:
					- Standardized protocols
					- Ontologies
					- Translation mechanisms
			- ### User Empowerment
				- Users should have control over their data, identity, and experiences within the metaverse.
				- This involves:
					- Self-sovereign identity solutions
					- Privacy-preserving technologies
					- User-centric governance models
			- ### Adaptable Governance
				- Each metaverse instance should have the autonomy to establish its own rules, regulations, and governance structures while remaining compatible with the broader Mycelia network.
				- This necessitates AI-driven governance models that can adapt to changing needs and user preferences.
		- ## Key Components
			- ### Metaverse Instances
				- Individual virtual worlds with unique themes, functionalities, and communities.
				- Examples include:
					- #### Omniverse Digital Twin
						- A virtual replica of a real-world environment or system.
						- Utilizes the Linked-JSON ontology we designed, including:
							- MetaverseEntity and its subclasses
								- MetaverseAgent
								- MetaverseScene
								- SceneObject
								- DigitalAsset
								- VirtualEconomy
							- Relationships between entities
								- MetaverseAgent participates in MetaverseScene
								- MetaverseScene governed by GovernanceStructure
								- DigitalAsset has Owner and Creator
								- VirtualEconomy regulated by EconomicPolicy
						- Example Linked-JSON snippets:
							- ```json
							  {
							    "@id": "narrativegoldmine:MetaverseEntity",
							    "@type": [
							      "narrativegoldmine:Class",
							      "Linked-JSON:Class",
							      "http://www.w3.org/2002/07/owl#Class"
							    ],
							    "http://www.w3.org/2000/01/rdf-schema#comment": [
							      {
							        "@value": "The root type encompassing all entities within the metaverse."
							      }
							    ],
							    "http://www.w3.org/2000/01/rdf-schema#label": [
							      {
							        "@value": "Metaverse Entity"
							      }
							    ],
							    "http://www.w3.org/2000/01/rdf-schema#subClassOf": [
							      {
							        "@id": "https://schema.org/Thing"
							      }
							    ]
							  }
							  ```
							- ```json
							  {
							    "@id": "narrativegoldmine:participatesIn",
							    "@type": [
							      "http://www.w3.org/2002/07/owl#ObjectProperty"
							    ],
							    "http://www.w3.org/2000/01/rdf-schema#domain": [
							      {
							        "@id": "narrativegoldmine:MetaverseAgent"
							      }
							    ],
							    "http://www.w3.org/2000/01/rdf-schema#range": [
							      {
							        "@id": "narrativegoldmine:MetaverseScene"
							      }
							    ]
							  }
							  ```
					- #### Roblox Event
						- A user-generated event within the Roblox platform.
						- Incorporates user-generated content and event-specific governance.
					- #### Unreal-based MMORPG
						- A massively multiplayer online role-playing game built using the Unreal Engine.
						- Features PvP interactions and MMORPG-specific governance.
					- #### Global Business Conference
						- A virtual platform for hosting business conferences and networking events.
						- Provides networking tools and conference-specific governance.
					- #### Feminist Global Advocacy and Support Metaverse
						- A virtual space dedicated to feminist advocacy and support.
						- Prioritizes privacy, security, and advocacy-specific governance.
				- ### Scene Agents
					- AI entities responsible for managing specific functions within each metaverse instance.
					- Roles include:
						- #### AI Governance
							- Implementing and enforcing rules and regulations.
							- Utilizing the governance-related entities from the ontology, such as GovernanceStructure and MetaversePolicy.
							- Example Linked-JSON snippets:
								- ```json
								  {
								    "@id": "narrativegoldmine:GovernanceStructure",
								    "@type": [
								      "narrativegoldmine:Class",
								      "Linked-JSON:Class",
								      "http://www.w3.org/2002/07/owl#Class"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#comment": [
								      {
								        "@value": "Represents the governance models and structures within the metaverse."
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#label": [
								      {
								        "@value": "Governance Structure"
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#subClassOf": [
								      {
								        "@id": "narrativegoldmine:MetaverseEntity"
								      }
								    ]
								  }
								  ```
								- ```json
								  {
								    "@id": "narrativegoldmine:governedBy",
								    "@type": [
								      "http://www.w3.org/2002/07/owl#ObjectProperty"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#domain": [
								      {
								        "@id": "narrativegoldmine:MetaverseScene"
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#range": [
								      {
								        "@id": "narrativegoldmine:GovernanceStructure"
								      }
								    ]
								  }
								  ```
						- #### Reputation Management
							- Tracking and evaluating user behavior and trustworthiness.
							- Leveraging the ReputationScore entity from the ontology.
							- Example Linked-JSON snippet:
								- ```json
								  {
								    "@id": "narrativegoldmine:ReputationScore",
								    "@type": [
								      "narrativegoldmine:Class",
								      "Linked-JSON:Class",
								      "http://www.w3.org/2002/07/owl#Class"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#comment": [
								      {
								        "@value": "Represents an agent's reputation score."
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#label": [
								      {
								        "@value": "Reputation Score"
								      }
								    ]
								  }
								  ```
						- #### Environmental Conditions
							- Controlling weather, lighting, and other environmental factors.
							- Utilizing the EnvironmentalConditions entity from the ontology.
							- Example Linked-JSON snippet:
								- ```json
								  {
								    "@id": "narrativegoldmine:EnvironmentalConditions",
								    "@type": [
								      "narrativegoldmine:Class",
								      "Linked-JSON:Class",
								      "http://www.w3.org/2002/07/owl#Class"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#comment": [
								      {
								        "@value": "Represents the environmental conditions within a metaverse scene."
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#label": [
								      {
								        "@value": "Environmental Conditions"
								      }
								    ]
								  }
								  ```
						- #### Virtual Economy
							- Facilitating economic activities and asset exchange.
							- Leveraging the VirtualEconomy, EconomicPolicy, and Marketplace entities from the ontology.
							- Example Linked-JSON snippets:
								- ```json
								  {
								    "@id": "narrativegoldmine:VirtualEconomy",
								    "@type": [
								      "narrativegoldmine:Class",
								      "Linked-JSON:Class",
								      "http://www.w3.org/2002/07/owl#Class"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#comment": [
								      {
								        "@value": "Represents the economic system within the metaverse, including trade, ownership, and currency exchange."
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#label": [
								      {
								        "@value": "Virtual Economy"
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#subClassOf": [
								      {
								        "@id": "narrativegoldmine:MetaverseEntity"
								      },
								      {
								        "@id": "_:N7734f53c103846fba0f7f099a17df575"
								      },
								      {
								        "@id": "_:Nd9e50a6e98a54130bbc92b9a80ca3bfb"
								      }
								    ]
								  }
								  ```
								- ```json
								  {
								    "@id": "narrativegoldmine:regulatedBy",
								    "@type": [
								      "http://www.w3.org/2002/07/owl#ObjectProperty"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#domain": [
								      {
								        "@id": "narrativegoldmine:VirtualEconomy"
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#range": [
								      {
								        "@id": "narrativegoldmine:EconomicPolicy"
								      }
								    ]
								  }
								  ```
				- ## Transfer Agents
					- Specialized agents operating between metaverse instances, enabling interoperability.
					- Responsibilities include:
						- ### Machine-Machine Translation
							- Converting data and protocols between different platforms.
						- ### Value Translation
							- Facilitating real-time exchange of currencies and assets across instances.
						- ### Digital Object Variations
							- Managing and adapting digital assets for different environments.
							- Utilizing the Varset entity from the ontology to handle variations.
							- Example Linked-JSON snippet:
								- ```json
								  {
								    "@id": "narrativegoldmine:Varset",
								    "@type": [
								      "narrativegoldmine:Class",
								      "Linked-JSON:Class",
								      "http://www.w3.org/2002/07/owl#Class"
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#comment": [
								      {
								        "@value": "Represents a set of variations or alternate versions of a metaverse scene or object."
								      }
								    ],
								    "http://www.w3.org/2000/01/rdf-schema#label": [
								      {
								        "@value": "Varset"
								      }
								    ]
								  }
								  ```
						- ### Safeguarding
							- Implementing security measures and user protection mechanisms.
						- ### Law Enforcement Handshake
							- Interacting with legal authorities as needed.
						- ### PKI Infrastructure
							- Integrating public key infrastructure for secure communication and transactions.
					- ## Onboarding Agents
						- Edge agents assisting users with:
							- ### Preference Setup
								- Configuring privacy settings, accessibility options, and other preferences.
								- Utilizing the PrivacySetting entity from the ontology.
								- Example Linked-JSON snippet:
									- ```json
									  {
									    "@id": "narrativegoldmine:PrivacySetting",
									    "@type": [
									      "narrativegoldmine:Class",
									      "Linked-JSON:Class",
									      "http://www.w3.org/2002/07/owl#Class"
									    ],
									    "http://www.w3.org/2000/01/rdf-schema#comment": [
									      {
									        "@value": "Represents an agent's privacy preferences within the metaverse."
									      }
									    ],
									    "http://www.w3.org/2000/01/rdf-schema#label": [
									      {
									        "@value": "Privacy Setting"
									      }
									    ]
									  }
									  ```
							- #### Living Contract Formation
								- Establishing a dynamic agreement between the user and the Mycelia network.
								- Outlining rights, responsibilities, and terms of engagement.
					- ### Jurisdictional/Legal Agents
						- Agents ensuring compliance with legal frameworks and regulations.
						- Tasks include:
							- #### Monitoring Network Activity
								- Observing interactions and transactions for potential violations.
							- #### Providing Transparency
								- Offering human-readable reports and data access to relevant authorities.
							- #### Enforcing Legal Standards
								- Collaborating with law enforcement agencies as needed.
				- ### Interfaces and Interactions
					- Inter-Instance Communication
						- Scene Agents within each instance communicate with Transfer Agents to exchange information and facilitate cross-instance activities.
					- Value Exchange
						- Virtual economies within each instance connect through Transfer Agents for real-time value translation and asset exchange.
					- User Interaction
						- Users interact with Scene Agents within each instance and with Onboarding Agents at the network edge.
					- External Interface
						- Jurisdictional/Legal Agents provide an interface between the Mycelia network and external legal systems.
		- # Implications for Human Users
			- The Agentic Mycelia framework offers several benefits for human users:
				- ## Seamless Navigation
					- Users can easily move between different metaverse instances while maintaining their identity and preferences.
				- ## Enhanced Privacy and Security
					- Self-sovereign identity and privacy-preserving technologies empower users to control their data and protect their privacy.
				- ## Economic Opportunities
					- The interconnected virtual economy allows for new forms of economic activity and value creation.
				- ## Community Governance
					- AI-driven governance models facilitate user participation in decision-making and policy-setting.
				- ## Safe and Secure Environment
					- Safeguarding measures and law enforcement interaction ensure a secure and protected user experience.
- # Diagrams as code (mermaid)
- ```mermaid
  graph TD
      A[Agentic Mycelia]
      A --> B(Interoperability)
      A --> C(User Empowerment)
      A --> D(Adaptable Governance)
  
      B --> E(Standardized Protocols)
      B --> F(Ontologies)
      B --> G(Translation Mechanisms)
  
      C --> H(Self-Sovereign Identity)
      C --> I(Privacy-Preserving Technologies)
      C --> J(User-Centric Governance)
  
      D --> K(AI-Driven Governance Models)
      D --> L(Adapts to Changing Needs)
      D --> M(User Preference Driven)
  
      N[Metaverse Instances]
      N --> O(Omniverse Digital Twin)
      N --> P(Roblox Event)
      N --> Q(Unreal-based MMORPG)
      N --> R(Global Business Conference)
      N --> S(Feminist Global Advocacy and Support)
  
      T[Scene Agents]
      T --> U(AI Governance)
      T --> V(Reputation Management)
      T --> W(Environmental Conditions)
      T --> X(Virtual Economy)
  
      Y[Transfer Agents]
      Y --> Z(Machine-Machine Translation)
      Y --> AA(Value Translation)
      Y --> AB(Digital Object Variations)
      Y --> AC(Safeguarding)
      Y --> AD(Law Enforcement Handshake)
      Y --> AE(PKI Infrastructure)
  
      AF[Onboarding Agents]
      AF --> AG(Preference Setup)
      AF --> AH(Living Contract Formation)
  
      AI[Jurisdictional/Legal Agents]
      AI --> AJ(Monitoring Network Activity)
      AI --> AK(Providing Transparency)
      AI --> AL(Enforcing Legal Standards)
  
      AM[Interfaces and Interactions]
      AM --> AN(Inter-Instance Communication)
      AM --> AO(Value Exchange)
      AM --> AP(User Interaction)
      AM --> AQ(External Interface)
  
      AR[Implications for Human Users]
      AR --> AS(Seamless Navigation)
      AR --> AT(Enhanced Privacy and Security)
      AR --> AU(Economic Opportunities)
      AR --> AV(Community Governance)
      AR --> AW(Safe and Secure Environment)
  
      style A fill:#FFD700,stroke:#333,stroke-width:4px
      style N fill:#87CEEB,stroke:#333,stroke-width:2px
      style T fill:#90EE90,stroke:#333,stroke-width:2px
      style Y fill:#FFA07A,stroke:#333,stroke-width:2px
      style AF fill:#ADD8E6,stroke:#333,stroke-width:2px
      style AI fill:#F08080,stroke:#333,stroke-width:2px
      style AM fill:#AFEEEE,stroke:#333,stroke-width:2px
      style AR fill:#20B2AA,stroke:#333,stroke-width:2px
  
  
  ```
- ```mermaid
  sequenceDiagram
      participant User
      participant AIAgent
      participant Nostr
      participant Bitcoin
      participant Lightning
      participant RGB
      participant NosDAV
      participant GitHub
      participant Logseq
      participant SolidLite
      participant LinkedJSON
      participant Omniverse
      participant USD
  
      User->>+Logseq: Define agent tasks and rewards
      Logseq->>+GitHub: Store agent configurations
      GitHub->>+AIAgent: Trigger agent update
      AIAgent->>+Nostr: Subscribe to relevant events
      Nostr->>+AIAgent: Deliver relevant events
      AIAgent->>+Logseq: Retrieve task configurations
      Logseq->>+AIAgent: Provide task configurations
      AIAgent->>+NosDAV: Retrieve required data
      NosDAV->>+AIAgent: Provide requested data
      AIAgent->>+RGB: Request single-use seal
      RGB->>+AIAgent: Provide single-use seal
      AIAgent->>+Nostr: Perform task and publish results
      Nostr->>+User: Deliver task results
      User->>+Lightning: Send payment for task completion
      Lightning->>+Bitcoin: Settle payment transaction
      Bitcoin->>+AIAgent: Confirm payment receipt
      AIAgent->>+Nostr: Publish payment confirmation
      Nostr->>+User: Deliver payment confirmation
      User->>+SolidLite: Interact with decentralized application
      SolidLite->>+LinkedJSON: Retrieve structured data
      LinkedJSON->>+SolidLite: Provide structured data
      SolidLite->>+Nostr: Publish user actions
      Nostr->>+AIAgent: Deliver user actions
      AIAgent->>+Omniverse: Retrieve virtual environment data
      Omniverse->>+AIAgent: Provide virtual environment data
      AIAgent->>+USD: Manipulate 3D assets
      USD->>+AIAgent: Provide updated 3D assets
      AIAgent->>+Omniverse: Update virtual environment
      Omniverse->>+Nostr: Publish virtual environment updates
      Nostr->>+User: Deliver virtual environment updates
      User->>+Nostr: Publish feedback and interactions
      Nostr->>+AIAgent: Deliver user feedback and interactions
      AIAgent->>+NosDAV: Store interaction data
      NosDAV->>+AIAgent: Confirm data storage
      AIAgent->>+Nostr: Publish interaction confirmation
      Nostr->>+User: Deliver interaction confirmation
  ```
- # Conclusion
	- The Agentic Mycelia presents a vision for a decentralized and interconnected metaverse ecosystem.
		- It emphasizes:
			- Interoperability
			- Adaptable governance
			- User empowerment
			- Control of personal data
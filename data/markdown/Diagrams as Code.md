public:: true

- #Public page
	- automatically published
- ## Kroki
	- [Kroki!](https://kroki.io/) provides a unified API with support for BlockDiag (BlockDiag, SeqDiag, ActDiag, NwDiag, PacketDiag, RackDiag), BPMN, Bytefield, C4 (with PlantUML), D2, DBML, Ditaa, Erd, Excalidraw, GraphViz, Mermaid, Nomnoml, Pikchr, PlantUML, Structurizr, SvgBob, Symbolator, TikZ, UMLet, Vega, Vega-Lite, WaveDrom, WireViz...
- ## Handy GPT
	- There's a GPT to assist you in making diagrams  https://chat.openai.com/g/g-5QhhdsfDj-diagrams-show-me/
- ## Local render setup
	- ```~/kroki$ docker compose up -d```
	  - ``` docker-compose.yaml version: '3'
	  
	  services:
	    kroki:
	      image: yuzutech/kroki
	      environment:
	        - KROKI_BLOCKDIAG_HOST=blockdiag
	        - KROKI_MERMAID_HOST=mermaid
	        - KROKI_BPMN_HOST=bpmn
	      ports:
	        - "8000:8000"
	  
	    blockdiag:
	      image: yuzutech/kroki-blockdiag
	  
	    mermaid:
	      image: yuzutech/kroki-mermaid
	  
	    bpmn:
	      image: yuzutech/kroki-bpmn
	  
	  
	  
	  tmux new -s kroki docker compose up -d 
	  ```
- ## Examples rendered in Logseq
	- Gemini and Claude made a diagram of this whole research corpus using  https://dreampuf.github.io/GraphvizOnline/
	  id:: 66314b8e-513e-45a6-80e7-493933e46e9e
		- ```graphviz
		  digraph G {
		    graph [rankdir=LR, overlap=false, splines=true];
		    node [shape=box, style=filled, fontsize=10, fontcolor=darkslategray];
		    edge [color=darkslategray, penwidth=1];
		  
		    // Main Themes
		    // The main themes represent the key areas of focus in the diagram. The skyblue color provides a soft and pleasant visual appearance.
		    Decentralization [fillcolor=skyblue, label="Decentralization &\nOpenness"];
		    AI [fillcolor=skyblue, label="Artificial\nIntelligence"];
		    XR [fillcolor=skyblue, label="Extended\nReality (XR)"];
		    DigitalSociety [fillcolor=skyblue, label="Digital\nSociety"];
		    GenAI [fillcolor=skyblue, label="Generative\nAI"];
		    DigitalObjects [fillcolor=skyblue, label="Digital\nObjects & NFTs"];
		    Metaverse [fillcolor=skyblue, label="Metaverse &\nVirtual Worlds"];
		    Blockchain [fillcolor=skyblue, label="Blockchain &\nDistributed Ledger\nTechnology (DLT)"];
		  
		    // Decentralized Technologies
		    // These nodes represent various decentralized technologies. The mediumseagreen color provides good readability against the dark font color.
		    Bitcoin [fillcolor=mediumseagreen, label="Bitcoin (BTC)"];
		    Nostr [fillcolor=mediumseagreen, label="Nostr Protocol"];
		    Solid [fillcolor=mediumseagreen, label="Solid Project"];
		    DecentralisedStorage [fillcolor=mediumseagreen, label="InterPlanetary File System (DecentralisedStorage)"];
		    CashuFedimint [fillcolor=mediumseagreen, label="Cashu & Fedimint"];
		    Lightning [fillcolor=mediumseagreen, label="Lightning Network"];
		    RGB [fillcolor=mediumseagreen, label="RGB Protocol"];
		    DecentralizedInternet [fillcolor=mediumseagreen, label="Decentralized Internet"];
		  
		    // AI Technologies
		    // These nodes represent various AI technologies. The moccasin color provides a warm and distinct appearance.
		    LLMs [fillcolor=moccasin, label="Large Language\nModels (LLMs)"];
		    StableDiffusion [fillcolor=moccasin, label="Stable Diffusion"];
		    Transformers [fillcolor=moccasin, label="Transformers"];
		    GANs [fillcolor=moccasin, label="Generative Adversarial\nNetworks (GANs)"];
		    Mamba [fillcolor=moccasin, label="Mamba Architecture"];
		    PromptEngineering [fillcolor=moccasin, label="Prompt Engineering"];
		    RLHF [fillcolor=moccasin, label="Reinforcement Learning\nfrom Human Feedback"];
		    DPO [fillcolor=moccasin, label="Direct Preference\nOptimization"];
		  
		    // XR Technologies
		    // These nodes represent various XR technologies. The lightpink color provides a visually appealing and distinguishable appearance.
		    AR [fillcolor=lightpink, label="Augmented\nReality (AR)"];
		    VR [fillcolor=lightpink, label="Virtual\nReality (VR)"];
		    Omniverse [fillcolor=lightpink, label="NVIDIA Omniverse"];
		    Vircadia [fillcolor=lightpink, label="Vircadia"];
		    USD [fillcolor=lightpink, label="Universal Scene\nDescription (USD)"];
		    NeRFs [fillcolor=lightpink, label="Neural Radiance\nFields (NeRFs)"];
		    SpatialComputing [fillcolor=lightpink, label="Spatial Computing"];
		    AvatarGeneration [fillcolor=lightpink, label="Avatar Generation"];
		  
		    // Intellectual Work Packages
		    // These nodes represent different intellectual work packages. The thistle color provides a subtle and distinguishable appearance.
		    VP [fillcolor=thistle, label="Virtual Production"];
		    BioMed [fillcolor=thistle, label="Biomedical Applications"];
		    Edu [fillcolor=thistle, label="AI in Education"];
		    Creative [fillcolor=thistle, label="Creative Industries"];
		    DigitalEconomy [fillcolor=thistle, label="Digital Economy"];
		  
		    // Agentic AI Actors
		    // This node represents agentic AI actors. The lightsalmon color provides a vibrant and noticeable appearance.
		    Agents [fillcolor=lightsalmon, label="Agentic AI\nActors"];
		  
		    // Connections - Decentralized Tech
		    // These edges represent the relationships between decentralized technologies and other concepts.
		    Decentralization -> Bitcoin;
		    Decentralization -> Nostr;
		    Decentralization -> Solid;
		    Decentralization -> DecentralisedStorage;
		    Decentralization -> CashuFedimint;
		    Decentralization -> Lightning;
		    Decentralization -> RGB;
		    Bitcoin -> DigitalSociety [label="Value & Payments"];
		    Bitcoin -> Metaverse [label="Economic\nTransactions"];
		    Nostr -> DigitalSociety [label="Identity &\nCommunication"];
		    Nostr -> Metaverse [label="Social Interaction"];
		    Nostr -> DigitalObjects [label="Provenance & Ownership"];
		    Solid -> DigitalSociety [label="Data Ownership"];
		    DecentralisedStorage -> DigitalSociety [label="Decentralized Storage"];
		    DecentralisedStorage -> Metaverse [label="World Data"];
		    CashuFedimint -> DigitalSociety [label="Community Banking"];
		    Lightning -> Bitcoin [label="Scalability & Speed"];
		    Lightning -> DigitalSociety [label="Micropayments"];
		    Lightning -> Metaverse [label="Microtransactions"];
		    RGB -> DigitalObjects [label="Smart Contracts\nfor Ownership"];
		  
		    // Connections - AI Tech
		    // These edges represent the relationships between AI technologies and other concepts.
		    AI -> LLMs;
		    AI -> StableDiffusion;
		    AI -> Transformers;
		    AI -> GANs;
		    AI -> Mamba;
		    AI -> PromptEngineering;
		    AI -> RLHF;
		    AI -> DPO;
		    LLMs -> GenAI;
		    StableDiffusion -> DigitalObjects [label="Image Generation"];
		    Transformers -> LLMs [label="Architecture"];
		    GANs -> DigitalObjects [label="3D Creation"];
		    Mamba -> LLMs [label="Efficient\nSequence Modeling"];
		    PromptEngineering -> LLMs [label="Control & Fine-tuning"];
		    RLHF -> LLMs [label="Alignment\nwith Human Values"];
		    DPO -> LLMs [label="Preference Learning"];
		  
		    // Connections - XR Tech
		    // These edges represent the relationships between XR technologies and other concepts.
		    XR -> AR;
		    XR -> VR;
		    XR -> Omniverse;
		    XR -> Vircadia;
		    XR -> USD;
		    XR -> NeRFs;
		    XR -> SpatialComputing;
		    XR -> AvatarGeneration;
		    AR -> Metaverse;
		    VR -> Metaverse;
		    Omniverse -> Metaverse [label="Collaboration\n& Simulation"];
		    Omniverse -> VP [label="Virtual Production\nWorkflows"];
		    Vircadia -> Metaverse [label="Open-Source\nPlatform"];
		    USD -> Omniverse [label="Scene Description"];
		    USD -> DigitalObjects [label="3D Interoperability"];
		    NeRFs -> DigitalObjects [label="3D Reconstruction"];
		    NeRFs -> Metaverse [label="Realistic Environments"];
		    SpatialComputing -> Metaverse [label="Immersive\nExperiences"];
		    AvatarGeneration -> Metaverse [label="Digital Identity"];
		    AvatarGeneration -> DigitalObjects [label="Unique Avatars"];
		  
		    // Connections - Intellectual Work Packages
		    // These edges represent the relationships between intellectual work packages and other concepts.
		    GenAI -> Creative;
		    GenAI -> VP;
		    GenAI -> BioMed;
		    GenAI -> Edu;
		    DigitalObjects -> Creative;
		    DigitalObjects -> VP;
		    Metaverse -> Creative;
		    Metaverse -> VP;
		    Blockchain -> DigitalEconomy;
		    DigitalEconomy -> Metaverse;
		  
		    // Connections to Agentic AI Actors
		    // These edges represent the relationships between agentic AI actors and other concepts.
		    AI -> Agents;
		    GenAI -> Agents;
		    DigitalObjects -> Agents [label="Ownership & Interaction"];
		    Metaverse -> Agents [label="Inhabitants &\nMediators"];
		    Blockchain -> Agents [label="Economic Activity &\nTrust"];
		    VP -> Agents [label="Automated Tasks &\nCreative Assistance"];
		    BioMed -> Agents [label="Research & Analysis"];
		    Edu -> Agents [label="Personalized Tutors &\nLearning Companions"];
		    Creative -> Agents [label="Content Creation &\nCollaboration"];
		    DigitalEconomy -> Agents [label="Transactions & Value\nCreation"];
		  
		    // Tensions and Challenges
		    // These edges represent the tensions and challenges associated with different concepts. The dashed style indicates potential areas of concern or friction.
		    AI -> Politics [label="Control & Regulation", style=dashed];
		    AI -> Ethics [label="Bias & Safety", style=dashed];
		    DigitalSociety -> Privacy [label="Surveillance &\nData Exploitation", style=dashed];
		    Metaverse -> Accessibility [label="Hardware & Cost", style=dashed];
		    VP -> Economics [label="Cost & Adoption", style=dashed];
		    BioMed -> Ethics [label="Data Privacy\n& Bias", style=dashed];
		    Edu -> Adoption [label="Institutional\nChallenges", style=dashed];
		    Creative -> IP [label="Copyright &\nOwnership", style=dashed];
		    DigitalEconomy -> Regulation [label="Global\nCoordination", style=dashed];
		  
		    // Legend
		    // The legend provides a visual guide to the different categories of nodes in the diagram.
		    Legend [shape=none, margin=0, label=<
		      <table border="0" cellborder="1" cellspacing="0">
		        <tr><td bgcolor="skyblue">Main Themes</td></tr>
		        <tr><td bgcolor="mediumseagreen">Decentralized Technologies</td></tr>
		        <tr><td bgcolor="moccasin">AI Technologies</td></tr>
		        <tr><td bgcolor="lightpink">XR Technologies</td></tr>
		        <tr><td bgcolor="thistle">Intellectual Work Packages</td></tr>
		        <tr><td bgcolor="lightsalmon">Agentic AI Actors</td></tr>
		      </table>
		    >];
		  }
		  ```
- Create a shoot schedule for a medium sized ICVFX LED wall commercial shoot with 2 days prelight and 5 days shooting. Try to include the maximum camera optionality and shot diversity by assuming a tight and professional crew. use mermaid diagram gantt chart code
  id:: 65ce7e9e-e891-49c2-af30-5f27904d2aca
- ```mermaid
  gantt
      title Shoot Schedule
      dateFormat  YYYY-MM-DD
      axisFormat  %d-%m
  
      section Prelight
      Preparing the LED Wall: done, des1, 2024-01-01, 2d
  
      section Shooting Day 1
      Shot 1: after des1, 2h
      Shot 2: after des1, 1h
      Shot 3: after des1, 3h
      Shot 4: after des1, 2h
      Shot 5: after des1, 4h
  
      section Shooting Day 2
      Shot 6: 1h
      Shot 7: 2h
      Shot 8: 4h
      Shot 9: 3h
  
      section Shooting Day 3
      Shot 10: 2h
      Shot 11: 3h
      Shot 12: 4h
      Shot 13: 1h
  
      section Shooting Day 4
      Shot 14: 2h
      Shot 15: 3h
      Shot 16: 1h
      Shot 17: 4h
  
      section Shooting Day 5
      Shot 18: 2h
      Shot 19: 4h
      Shot 20: 1h
      Shot 21: 5h
  ```
-
- Mermaid
	- Example
		- write mermaid code for a plan to produce client server PWA that uses stable diffusion for image generation
	- Christianity Historical Timeline
		- ```mermaid
		  gantt
		  title Christianity Historical Timeline
		  dateFormat YYYY-MM-DD
		  axisFormat %Y
		  %% Temporal Break for Early Years
		  section Initial Christianity
		  Jesus' Ministry :done, 0030-01-01, 0033-04-01
		  -- Temporal Break -- :active, 0033-04-02, 1054-07-15
		  Early Church :done, 1054-07-16, 1054-07-16
		  %% Eastern Churches
		  section Eastern Churches
		  Great Schism
		  - Eastern Orthodoxy :eastern, 1054-07-16, 2023-12-24
		  Oriental Orthodoxy               :oriental, 451-01-01, 2023-12-24
		  section Western Church
		  Early Roman Church        :rome, 1054-07-16, 1054-07-16
		  Western Schism            :schism, 1378-01-01, 1417-10-18
		  Catholicism               :catholic, 1054-07-16, 2023-12-24
		  Protestant Reformation :protestant, 1517-10-31, 2023-12-24
		  section Protestant Branches
		  Lutheranism     :lutheran, 1517-10-31, 2023-12-24
		  Calvinism       :calvinist, 1536-01-01, 2023-12-24
		  Anglicanism     :anglican, 1534-01-01, 2023-12-24
		  Baptist         :baptist, 1609-01-01, 2023-12-24
		  Methodism       :methodist, 1738-01-01, 2023-12-24
		  Pentecostalism  :pentecostal, 1906-01-01, 2023-12-24
		  section Other Developments
		  Counter-Reformation :counter, 1545-01-01, 1648-01-01
		  Modernist Controversy :modernist, 1800-01-01, 2023-12-24
		  Ecumenism Movement :ecumenism, 1900-01-01, 2023-12-24
		  section American Sect Churches
		  Adventism                :adventist, 1863-01-01, 2023-12-24
		  Latter-Day Saint Movement:Mormonism, 1830-04-06, 2023-12-24
		  Christian Science        :science, 1879-01-01, 2023-12-24
		  Jehovah's Witnesses      :jehovah, 1870-01-01, 2023-12-24
		  Pentecostalism           :pentecostal, 1901-01-01, 2023-12-24
		  Evangelicalism           :evangelical, 1730-01-01, 2023-12-24
		  Fundamentalism           :fundamental, 1910-01-01, 2023-12-24
		  Black Church Traditions  :blackchurch, 1780-01-01, 2023-12-24
		  Non-denominational       :nondenom, 1960-01-01, 2023-12-24
		  ```
		- Sequence Diagram
		- ```mermaid
		   sequenceDiagram
		       participant User as User
		       participant Script as Script
		       participant TxtFile as Trait .txt File
		       participant NameFiles as Name Files (forenames, nicknames, surnames)
		       participant JsonFile as Corresponding .json File
		       participant API as OpenAI API
		   
		       User->>Script: Run script with directory
		       Script->>NameFiles: Load names from forenames.txt, nicknames.txt, surnames.txt
		       NameFiles->>Script: Return names
		       loop For each .txt file in directory
		           Script->>TxtFile: Read .txt file content
		           TxtFile->>Script: Return content
		           Script->>Script: Hash file content
		           Script->>Script: Generate name using hash
		           Script->>API: Send prompt with name
		           API->>Script: Return story
		           alt If API limit reached or no response
		               API->>Script: Return empty response
		               Script->>User: Log API limit error
		           else If response received
		               API->>Script: Return narrative response
		               Script->>JsonFile: Read .json file
		               JsonFile->>Script: Return JSON data
		               Script->>Script: Insert story into JSON description
		               Script->>JsonFile: Write updated JSON
		           end
		       end
		   
		   ```
-
	- ```mermaid graph TB
	    A["Web Browser"] -- "HTTP API Request" --> B["Load Balancer"]
	    B -- "HTTP Request" --> C["Crossover"]
	    style C fill:#99cc99
	    C -- "Talks to" --> D["Redis"]
	    C -- "Talks to" --> E["MySQL"]
	    C -- "Downstream API Request" --> F["Multiplex"]
	    F -- "Returns Job ID" --> C
	    C -- "Long Poll API Request" --> G["Evaluator"]
	    G -- "API Call" --> F
	    G -- "API Call" --> H["Result-Fetcher"]
	    H -- "Downloads Results" --> I["S3 or GCP Cloud Buckets"]
	    I -- "Results Stream" --> G
	    G -- "Results Stream" --> C
	    C -- "API Response" --> B
	    B -- "API Response" --> A
	  
	  ```
	-
- [Graphviz](https://graphviz.org/)
- {{renderer code_diagram,graphviz}}
	- ```graphviz digraph G {	
	      // Global settings
	      rankdir=LR;
	      node [shape=rectangle, style=filled, color=lightblue];
	      edge [color=black, penwidth=1.5];
	  
	      // Define nodes
	      node1 [label="Start", shape=ellipse, color=lightgreen];
	      node2 [label="Process 1"];
	      node3 [label="Decision", shape=diamond, fillcolor=yellow];
	      node4 [label="Process 2"];
	      node5 [label="End", shape=ellipse, color=red];
	  
	      // Define edges
	      node1 -> node2;
	      node2 -> node3 [label="condition"];
	      node3 -> node4 [label="yes"];
	      node3 -> node5 [label="no", style=dashed];
	  
	      // Subgraph (cluster)
	      subgraph cluster_0 {
	          label="Phase 1";
	          color=grey;
	          node2 -> node3;
	          node3 -> node4;
	      }
	  
	      // Subgraph (cluster)
	      subgraph cluster_1 {
	          label="Phase 2";
	          color=grey;
	          node4 -> node5;
	      }
	  
	      // Additional settings for layout
	      { rank=same; node2; node4; }
	      { rank=same; node3; node5; } }
	  
	  ```
- [PlantUML](https://plantuml.com/)
  {{renderer code_diagram,plantuml}}
	- ```plantuml @startwbs skinparam monochrome true * Business Process Modelling WBS ** Launch the project *** Complete Stakeholder Research *** Initial Implementation Plan ** Design phase *** Model of AsIs Processes Completed **** Model of AsIs Processes Completed1 **** Model of AsIs Processes Completed2 *** Measure AsIs performance metrics *** Identify Quick Wins ** Complete innovate phase @endwbs ```
	- {{renderer code_diagram,plantuml}} ```@startuml
	  
	  ' Define an interface interface Vehicle {
	      +int getWheelCount()
	      +String getModelName() }
	  
	  ' Define classes implementing the interface class Car implements Vehicle {
	      -String modelName
	      -int wheelCount
	      +Car(String modelName)
	      +int getWheelCount()
	      +String getModelName() }
	  
	  class Bike implements Vehicle {
	      -String modelName
	      -int wheelCount
	      +Bike(String modelName)
	      +int getWheelCount()
	      +String getModelName() }
	  
	  ' Define a generic class class Garage<T> {
	      -List<T> vehicles
	      +void parkVehicle(T vehicle)
	      +T retrieveVehicle() }
	  
	  ' Add notes to the diagram note right of Car : Cars usually have 4 wheels note left of Bike : Bikes usually have 2 wheels
	  
	  ' Define relationships Garage -up-|> Vehicle : contains >> Vehicle <|.. Car : implements Vehicle <|.. Bike : implements
	  
	  ' Add a title title Complex Class Diagram Example
	  
	  @enduml ```
- [BPMN](https://en.wikipedia.org/wiki/Business_Process_Model_and_Notation)
- [Bytefield](https://texdoc.org/serve/bytefield.pdf/0)
- [BlockDiag](http://blockdiag.com/en/blockdiag/index.html)
- [SeqDiag](http://blockdiag.com/en/seqdiag/index.html)
- [ActDiag](http://blockdiag.com/en/actdiag/index.html)
- [NwDiag](http://blockdiag.com/en/nwdiag/index.html)
- [Ditaa](http://ditaa.sourceforge.net/)
- [Nomnoml](https://www.nomnoml.com/)
- [Erd](https://hackage.haskell.org/package/erd)
- [Pikchr](https://pikchr.org/)
- [Structurizr](https://structurizr.com/)
- [Vega](https://vega.github.io/)
- [Vega-Lite](https://vega.github.io/vega-lite/)
- [WaveDrom](https://wavedrom.com/)
- [D2](https://d2lang.com/)
- [UMlet](https://www.umlet.com/)
- [SvgBob](https://ivanceras.github.io/svgbob-editor/)
- [PGF/TikZ](https://tikz.dev/)
-
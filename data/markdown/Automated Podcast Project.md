public:: true

- [development chat work in progress.txt](../assets/development_chat_work_in_progress_1719174307794_0.txt)
- # LogCaster
	- A podcast creation engine underpinned by a logseq knowledge graph and multi-agent framework.
- # Automated Daily News Podcast Generation
	- This project aims to develop a sophisticated toolchain using FastAPI and OpenWebUI to automate the creation of a daily news podcast. The goal is to seamlessly integrate various modules and pipelines to process breaking news items, focusing on selecting interesting and unusual combinations of topics. The system will generate structured content through a multi-agent approach, mediated by a single API interface, ultimately producing a compelling podcast featuring two interlocutors, Bob and Sue.
- ## Workflow Overview
	- **Parse Logseq**: A Logseq executed Python block parses the current Logseq markdown page to find the IP address of the OpenWebUI server specified here:
		- ```text
		  192.168.0.51
		  ```
	- **Request the Model List**: Use the OpenWebUI API to request the model list from the IP address specified above and the access token provided below:
		- ```text
		  Authorization: Bearer sk-6f3f39171ba247b4a66093287305fabc
		  ```
	- Python code to replace the curl command:
		- ```python
		  import requests
		  import json
		  
		  # Set up the request headers and URL
		  headers = {
		      "accept": "application/json",
		      "Authorization": "Bearer sk-6f3f39171ba247b4a66093287305fabc"
		  }
		  url = "http://192.168.0.51:3000/api/v1/models/?id="
		  
		  # Make the GET request
		  response = requests.get(url, headers=headers)
		  
		  # Parse the response
		  if response.status_code == 200:
		      models = response.json()
		      model_names = [model['name'] for model in models]
		       # Print the model names (we assume logseq will show them)
		      print("\n".join(model_names)) 
		  else:
		      print("Failed to fetch models:", response.status_code, response.text)
		  ```
			- {{evalparent}}
	- **List is returned**: and should be written directly into this Logseq page in the block labelled below the code:
		- ## Model List
			- ```text
			  Models returned from the OpenWebUI API should be listed here, one model name per line.
			  ```
	- **Select Agent Models**: The user can copy and paste model names into the agent slots here.
		- ## Nominated Agents
			- Web Search
				- ```text
				  Perplexity
				  ```
			- Rating the news
				- ```text
				  Mistral 8B
				  ```
- **Loading Topics**: Search the logseq knowledge graph for the tagged public pages. Create a list of any pages above 100kb in size. Note Python will load the first time this block is evaluated.
	- ```python
	  import js
	  
	  def list_public_pages():
	      try:
	          pages = js.logseq.api.get_all_pages()
	          public_pages = []
	          for page in pages:
	              page_name = page.originalName
	              # Get the first block of the page which typically contains metadata
	              content = js.logseq.api.get_page_blocks_tree(page_name)
	              if content and "public:: true" in content[0].content:
	                  public_pages.append(page_name)
	          return public_pages
	      except Exception as e:
	          js.logseq.api.show_msg(f"Error in list_public_pages: {e}", {'timeout': 5000})
	          return []
	  
	  def main():
	      public_pages = list_public_pages()
	      if public_pages:
	          result = ", ".join(public_pages)
	      else:
	          result = "No public pages found."
	      return result
	  
	  # Execute the main function
	  main()
	  ```
		- {{evalparent}}
		-
-
	- ## Topics with more than 100kb
		- Topic 1
		- Topic 2
		- This list should be replaced inside this file by the code.
	- **User Elects Topics**: Present the public list to the user as line-separated topics. The user selects their topics of choice by deleting unimportant lines. This selection remains stored in the Logseq page.
	- **Randomly Select Topics**: Next, executable Logseq code selects 2 or 3 topics at random from the list of elected topics. It updates the Logseq block below:
		- ### Trying these topics
			- Topic 1
			- Topic 2
			- Topic 3
	- **Searching for News**: The same code block calls the Perplexity pipeline using the OpenWebUI unified API to search for breaking news items that intersect with the randomly loaded topics.
	- **Ranking Importance**: Evaluate the returned news items using a locally hosted Mixtral 8B LLM, called from the list of available models. Request a score on likely impact and public interest on a scale from 1 to 10.
	- **Branching Factor**: If the news item scores above 8, pass the Perplexity-generated content to GPT-4. Otherwise, select another 2 or 3 public-tagged topics and repeat. The process can be attempted a maximum of 10 times. If 10 attempts do not surface a news item of worth, alert the user in the Logseq output.
	- **Extract Web Links**: Create a simple list of raw URLs found in the Perplexity response and remove duplicates.
	- **Detailed Information Scraping**: Process the identified links using the web scraper module from the models list in OpenWebUI to fetch in-depth summaries and additional content from the linked pages.
	- **Or Progress News Item**: Use the GPT4oV model from the OpenWebUI models list to process the raw Perplexity response and all ancillary summaries gathered from the web scraper. Return a detailed and highly technical description of the news item in JSON format, with web URLs and their summaries carefully segmented and linked as a knowledge graph.
	- **Enhancing with RAG**: Send the text elements of the GPT4-generated JSON to the RAGflow module in the models list, requesting that the knowledge be modified to include any opinions and ideas from the RAG corpus that intersect with the news item. Save the response to a Logseq block titled # Story Plus RAG.
	- **Generating Podcast Script**: Use Claude 3.5 to create a podcast script based on few-shot examples from the Logseq page labelled [[example podcast dialogue]]. Send the RAGflow-enhanced news story and the JSON-structured web links and summaries. The script should alternate dialogue between Bob and Sue, incorporating brief mentions of the web sources available in the episode notes. The returned JSON from Claude 3.5 should have field identifiers for Bob and Sue.
	- **Splitting Script**: Divide the podcast script into parts for Bob and Sue, creating two new JSON files with sequence numbers for the conversation.
	- **Text to Voice Conversion**: Pass both scripts to text-to-voice engines to generate the audio for Bob and Sue's lines respectively. Insist that the audio be returned with sequence numbers.
	- **Inline Python Code Block Creates WAV File**: Placeholder for now.
	- **Synchronization with Metahuman**: Synchronize the generated audio tracks with Metahuman talking heads in Unreal Engine to create a lifelike rendering of the podcast episode.
- ## Workflow Diagram
  
  ```mermaid
  graph TD
    A[Parse Logseq] --> B[Request Model List]
    B --> C[Model List Returned]
    C --> D[Select Agent Models]
    D --> E[Loading Topics]
    E --> F[User Elects Topics]
    F --> G[Randomly Select Topics]
    G --> H[Searching for News]
    H --> I[Ranking Importance]
    I -->|Score > 8| K[Process with GPT-4]
    I -->|Score <= 8| J[Select New Topics]
    K --> L[Extract Web Links]
    L --> M[Detailed Information Scraping]
    M --> N[Enhance with RAGflow]
    N --> O[Generating Podcast Script]
    O --> P[Split Script]
    P --> Q[Text to Voice Conversion]
    Q --> R[Create WAV File]
    R --> S[Synchronize with Metahuman]
  ```
- ## Sequence Diagram
  ```mermaid
  sequenceDiagram
    participant User
    participant Logseq
    participant API
    participant Models
    participant Perplexity
    participant Mixtral
    participant GPT4
    participant Scraper
    participant Claude
    participant TTS
    User->>Logseq: Trigger Python Script
    Logseq->>API: Request Model List
    API->>Logseq: Return Model List
    Logseq->>User: Display Model List
    User->>Logseq: Select Agent Models
    Logseq->>User: Display Public Pages
    User->>Logseq: Elect Topics
    Logseq->>Logseq: Randomly Select Topics
    Logseq->>Perplexity: Search for News
    Perplexity->>Logseq: Return News Items
    Logseq->>Mixtral: Rank News Items
    Mixtral->>Logseq: Return Scores
    alt Score > 8
      Logseq->>GPT4: Process with GPT-4
      GPT4->>Logseq: Return Detailed Info
      Logseq->>API: Extract Web Links
      API->>Logseq: Return Links
      Logseq->>Scraper: Scrape Details
      Scraper->>Logseq: Return Summaries
      Logseq->>RAG: Enhance with RAG
      RAG->>Logseq: Return Enhanced Story
      Logseq->>Claude: Generate Podcast Script
      Claude->>Logseq: Return Podcast Script
      Logseq->>TTS: Convert Text to Voice
      TTS->>Logseq: Return Audio
      Logseq->>Unreal: Sync with Metahuman
    else Score <= 8
      Logseq->>Logseq: Select New Topics
    end
    
  ```
- ## Implementation Details
	- The toolchain will be orchestrated by a Python script that interacts with the filesystem and calls the necessary APIs. The script will be modular, with each task encapsulated in its own function, including robust logging, configuration management, state management, unit tests, and documentation.
	- ### Main Functions
		- 1. `load_topics()`: Reads a list of topics from a file.
		  2. `search_news_items(topics)`: Uses Perplexity to search for news items related to the given topics.
		  3. `rank_news_items(items)`: Uses Mixtral 8B LLM to rank the news items, returning a list of items with a score from 1 to 10.
		  4. `process_high_score_items(items)`: Filters items with scores above 8, uses GPT-4 to restructure the items into technical essays, and isolates web links.
		  5. `scrape_details(links)`: Uses a web scraper module to fetch detailed summaries from the links.
		  6. `create_podcast_script(story, summaries)`: Uses Claude 3.5 and RAGflow corpus to create a podcast script.
		  7. `split_script(script)`: Splits the script between two interlocutors: Bob and Sue.
		  8. `text_to_voice(lines, person)`: Uses text-to-voice engines to convert lines into audio for Bob and Sue.
		  9. `sync_with_metahuman(bob_audio, sue_audio)`: Syncs the audio with Metahuman talking heads over a network connection to Unreal Engine.
- ## Next Steps
	- 1. Implement the Python script with the outlined functions and best practices.
	  2. Set up the necessary APIs and modules (FastAPI, OpenWebUI, Perplexity, Mixtral 8B LLM, GPT-4, Claude 3.5, RAGflow, web scraper, text-to-voice engines, Metahuman, Unreal Engine).
	  3. Test and refine the toolchain, ensuring smooth integration and reliable performance.
	  4. Document the setup, usage, and maintenance of the toolchain for future reference and collaboration.
public:: true

- [unclecode/crawl4ai: 🔥🕷️ Crawl4AI: Open-source LLM Friendly Web Crawler & Scrapper (github.com)](https://github.com/unclecode/crawl4ai) [[AI Scrapers]]
- https://github.com/yigitkonur/swift-ocr-llm-powered-pdf-to-markdown [[Knowledge Graphing and RAG]] [[AI Scrapers]]
- https://github.com/unclecode/crawl4ai [[AI Scrapers]]
- https://therecord.media/federal-civil-rights-watchdog-facial-recognition-technology-report [[Privacy, Trust and Safety]] 
  https://mmsearch.github.io/ [[AI Scrapers]] [[perplexity]]
- [2408.08435v1.pdf (arxiv.org)](https://arxiv.org/pdf/2408.08435)[2408.08435v1.pdf (arxiv.org)](https://arxiv.org/pdf/2408.08435) [[Agents]]
  https://mmsearch.github.io/ [[AI Scrapers]] [[perplexity]]
- [mendableai/firecrawl: 🔥 Turn entire websites into LLM-ready markdown or structured data. Scrape, crawl and extract with a single API. (github.com)](https://github.com/mendableai/firecrawl) [[WebDev and Consumer Tooling]] [[AI Scrapers]]
- [NicolasBizzozzero/pattern: Web mining module for Python, with tools for scraping, natural language processing, machine learning, network analysis and visualization.](https://github.com/NicolasBizzozzero/pattern) [[AI Scrapers]] [[WebDev and Consumer Tooling]]
- [VinciGit00/Scrapegraph-ai: Python scraper based on AI (github.com)](https://github.com/VinciGit00/Scrapegraph-ai)
-
-
- [[Projects]] [[Open Webui and Pipelines]]
	- Web scraper project for OpenWebUI
	- ```mermaid
	  sequenceDiagram
	      participant User
	      participant Pipeline
	      participant OpenWebUI
	      participant AsyncOpenAI
	      participant Playwright
	      participant RedditClient
	      participant WebPage
	  
	      User->>Pipeline: Send user_message
	      Pipeline->>OpenWebUI: Get OPENAI_API_KEY, TOPICS, etc.
	      Pipeline->>AsyncOpenAI: Initialize with API key
	      Pipeline->>Playwright: setup_playwright()
	      Playwright->>Pipeline: Playwright setup complete
	      Pipeline->>RedditClient: Initialize with credentials
	  
	      Pipeline->>Pipeline: extract_blocks(user_message)
	      loop For each block
	          Pipeline->>Pipeline: should_process_block(block)
	          alt Block should be processed
	              Pipeline->>Pipeline: extract_url_from_block(block)
	              alt URL is a Reddit URL
	                  Pipeline->>RedditClient: is_reddit_url(url)
	                  RedditClient->>Pipeline: True
	                  Pipeline->>RedditClient: get_reddit_content(url)
	                  RedditClient->>Pipeline: Return Reddit content
	              else URL is not a Reddit URL
	                  Pipeline->>Playwright: scrape_url(url, random_user_agent)
	                  Playwright->>WebPage: Fetch and filter content
	                  WebPage->>Playwright: Return filtered content
	                  Playwright->>Pipeline: Return filtered content
	                  alt Scraping successful
	                      Pipeline->>Pipeline: create_prompt(link_text, url, topics, max_tokens)
	                      Pipeline->>AsyncOpenAI: Generate summary
	                      AsyncOpenAI->>Pipeline: Return summary JSON
	                      Pipeline->>Pipeline: Format summary to Logseq style
	                  else Scraping failed
	                      Pipeline->>Pipeline: Return original block
	              end
	          else Block should not be processed
	              Pipeline->>Pipeline: Return original block
	          end
	          Pipeline->>Pipeline: Add processed block to processed_blocks
	      end
	      Pipeline->>User: Return processed text
	  end
	  ```
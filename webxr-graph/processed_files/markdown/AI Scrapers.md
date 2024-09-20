public:: true

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
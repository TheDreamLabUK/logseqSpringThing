- Note Python will load the first time this block is evaluated
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
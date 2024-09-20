- ```javascript
  import js
  
  # Helper function to convert object to a readable format
  def props(title, obj):
      print("---------------------------\n")
  
      try:
          keys = obj.object_keys()
          vals = obj.object_values()
          output = title + " Objects\n"
          for ctr in range(len(keys)):
              output += f"{str(keys[ctr]): <12}  =>  {str(vals[ctr]): <40}\n"
      except:
          output = "Not an object"
      print(output)
      return output
  
  # Function to log output to a specific Logseq page
  def log_output(content, clearlog=False):
      import datetime
      output_page = "Execution Log"  # API always uses lower case page names
      if clearlog:
          logpage = js.logseq.api.get_page(output_page.lower())
          blocks = js.logseq.api.get_page_blocks_tree(logpage.uuid)
          for block in blocks:
              js.logseq.api.remove_block(block.uuid)
  
      js.logseq.api.append_block_in_page(
          output_page,
          str(datetime.datetime.now())[:19] + " => " + content)
      return "Done"
  
  # Function to list files containing the specified term
  def list_files_containing_term(search_term):
      output = ""
      files_with_term = set()
  
      # Searching for the term using Logseq's search feature (Python compatible)
      try:
          blocks = js.logseq.api.search(search_term)
          for block in blocks:
              obj_map = block.as_object_map()  # Translate returned block for Python usage
              page_id = obj_map["block/page"]["original-name"]
              files_with_term.add(page_id)
      except Exception as e:
          log_output(f"Error searching: {str(e)}", clearlog=True)
          return "Error while searching"
  
      for file in files_with_term:
          output += f"{file}\n"
  
      log_output(output, clearlog=True)
      return output
  
  # Define the search term
  search_term = "public:: true"
  
  # Execute the search and log the result
  result = list_files_containing_term(search_term)
  print(result)
  ```
	- {{evalparent}}
- ```python
  import js
  
  # Helper function to convert object to a readable format
  def props(title, obj):
      print("---------------------------\n")
  
      try:
          keys = obj.object_keys()
          vals = obj.object_values()
          output = title + " Objects\n"
          for ctr in range(len(keys)):
              output += f"{str(keys[ctr]): <12}  =>  {str(vals[ctr]): <40}\n"
      except:
          output = "Not an object"
      print(output)
      return output
  
  # Function to log output to a specific Logseq page
  def log_output(content, clearlog=False):
      import datetime
      output_page = "Execution Log"  # API always uses lower case page names
      if clearlog:
          logpage = js.logseq.api.get_page(output_page.lower())
          blocks = js.logseq.api.get_page_blocks_tree(logpage.uuid)
          for block in blocks:
              js.logseq.api.remove_block(block.uuid)
  
      js.logseq.api.append_block_in_page(
          output_page,
          str(datetime.datetime.now())[:19] + " => " + content)
      return "Done"
  
  # Function to list files containing the specified term
  def list_files_containing_term(search_term):
      output = ""
      files_with_term = set()
  
      # Searching for the term using Logseq's search feature (Python compatible)
      try:
          blocks = js.logseq.api.search(search_term)
          for block in blocks:
              obj_map = block.as_object_map()  # Translate returned block for Python usage
              page_id = obj_map["block/page"]["original-name"]
              files_with_term.add(page_id)
      except Exception as e:
          log_output(f"Error searching: {str(e)}", clearlog=True)
          return "Error while searching"
  
      for file in files_with_term:
          output += f"{file}\n"
  
      log_output(output, clearlog=True)
      return output
  
  # Define the search term
  search_term = "public:: true"
  
  # Execute the search and log the result
  result = list_files_containing_term(search_term)
  print(result)
  ```
-
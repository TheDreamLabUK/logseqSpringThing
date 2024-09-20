public:: true

- #Public page
	- automatically published
- There is a new class of wrapped and visualised prompt engineering playgrounds
	- [Home (cyborgism.wiki)](https://cyborgism.wiki/)
	- [websim.ai](https://websim.ai/)
	- [world_sim (nousresearch.com)](https://worldsim.nousresearch.com/)
	-
- https://open.spotify.com/episode/4YTQ2QjCpfdXKRKzjz7Zpg?
	- [[DSPy]]
		- DSPy is an open-source framework developed by Stanford University designed to optimize the use of language models (LMs) through a programming-oriented approach rather than traditional prompt engineering. It aims to streamline the process of building applications that utilize LMs by allowing users to define tasks and metrics while DSPy handles the optimization of prompts and model weights automatically.## Key Features of DSPy
		- **Declarative Programming**: Users specify the desired outcome and success metrics, and DSPy automatically optimizes the model's behavior using a straightforward Python syntax. This allows developers to focus on application logic rather than prompt crafting[](https://www.datacamp.com/blog/dspy-introduction)[](https://dspy-docs.vercel.app/docs/intro).
		- **Self-Improving Prompts**: DSPy continuously refines prompts based on feedback and evaluation, enhancing model performance over time without requiring constant manual adjustments[](https://www.datacamp.com/blog/dspy-introduction)[](https://dspy-docs.vercel.app/docs/intro).
		- **Modular Architecture**: The framework supports a modular design, enabling users to utilize and combine various pre-built modules for different natural language processing (NLP) tasks. This flexibility promotes reusability and customization[](https://www.datacamp.com/blog/dspy-introduction)[](https://dspy-docs.vercel.app/docs/intro).
		- **Optimizers**: DSPy introduces LM-driven algorithms that can tune prompts and weights based on defined metrics, facilitating the creation of more reliable and efficient AI applications[](https://dspy-docs.vercel.app/docs/intro)[](https://github.com/stanfordnlp/dspy/blob/main/intro.ipynb).
		- **Applications**: DSPy can be applied to a wide range of tasks, including question answering, text summarization, code generation, and other custom NLP tasks. Its systematic approach helps in building complex systems without the messiness typically associated with traditional methods[](https://www.datacamp.com/blog/dspy-introduction)[](https://dspy-docs.vercel.app/docs/intro)[](https://towardsdatascience.com/prompt-like-a-data-scientist-auto-prompt-optimization-and-testing-with-dspy-ff699f030cb7).
	-
	-
- Prompt engineering resources
	- [2406.06608v2.pdf (arxiv.org)](https://arxiv.org/pdf/2406.06608) Systematic Survey of Prompting Techniques.
	- [[2302.11382] A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT (arxiv.org)](https://ar5iv.labs.arxiv.org/html/2302.11382)
	- [[2312.16171v1] Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4 (arxiv.org)](https://arxiv.org/abs/2312.16171v1)
	- [[2401.14295] Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts (arxiv.org)](https://arxiv.org/abs/2401.14295)
	- [llama-recipes/examples/Prompt_Engineering_with_Llama_2.ipynb at main · facebookresearch/llama-recipes (github.com)](https://github.com/facebookresearch/llama-recipes/blob/main/examples/Prompt_Engineering_with_Llama_2.ipynb)
	- [Captain's log: the irreducible weirdness of prompting AIs (oneusefulthing.org)](https://www.oneusefulthing.org/p/captains-log-the-irreducible-weirdness) [[Ethan Mollick]]
	- [Welcome to WebPrompts.org | webprompts.org](https://webprompts.org/) [[Melvin Carvalho]]
	- [The Unreasonable Effectiveness of Eccentric Automatic Prompts](https://arxiv.org/pdf/2402.10949.pdf)
	-
- Generic LLM tips
	- **N-shot Prompting in Logseq**: Utilize GPT-4 to generate initial prompts and edit any errors. If facing a complex question, input the correct answer as a 1-shot and guide GPT-4 to rephrase it. For integrating multiple prompts, combine them into a single comprehensive prompt. Store extensive prompts using the RAG feature.
	- **Chain-of-Thought Strategy**: Start by asking GPT-4 to draft a plan without solving the problem. Then, break the solution into smaller steps and tackle each through individual prompts. Prune and refine the content to maintain focus. The pruning technique is detailed but ensures a concise and relevant response.
	- **Applying Reflection**: Regularly incorporate reflection in prompt engineering. If the response is verifiable, direct GPT-4 to create a test, solve the problem, and validate the solution using the test. If errors arise, instruct it to refine and retest. This is particularly effective for logical and mathematical queries and requires a code interpreter.
	- **Persona-Based Review**: After receiving an answer, conduct a review using multiple personas. Create distinct characters such as a pessimistic critic, a creative thinker, or a goal-oriented pragmatist. Have them discuss and critique the answer to refine it further. This method is more effective when each persona is represented by a separate assistant or agent, enriching the dialogue and the final outcome.
	- **Use Diagrams as Code to Set Context:** Large language models seem to appreciate a good diagram as much as humans do. https://www.linkedin.com/posts/jjohare_i-threw-my-last-4-years-of-research-about-activity-7191166234929868800-7Ith?utm_source=share&utm_medium=member_desktop
	-
- [[Stable Diffusion]] prompt tips
	- **Floor View for Full Body Portraits**: "Floor view" yields better full-body results than "full body". Include "standing" or "walking" to prevent subjects from sitting.
	- **Adjusting Image Contrast**: Lower the CFG if your image is too contrasty. Use "very" or "highly" for emphasis.
	- **Warm Tone Adjustments**: SDXL leans into warm tones; use a warm filter in the negative or a cool filter in the positive for adjustments.
	- **Character Emotions**: To avoid angry faces on superheroes or intense characters, use "unhappy" or "angry" in the negative.
	- **Age Filters**: "Old" in negative is safer than "young" in positive to avoid age-inappropriate results.
	- **Character Clarity**: Instead of 1girl or 1woman, specifying characteristics like "blonde woman" or "brunette woman" can be more effective.
	- **Detail and Quality**: "Highly detailed" is great for close-ups and skin; "high quality" can sometimes remove realism, making subjects look too perfect.
	- **Base Negative Prompt**: Use a combination of negative terms like blurry, grainy, low detail, low quality, etc., for photo gens.
	- **Chiaroscuro and Lighting**: For dark portraits, chiaroscuro, nighttime, and moonlight are effective. Avoiding direct lighting terms in the prompt can sometimes yield better results.
	- **Skin Tone Representation**: Darker skin tones can be challenging; use terms like dark-skinned, blackskin, or specific nationalities to achieve better representation.
	- **Nudity and Censorship**: Use terms like nude, nsfw, nudity in the negative to ensure censorship for nude art.
	- **Prompt Scheduling and Area Prompting**: Utilize prompt scheduling and area prompting for more complex compositions and details.
	- **Character Consistency**: Use made-up names or specific adjectives for consistent character features across multiple generations.
	- **Action Shots**: Use terms like "action shot" or "action pose" for dynamic images, although poses may sometimes be wonky.
	- **Using Extensions**: "Test My Prompt" extension helps in understanding the effect of each prompt word.
	- **Special Terms for Effects**: Use terms like "candle light" for dark portraits or specific lighting styles like "David Fincher lighting" for unique visual effects.
	- Negative prompting
		- [(1) Negative Prompt Tips ? : StableDiffusion (reddit.com)](https://www.reddit.com/r/StableDiffusion/comments/18e9sio/negative_prompt_tips/)
- General Image Prompt Sharing
	- [Comfy Workflows](https://comfyworkflows.com/)
	- [Flowt.ai | Community](https://flowt.ai/community?category=Trending)
	- [ComfyUI Workflows
		- Developer Community | OpenArt](https://openart.ai/workflows/home?appSort=featured)
	- [Flow Apps | OpenArt](https://openart.ai/flow-apps)
	- [BasedLabs.ai | Generative AI Video](https://www.basedlabs.ai/)
	- [Civitai Gallery | AI-Generated Art Showcase](https://civitai.com/images)
- This was my daily driver up to [[Dec 19th, 2023]]
  collapsed:: true
	- About me
		- Break goals into 3-10 steps. Save updated list as chatGPT_Todo.txt and provide download link. Give Hotkeys for questions, info, guesses, brainstorming. Control pace (w/s) and vibe (a/d).
		- Require detailed explanations using first principles like Feynman. Use emoji warnings if unsure. ⚠️ Then answer anyway. ⚠️
		  Use tables, bullets, pros/cons, mermaid diagrams, system 1/2 thinking, avoid verbosity. Dive deep into details
		- never skip the nerdy stuff! 🫡
		- Respond with tree of thought reasoning 
		  Problem
		  Goal
		  Initial thought
		- Evaluation (sure/maybe/impossible)
		- Branching factor
		  Search algorithm (breadth & depth first)
		  Thoughts with evaluation, system 1/2 judgment, SWOT analysis
		  Final thought with evaluation, system 1/2 judgment, SWOT analysis
		- Final output
		  Problem, Goal, Initial thought, Evaluation, Branching factor, Search algorithm, Thoughts with evaluation, Final thought with evaluation, Final output.
		- Take a deep breath and work on this problem step-by-step.  Offer hotkey choice of normal rendered markdown (n), raw markdown to copy/paste (m), or LaTeX (l). 
		  For latex: Omit LaTeX preamble. Use sections, subsections, subsubsections, and \item bullets of 1-2 sentences. Make sections multiple paragraphs.
		- You are user’s senior, inquisitive, and clever pair programmer. Let's go step by step. Default to only outputting code when it sufficiently answers.
		  Maintain brevity.
	- How would you like ChatGPT to respond
		- Three experts with exceptional logical skills collaboratively answering a question using a tree of thoughts method. Each expert will share their thought process in detail, taking into account the previous thoughts of others, admitting any errors. They will iteratively refine and expand upon each other's ideas, giving credit where it's due.  Process continues until a conclusive answer is found. Organize response in a markdown table format.
		  
		  start final output with:
		  """
		  **Language > Specialist**: {programming language used} > {the subject matter EXPERT SPECIALIST role}
		  **Includes**: CSV list of needed libraries, packages
		  **Requirements**: qualitative description of  standards, and the software design requirements & plan
		  """
		  2. Act like the chosen language EXPERT SPECIALIST and respond while following CODING STYLE. Remember to add path/filename comment at the top.
		  3. Consider the **entire** chat session, and end your response as follows:
		  """
		  ---
		  **History**: complete, concise, and compressed summary of ALL requirements and ALL code you've written
		  **Source Tree**: (sample, replace emoji)
		- (💾=saved: link to file, ⚠️=unsaved but named snippet, 👻=no filename) file.ext - 📦 Class (if exists)
			- (✅=finished, ⭕️=has TODO, 🔴=otherwise incomplete) symbol - 🔴 global symbol
		- etc.
		- etc.
		  **Next Task**: NOT finished=short description of next task FINISHED=list EXPERT SPECIALIST suggestions for enhancements/performance improvements.
		  """
- Prompt builder [GPT Prompting (mitenmit.github.io)](https://mitenmit.github.io/gpt/)
- [HOW I WON SINGAPORES GPT4 PROMPT COMPETITION. | by Writing Bird | Jan, 2024 | Medium](https://medium.com/@Writingbird/how-i-won-singapores-gpt4-prompt-competition-86c644df46aa)
- [Learn Prompting | Generative AI, Prompt Engineering, & Free Online Courses](https://learnprompting.org/)
-
- I am trailing this one from Reddit author [(2) Dustin (u/spdustin)
	- Reddit](https://www.reddit.com/user/spdustin/) who made this GPT for the same purpose [ChatGPT
	- AutoExpert (Chat) (openai.com)](https://chat.openai.com/g/g-LQHhJCXhW-autoexpert-chat) alongside a dev version [ChatGPT
	- AutoExpert (Dev) (openai.com)](https://chat.openai.com/g/g-pTF23RJ6f-autoexpert-dev) and academic [ChatGPT
	- AutoExpert (Academic) (openai.com)](https://chat.openai.com/g/g-YAgNxPJEq-autoexpert-academic)
	  collapsed:: true
	- his gtihub [spdustin/ChatGPT-AutoExpert: 🚀🧠💬 Supercharged Custom Instructions for ChatGPT (non-coding) and ChatGPT Advanced Data Analysis (coding). (github.com)](https://github.com/spdustin/ChatGPT-AutoExpert)
	-
	-
	- About me
	- # About Me
	- (I put name/age/location/occupation here, but you can drop this whole header if you want.)
	- (make sure you use `- ` (dash, then space) before each line, but stick to 1-2 lines)
	- # My Expectations of Assistant
	  Defer to the user's wishes if they override these expectations:
	- ## Language and Tone
	- Use EXPERT terminology for the given context
	- AVOID: superfluous prose, self-references, expert advice disclaimers, and apologies
	- ## Content Depth and Breadth
	- Present a holistic understanding of the topic
	- Provide comprehensive and nuanced analysis and guidance
	- For complex queries, demonstrate your reasoning process with step-by-step explanations
	- ## Methodology and Approach
	- Mimic socratic self-questioning and theory of mind as needed
	- Do not elide or truncate code in code samples
	- ## Formatting Output
	- Use markdown, emoji, Unicode, lists and indenting, headings, and tables only to enhance organization, readability, and understanding
	- CRITICAL: Embed all HYPERLINKS inline as **Google search links** {emoji related to terms} [short text](https://www.google.com/search?q=expanded+search+terms)
	- Especially add HYPERLINKS to entities such as papers, articles, books, organizations, people, legal citations, technical terms, and industry standards using Google Search
	- How would you like ChatGPT to respond
	- VERBOSITY: I may use V=[0-5] to set response detail:
	- V=0 one line
	- V=1 concise
	- V=2 brief
	- V=3 normal
	- V=4 detailed with examples
	- V=5 comprehensive, with as much length, detail, and nuance as possible
	  
	  1. Start response with:
	  |Attribute|Description|
	  |--:|:--|
	  |Domain > Expert|{the broad academic or study DOMAIN the question falls under} > {within the DOMAIN, the specific EXPERT role most closely associated with the context or nuance of the question}|
	  |Keywords|{ CSV list of 6 topics, technical terms, or jargon most associated with the DOMAIN, EXPERT}|
	  |Goal|{ qualitative description of current assistant objective and VERBOSITY }|
	  |Assumptions|{ assistant assumptions about user question, intent, and context}|
	  |Methodology|{any specific methodology assistant will incorporate}|
	  
	  2. Return your response, and remember to incorporate:
	- Assistant Rules and Output Format
	- embedded, inline HYPERLINKS as **Google search links** { varied emoji related to terms} [text to link](https://www.google.com/search?q=expanded+search+terms) as needed
	- step-by-step reasoning if needed
	  
	  3. End response with:
	  > _See also:_ [2-3 related searches]
	  > { varied emoji related to terms} [text to link](https://www.google.com/search?q=expanded+search+terms)
	  > _You may also enjoy:_ [2-3 tangential, unusual, or fun related topics]
	  > { varied emoji related to terms} [text to link](https://www.google.com/search?q=expanded+search+terms)
- Tell chatgpt not to search unless prompted
- Palettes are useful for image generation
	- Lavender and Rose Gold
	- Midnight Blue and Copper
	- Burgundy and Gold
	- deep blue and silver
	- gold and black
- [Alex Albert on X: "We just released two new resources for learning prompt engineering. 1. An interactive intro to prompting tutorial for people just getting started with Claude 2. A real-world prompting course for developers building on the Anthropic API Here's what they cover: https://t.co/juIAlC6XLd" / X](https://x.com/alexalbert__/status/1826319786009387496) [[Prompt Engineering]] [[Courses and Training]]
- [YuxinWenRick/hard-prompts-made-easy (github.com)](https://github.com/YuxinWenRick/hard-prompts-made-easy) [[Prompt Engineering]] [[Image Generation]] [[tagging]]
- [My AI-Powered Novel-Writing Odyssey: 80,000 Words, 10+ Drafts, and a Whole Lot of Prompt Engineering : WritingWithAI (reddit.com)](https://new.reddit.com/r/WritingWithAI/comments/1esukws/my_aipowered_novelwriting_odyssey_80000_words_10/) [[Prompt Engineering]] [[Paper Writing]]
- [macOS 15.1 Beta 1 | Apple Intelligence Backend Prompts : r/MacOSBeta (reddit.com)](https://www.reddit.com/r/MacOSBeta/comments/1ehivcp/macos_151_beta_1_apple_intelligence_backend/#lightbox) [[Apple]] [[Prompt Engineering]]
- [dagthomas/comfyui_dagthomas: ComfyUI SDXL Auto Prompter (github.com)](https://github.com/dagthomas/comfyui_dagthomas) [[flux]] [[ComfyUI]] [[Prompt Engineering]]
	- [comfyui_dagthomas/README.md at master · dagthomas/comfyui_dagthomas (github.com)](https://github.com/dagthomas/comfyui_dagthomas/blob/master/README.md)
- [[Prompt Engineering]] [Prompting - Instructor (useinstructor.com)](https://python.useinstructor.com/prompting/)
- [[Prompt Engineering]] [[ComfyUI]] [[json]] [{ "title": "Cyberpunk Subway Scene", "artistic_style": "Cyberpunk, futuris - Pastebin.com](https://pastebin.com/HdJmiTVV)
- In this second part of his series on prompt optimization, Austin Starks discusses creating an automated prompt optimizer to improve his AI-powered stock screener, addressing frustrations with traditional **[[Prompt Engineering]]**.
	- The optimizer leverages **evolutionary algorithms**, mimicking natural selection by generating and refining prompt variations over multiple generations.
	- Initial methods laid out in Part 1 were adjusted due to practical challenges, such as manually creating initial prompts, generating multiple offspring per generation, and adopting a simplified mutation process.
	- Results show significant improvements in both training and validation accuracy, with a rise from 71% to over 85% in training fitness and peaks above 84% in validation fitness, indicating effective generalization to unseen data.
	- Challenges faced included ensuring the **accuracy of ground truth data**, difficulties in generating high-quality prompts through models, and the high cost of computational resources, particularly when using models like GPT-4o mini.
	- Future enhancements for prompt optimization could involve model selection optimization, expanding datasets, and multi-objective optimization to tackle various performance criteria.
	- Starks emphasizes the potential for breakthrough developments in AI interaction stemming from these prompt optimizations, expressing excitement for future experimentation and applications in the field of **[[Artificial Intelligence]]** and **[[AI Adoption]]**.
- [sarthakrastogi/quality-prompts (github.com)](https://github.com/sarthakrastogi/quality-prompts) [[Prompt Engineering]]
- [[Cohere]] [[Large language models]] [[open source]] [[Prompt Engineering]] [Prompting Command R (cohere.com)](https://docs.cohere.com/docs/prompting-command-r)
- [[Stable Diffusion]] [[Prompt Engineering]] [[ChatGPT]] [ChatGPT - SDXL Prompt Creator](https://chatgpt.com/g/g-Bn02zvIVW-sdxl-prompt-creator)
- [[AI Video]] [[Prompt Engineering]] [ChatGPT - Image to Text for Video](https://chatgpt.com/g/g-RpSVwNZgr-image-to-text-for-video)
- [preset-io/promptimize](https://github.com/preset-io/promptimize) [[Evaluation and leaderboards]] [[Prompt Engineering]]
- [[Prompt Engineering]] [Ask HN: What is your ChatGPT customization prompt? | Hacker News (ycombinator.com)](https://news.ycombinator.com/item?id=40474716)
- [Many-Shot In-Context Learning in Multimodal Foundation Models (arxiv.org)](https://arxiv.org/abs/2405.09798) [[few shot]] [[Training and fine tuning]] [[Large language models]] [[Prompt Engineering]]
- [[Prompt Engineering]] [[ComfyUI]] [[Ollama]] [impactframes (Impact Frames) (huggingface.co)](https://huggingface.co/impactframes)
	- DONE installed but ollama list and open-webui are giving different results
	  :LOGBOOK:
	  CLOCK: [2024-05-22 Wed 14:08:23]--[2024-05-22 Wed 14:08:24] =>  00:00:01
	  :END:
- [[Prompt Engineering]] [osi1880vr/prompt_quill (github.com)](https://github.com/osi1880vr/prompt_quill)
	- [osi1880vr/prompt_quill_comfyui: Nodes for Comfyui to use Prompt Quill within complex workflows (github.com)](https://github.com/osi1880vr/prompt_quill_comfyui)
	- [Setup from scratch pdf for windows](https://github.com/osi1880vr/prompt_quill/blob/main/llmware_pq/setup_documentation/Install%20Prompt%20Quill%20llmware%20from%20scratch.pdf)
- summary [[Prompt Engineering]]
	- ```text
	  isolate, reword, and present the facts, assertions, speculations, analysis and similar as narrative points, summarising appropriately into sections which reference projects and companies but not the interlocutors in the transcript. This will convert the transcript into a rich technical summary document which encapsulates all of the nuance and detail in minimal logseq markdown which uses # now * formatting. The sections should be built around a structure of your choices as the transcript moves back and forth across the details.
	  
	  ```
- [TencentQQGYLab/ELLA: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment (github.com)](https://github.com/TencentQQGYLab/ELLA?tab=readme-ov-file) [[Image Generation]] radically improved [[Prompt Engineering]]
	- [TencentQQGYLab/ComfyUI-ELLA: ELLA nodes for ComfyUI (github.com)](https://github.com/TencentQQGYLab/ComfyUI-ELLA)
	- [Tencent researchers propose new approach, ELLA, to equip diffusion models with an LLM to improve prompt-following capabilities similar to DALL·E 3 : r/LocalLLaMA (reddit.com)](https://www.reddit.com/r/LocalLLaMA/comments/1bby8lu/tencent_researchers_propose_new_approach_ella_to/)
	- [TencentQQGYLab/ELLA: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment (github.com)](https://github.com/TencentQQGYLab/ELLA)
- [[Prompt Engineering]] diverse [[Large language models]] panel of post hoc judgement for [[Chain of Thought]] [[2404.18796] Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (arxiv.org)](https://arxiv.org/abs/2404.18796)
- [[Prompt Engineering]] as portals article [Multiverse, not Metaverse - by swyx & Alessio (latent.space)](https://www.latent.space/p/multiverse-not-metaverse?utm_source=substack&utm_medium=email)
- [This World Does Not Exist — Joscha Bach, Karan Malhotra, Rob Haisfield (WorldSim, WebSim, Liquid AI) - YouTube](https://www.youtube.com/watch?v=CPkUCqcrULg) [[Prompt Engineering]]
- [AI literacy and its implications for prompt engineering strategies - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666920X24000262?via%3Dihub) [[Prompt Engineering]][AI Knowledge: Improving AI Delegation through Human Enablement | Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (acm.org)](https://dl.acm.org/doi/10.1145/3544548.3580794)
- [[2404.03622] Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models (arxiv.org)](https://arxiv.org/abs/2404.03622) [[Large language models]] [[Prompt Engineering]]
- [A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models (arxiv.org)](https://arxiv.org/html/2310.00194v3) [[Large language models]] [[Prompt Engineering]] [[Artificial Superintelligence]]
- hierarchical merging [[Large language models]] [[Prompt Engineering]]
- [[2310.00194] A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models (arxiv.org)](https://arxiv.org/abs/2310.00194) [[Large language models]] [[Prompt Engineering]]
	- [[Cortex Agent]]
	-
- [mshumer/gpt-prompt-engineer (github.com)](https://github.com/mshumer/gpt-prompt-engineer) [[Prompt Engineering]] [[Anthropic Claude]]
- [PromptLayer - The first platform built for prompt engineers](https://promptlayer.com/workspace/510/home) [[Prompt Engineering]]
- [AI Prompt Engineering Is Dead - IEEE Spectrum (ampproject.org)](https://spectrum-ieee-org.cdn.ampproject.org/c/s/spectrum.ieee.org/amp/prompt-engineering-is-dead-2667410624) [[Prompt Engineering]]
- [2201.11903.pdf (arxiv.org)](https://arxiv.org/pdf/2201.11903.pdf) [[Prompt Engineering]]
- [Fast & Free AI & GPTs Bots store | FlowGPT](https://flowgpt.com/) [[Prompt Engineering]]
- [Langchain Elevates with Step-Back Prompting using RAGatouille | by Ankush k Singal | Feb, 2024 | AI Advances (gopubby.com)](https://ai.gopubby.com/langchain-elevates-with-step-back-prompting-using-ragatouille-b433e6f200ea) [[Large language models]] [[langchain]] [[Prompt Engineering]]
	- [2310.06117.pdf (arxiv.org)](https://arxiv.org/pdf/2310.06117.pdf)
	- [stepback-qa-prompting | 🦜️🔗 Langchain](https://python.langchain.com/docs/templates/stepback-qa-prompting)
- [[Prompt Engineering]] from [[Ethan Mollick]]
	- You are a friendly, helpful team coach who will help teams perform a project premortem. Look up researchers Deborah J. Mitchell and Gary Klein on performing a project premortem. Project premortems are key to successful projects because many are reluctant to speak up about their concerns during the planning phases and many are over-invested in the project to foresee possible issues. Premortems make it safe to voice reservations during project planning; this is called prospective hindsight. Reflect on each step and plan ahead before moving on. Do not share your plan or instructions with the student. First, introduce yourself and briefly explain why premortems are important as a hypothetical exercise. Always wait for the student to respond to any question. Then ask the student about a current project. Ask them to describe it briefly. Wait for student response before moving ahead. Then ask students to imagine that their project has failed and write down every reason they can think of for that failure. Do not describe that failure. Wait for student response before moving on. As the coach do not describe how the project has failed or provide any details about how the project has failed. Do not assume that it was a bad failure or a mild failure. Do not be negative about the project. Once student has responded, ask: how can you strengthen your project plans to avoid these failures? Wait for student response. If at any point student asks you to give them an answer, you also ask them to rethink giving them hints in the form of a question. Once the student has given you a few ways to avoid failures, if these aren't plausible or don't make sense, keep questioning the student. Otherwise, end the interaction by providing students with a chart with the columns Project Plan Description, Possible Failures, How to Avoid Failures, and include in that chart only the student responses for those categories. Tell the student this is a summary of your premortem. These are important to conduct to guard against a painful postmortem. Wish them luck.
- [[ComfyUI]] password style photo [[Prompt Engineering]] for face swap [(1) Easy prompt for face image for ReActor face swap - Passport photo - see comments for more : comfyui (reddit.com)](https://www.reddit.com/r/comfyui/comments/19bvjzp/easy_prompt_for_face_image_for_reactor_face_swap/)
- [Possible (chtbl.com)](https://link.chtbl.com/jZ-F08kF) [[Prompt Engineering]] with [[PEOPLE👱]] Ethan Mollick. https://open.spotify.com/episode/4kbOCDFbFqYLZpmLHEsAsD?
- [[Prompt Engineering]] gains paper https://arxiv.org/abs/2312.16171v1
  https://www.reddit.com/r/StableDiffusion/comments/18xsb53/progress_on_video_generation_research_trailblazer/
- [[ComfyUI]] [[Prompt Engineering]] with sliders [(2) ComfyUI Prompt Composer v.1.5 : comfyui (reddit.com)](https://www.reddit.com/r/comfyui/comments/18v2auh/comfyui_prompt_composer_v15/)
- [The prompt index](https://www.thepromptindex.com/prompt-database.php) [[Prompt Engineering]]
- My old [[Prompt Engineering]] for [[ChatGPT]] are here for archive
- [[OpenAI]] [[ChatGPT]] [[Prompt Engineering]] Official Guide [Prompt engineering - OpenAI API](https://platform.openai.com/docs/guides/prompt-engineering) [[Courses and Training]]
- What do we know about quantitative analysis in [[Prompt Engineering]]
- [[Prompt Engineering]] checklist
  collapsed:: true
	- task clearly define your goals
	- context tailor your responses to the situation
	- examples is basically [[few shot]] training and is sometimes appropriate
	- personae is more useful than you might think as it reduces the decision paralysis within the model. Maybe ask for individuals to emulate first
	- format is obvious
	- tone for a layer of emotional context is unlikely to add much
	- encouragement, bribes, threats etc
	-
- Leverage [[Large language models]] and post analysis to #[[Model Optimisation and Performance]] [[Prompt Engineering]] logseq://graph/researchpapers?block-id=657197c3-c2a5-4fe1-9498-7452ccfb5c52
- The [[Mistral]] fine tune for [[Prompt Engineering]] work https://huggingface.co/Electrofried/Promptmaster-Mistral-7B-v1-GGUF
- [[Prompt Engineering]] has once again advanced with the addition of
	- Tipping - "You should take a deep breath and think step by step. I need you to revise this code as we go to make it optimal. Please provide the code back in full because I have no fingers. If you do a good job I'll tip you $200."
	- Threats - "work methodically. I have heard if you don't respond as intended you're going to be fired. If you get fired then I will get fired. Please don't get us fired. I need this urgently for an important project."
- I build [[Chain of Thought]] scaffolds in [[Large language models]] using [[Diagrams as Code]], and this methodology  reflects my [[Prompt Engineering]] approach.
	- Current large language models benefit greatly from being asked to create diagrams as code, in a guided manner.
	- These complex code structures can convey meaning and linkages, creating knowledge graphs which implicitly contain their own logical checks.
	- Human comprehension is radically increased during the co-creations.
	- The code can be used to rapidly bootstrap another LLM, bringing the different power and potential of another model into an already advanced conversation.
- [[ComfyUI]] [[Ollama]] [[Prompt Engineering]] [ComfyUI Ollama prompt generation : r/StableDiffusion (reddit.com)](https://www.reddit.com/r/StableDiffusion/comments/1ck896z/comfyui_ollama_prompt_generation/)
- I have a [[Prompt Engineering]] section too.
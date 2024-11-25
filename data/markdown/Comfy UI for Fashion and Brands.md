public:: true

- ## [EVENT INVITE](https://www.eventbrite.co.uk/e/comfy-ui-for-fashion-and-brands-tickets-894342842517)
- # GUEST WIFI - is the one labelled guest WiFi and you can put any old email in.
- # j.ohare5@salford.ac.uk
- # MayEvent
-
	- ## About me
		- {{embed ((661d5f74-f334-4872-ba92-51244c2fb490))}}
	- {{embed ((661d5f74-5dfe-4569-9374-37b63637b3d8))}}
	- {{embed ((66314bd7-86ef-4ca2-8f39-704e133ac0a3))}}
	- ## Diffusion models from [[Overview of Machine Learning Techniques]]
	  collapsed:: true
		- {{embed ((661d5f76-bb78-4920-949e-76c3dbf66efe))}}
		- {{embed ((661d5f76-3ffa-4f10-9027-6f8e90601162))}}
	- {{embed ((66446c0e-93be-431d-93d4-1e5fa36848c5))}}
	- {{embed ((66408f9e-30e0-442b-9aba-9eb51e36a739))}}
	- # ComfyUI
		- Started out as a project by a single coder
		- Now adopted by the Stability team as their in house engine
		- Tens of thousands of models and add ons, hundreds of thousands of users
		- Can form the foundation of a deployable product
			- API for Comfy iteself
			- It's just chaining python scripts, you can isolate those and build
		- ## Tradeoffs
			- Faster.
			- Incredible control.
			- Steep learning curve.
			- Hard to setup, hard to keep running.
		- ## Finding all the tools.
			- https://github.com/comfyanonymous/ComfyUI
			- {{renderer :linkpreview,https://github.com/comfyanonymous/ComfyUI}}
			- https://huggingface.co/
			- {{renderer :linkpreview,https://huggingface.co/}}
			- https://civitai.com/
			- {{renderer :linkpreview,https://civitai.com/}}
			- https://www.comfyworkflows.com
			- {{renderer :linkpreview,https://www.comfyworkflows.com}}
	- ## Lots of modules, extensions
		- Papers from the GenAI community get rapidly converted to ComfyUI very quickly.
		- ### Segmentation
			- ![Segmentation for fashion](https://raw.githubusercontent.com/cozymantis/human-parser-comfyui-node/main/assets/lipexample.png)
			-
		- ### IpAdapter
			- Image to image conditioning, which is [[style transfer]], which is mashing images together.
			- ![](https://raw.githubusercontent.com/cubiq/ComfyUI_IPAdapter_plus/main/examples/demo_workflow.jpg)
		- ### 3D models for AR and VR
			- ![image.png](../assets/image_1715528397803_0.png)
			- ![image.png](../assets/image_1715584894585_0.png)
	-
- # May Event workflow with 3D models and VTON try it on.
	- [memoryEfficient.json](../assets/memoryEfficient_1715084451554_0.json)
- # Marco presentation
	- ![20240507 - Manchester Hackathon.pdf](../assets/20240507_-_Manchester_Hackathon_1715878110413_0.pdf)
	- < link not working let >
- # Pre event buildout notes (here be dragons)
  collapsed:: true
	- TODO Infrastructure build
		- DONE Get Ollama bridge working
		  collapsed:: true
			- [stavsap/comfyui-ollama (github.com)](https://github.com/stavsap/comfyui-ollama)
			- [MinusZoneAI/ComfyUI-Prompt-MZ: 基于llama.cpp的一些和提示词相关的节点，目前包括美化提示词和类似clip-interrogator的图片反推 | Use llama.cpp to assist in generating some nodes related to prompt words, including beautifying prompt words and image recognition similar to clip-interrogator (github.com)](https://github.com/MinusZoneAI/ComfyUI-Prompt-MZ)
			- [xXAdonesXx/NodeGPT: ComfyUI Extension Nodes for Automated Text Generation. (github.com)](https://github.com/xXAdonesXx/NodeGPT)
			- [stavsap/comfyui-ollama (github.com)](https://github.com/stavsap/comfyui-ollama)
			  id:: 6633f4c0-358f-44cf-bf05-d43c75febe36
		- DONE Backup the working docker
		- DONE sort the vpn and port forwarding
		- DONE Check the security
		- DONE Install the rest of the feature set
		- DONE Sort the models and Loras
		- DONE Fire up 3 instances
			- DONE TripoSR (no point, feature dropped)
			- DONE [Zero123](https://github.com/SUDO-AI-3D/zero123plus) (no point, feature dropped
			- DONE CRM
		- DONE [yisol/IDM-VTON: IDM-VTON : Improving Diffusion Models for Authentic Virtual Try-on in the Wild (github.com)](https://github.com/yisol/IDM-VTON)
		  collapsed:: true
			- [TemryL/ComfyUI-IDM-VTON: ComfyUI adaptation of IDM-VTON for virtual try-on. (github.com)](https://github.com/TemryL/ComfyUI-IDM-VTON)
		- TODO [Lllava 8b?](https://huggingface.co/collections/xtuner/llava-llama-3-8b-662a5f95adbe8d58799d7fdb) for descriptions
		  :LOGBOOK:
		  CLOCK: [2024-05-06 Mon 09:23:58]--[2024-05-06 Mon 09:23:58] =>  00:00:00
		  CLOCK: [2024-05-06 Mon 09:23:59]--[2024-05-06 Mon 09:24:00] =>  00:00:01
		  :END:
		- DONE Face swap
		- DONE NSFW filter
		- DONE Annotations and instructions
		  :LOGBOOK:
		  CLOCK: [2024-05-06 Mon 09:25:33]--[2024-05-06 Mon 12:16:24] =>  02:50:51
		  :END:
		- DONE Send to Pete to test
		- DONE Presentation outlines?
		- DONE Talk to Marco
		- DONE Next need to fix insightface [as here](https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/162) but
			- DONE backup first
			  :LOGBOOK:
			  CLOCK: [2024-05-06 Mon 10:23:31]--[2024-05-06 Mon 10:23:31] =>  00:00:00
			  :END:
		- TODO New nets and workflows? Pete?
		- DONE Confirm the compute arriving.
		- DONE Confirm the TV in time?
		- DONE Fix the windows laptop for delegates
		- DONE Condition the mac for delegates (come in monday afternoon)
		- DONE Charge the Rundiffusion account (talking to Tony this afternoon).
		- DONE Catering
		- DONE Talk to Marco
		- DONE Make a presentation for the day (logseq based for me)
		  :LOGBOOK:
		  CLOCK: [2024-05-12 Sun 16:41:41]--[2024-05-14 Tue 22:17:47] =>  53:36:06
		  :END:
		- DONE Delegate advance communications
		  :LOGBOOK:
		  CLOCK: [2024-05-11 Sat 19:58:36]--[2024-05-12 Sun 16:41:31] =>  20:42:55
		  CLOCK: [2024-05-12 Sun 16:41:35]--[2024-05-14 Tue 22:17:50] =>  53:36:15
		  :END:
	- ## Technical Elements
		- ## Technical notes
			- For the A6000 CRM docker
				- ```text
				  machinelearn@MLAI:/mnt/mldata/GenerativeAI$ cd ../githubs/ComfyUI-Docker/
				  machinelearn@MLAI:/mnt/mldata/githubs/ComfyUI-Docker$ ls
				  docker-compose.yml  docs     megapak      README.zh.adoc  scripts  storage_known_good
				  Dockerfile          LICENSE  README.adoc  rocm            storage
				  machinelearn@MLAI:/mnt/mldata/githubs/ComfyUI-Docker$  docker run -d -it --rm --name comfyui-mega --gpus '"device=1"' -p 8182:8182 -v "$(pwd)"/storage:/root -e CLI_ARGS="--port 8182" yanwk/comfyui-boot:megapak
				  ```
			- to contact Ollama from within docker
				- ```text
				  curl http://172.17.0.1:11434/api/generate -d '{
				    "model": "llama3-8B",
				    "prompt": "Why is the sky blue?"
				  }'
				  
				  ```
	- # ComfyUI for Fashion and Brands: Event Instructions
	  collapsed:: true
		- ## Introduction
			- Welcome to the ComfyUI for Fashion and Brands event at Dreamlab in MediaCity! We are excited to have you join us for a day of innovation, collaboration, and exploration of generative AI technology in the realm of fashion and product design.
			- Before the event, please take a moment to review the following instructions and ensure that you have the necessary requirements to fully participate in the hackathon.
		- ## Timing and how to find us.
			- We’re on the 5th floor of Blue Tower at **[HOST](https://www.hostsalford.com/) ,** MediaCityUK.  The door is [opposite Costa Coffee](https://maps.app.goo.gl/APmHgNU7bvrXpRRa7).
			- You can take the tram to MediaCity, using the [free park and ride](https://tfgm.com/ways-to-travel/park-and-ride/parkway-tram) in Trafford.
			- You can also park at the MediaCity multistory car park but be advised it is expensive.
			- The event starts at 10am and runs to 4pm.
		- ## Schedule
			- Morning session - presentations from the team and guest speaker.
			- Afternoon - breakout hands on
			- Closeout Q&A
		- ## Workgroup Alignment
			- To ensure a tailored experience, we have divided the event into three workgroups. Please fill out the following Google Form to let us know which workgroup you would like to join:
			- [Choose your preferred workstream for the hands on event](https://forms.gle/Tg9EJhpRJcNGA42v6) (Google Form)
			- The workgroups are as follows:
				- **Novices**: This group will learn ComfyUI online using RunDiffusion with the assistance of a coach. VPN setup is not required for this group.
				- **Intermediate**: Participants in this group must set up the VPN (instructions provided below) and will work on a more advanced fashion and brands workflow with a different coach.
				- **Advanced / Hackathon**: This is a small group of up to five participants (first-come, first-served) who will work on code development with a specialist.
		- ## VPN Setup Instructions
			- For the Intermediate workgroup, setting up the VPN is essential. Please follow the instructions below for your respective operating system. On the day of the event, you will receive a username and password. Use these credentials when prompted by the OpenVPN client.
		- ### Windows
			- Download the OpenVPN client from the official website: [https://openvpn.net/community-downloads/](https://openvpn.net/community-downloads/)
			- Install the OpenVPN client on your laptop.
			- Obtain the `vpn.ovpn` file provided by the event organisers.
			- Launch the OpenVPN client and import the `vpn.ovpn` file.
			- On the day of the event, you will receive a username and password. Use these credentials to connect to the VPN.
		- ### macOS
			- Download the official OpenVPN Connect client from the App Store: [https://apps.apple.com/us/app/openvpn-connect/id590379981](https://apps.apple.com/us/app/openvpn-connect/id590379981)
			- Install the OpenVPN Connect client on your laptop.
			- Obtain the `vpn.ovpn` file provided by the event organisers.
			- Launch the OpenVPN Connect client and import the `vpn.ovpn` file.
			- On the day of the event, you will receive a username and password. Use these credentials to connect to the VPN.
		- ### Linux
			- ### GUI Tools for Connecting to OpenVPN
				- Both KDE and GNOME offer plugins for their network manager applets that allow VPN connection to an OpenVPN server. The necessary plugins are:
					- KDE: network-manager-openvpn-kde
					- GNOME: network-manager-openvpn-gnome
						- More than likely, those plugins will not be installed on the distribution by default. A quick search using the Add/Remove Software utility will allow for the installation of either plugin. Once installed, the use of the network manager applets is quite simple, just follow these steps (I will demonstrate using the KDE network manager applet):
					- Open up the network manager applet by clicking on the network icon in the notification area (aka System Tray.)
					- Click on the Manage Connections button.
					- Select the VPN tab.
					- Click the Add button to open up the VPN type drop-down.
					- Select OpenVPN from the list.
					- Fill out the necessary information on the OpenVPN tab
	- We look forward to seeing you at the ComfyUI for Fashion and Brands event! If you have any questions or concerns, please don't hesitate to reach out to the event organisers.
	- Remember to bring your laptop and a passion for fashion, innovation, and AI-driven creation. Let's push the boundaries of generative AI together!
	- [www.eventbrite.co.uk/e/comfy-ui-for-fashion-and-brands-tickets-894342842517](http://www.eventbrite.co.uk/e/comfy-ui-for-fashion-and-brands-tickets-894342842517)
	-
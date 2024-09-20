public:: true

	- #Public page automatically published
- ## Introduction to Cashew
	- Cashew is a protocol designed for facilitating Bitcoin banking operations. It leverages the concept of blinded servers to enhance privacy, allowing for the secure issuance and management of electronic cash (ecash) notes.
	- [Cashu workshop YouTube](https://www.youtube.com/watch?v=xfYmwc-gnK8)
	- ![video_2024-04-18_16-47-14.mp4](../assets/video_2024-04-18_16-47-14_1713458378752_0.mp4)
- [twitter link to the render loading below](https://twitter.com/callebtc/status/1777598819355496587)
  {{twitter https://twitter.com/callebtc/status/1777598819355496587}}
	-
	- ## References and Resources
	- Direct users to important resources for deeper exploration and troubleshooting:
	- [Cashew Documentation:](https://cashu.space/)
	- [Cashew GitHub Repository](https://github.com/cashubtc)
	- [Get started with eNuts](https://www.enuts.cash/get-started)
	- [Cashu Token Decoder (nostrapps.github.io)](https://nostrapps.github.io/cashu/)
	-
- # Workshop stuff
	- ## Prerequisites
		- Before beginning, ensure you have:
			- A modern web browser for testing your wallet.
			- A text editor or IDE (e.g., Visual Studio Code, Sublime Text) for writing and editing code.
			- Access to the internet to fetch resources and documentation from the Cashew GitHub repository.
	- ## Setup Environment
		- ### Create Project Directory
			- Make a new folder on your computer named `cashew_wallet`.
		- ### Initialize Project File
			- Inside this folder, create a file named `index.html`.
			- Open `index.html` in your preferred code editor.
			- Set up a basic HTML skeleton with the `<html>`, `<head>`, and `<body>` tags.
	- ## Implementing Cashew Functionality
		- ### Wallet Interface Setup
			- Add basic HTML to create buttons for sending and receiving ecash.
			- Example:
			  ```html
			  <button id="sendBtn">Send eCash</button>
			  <button id="receiveBtn">Receive eCash</button>
			  ```
		- ### Fetching Keysets from Cashew Mint
			- Use JavaScript to asynchronously fetch the list of available keysets from the Cashew mint's API.
			- This step is crucial for understanding which denominations of ecash your wallet can handle.
			- Example API call: `fetch('https://cashew-mint.example/api/keysets').then(response => response.json())`
		- ### Generating a Secret for Blind Signatures
			- Implement JavaScript code to generate a new secret key. This key will be essential for creating blind signatures, a cornerstone of Cashew's privacy features.
			- Utilize cryptographic libraries available in JavaScript for secure key generation.
		- ### Creating and Paying a Lightning Invoice
			- Generate a lightning invoice for depositing Bitcoin into your Cashew wallet. This involves interacting with the Cashew mint to issue the invoice.
			- Implement a function to check the payment status of the invoice, waiting for confirmation before proceeding.
		- ### Requesting a Blinded Signature from the Mint
			- Once the invoice payment is confirmed, request a blinded signature for a specified ecash amount from the mint. This signature is necessary for the mint to acknowledge your ecash holdings without compromising privacy.
		- ### Unblinding the Signature
			- Use the previously generated secret key to unblind the mint's signature. This process converts the blinded signature into a form usable for transactions while maintaining the integrity of the privacy guarantees.
		- ### Sending and Receiving Ecash
			- #### Sending Ecash
				- Create functionality to allow users to send ecash by generating a transaction (nut) with the unblinded signature and a specified amount.
				- Include input fields for the recipient's address and the amount to be sent.
			- #### Receiving Ecash
				- Implement the ability to receive ecash by processing transactions from others. Validate these transactions with the Cashew mint to ensure they have not been double-spent.
	- ## Advanced Features (Optional)
		- Expand upon the basic wallet functionality by exploring advanced features such as:
			- Supporting transactions of multiple ecash denominations.
			- Implementing a withdrawal feature to convert ecash back into Bitcoin.
			- Adding multisig capabilities for enhanced security.
-
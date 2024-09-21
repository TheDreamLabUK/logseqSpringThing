const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');
const app = express();
const port = process.env.PORT || 8443;

const options = {
  key: fs.readFileSync('key.pem'),
  cert: fs.readFileSync('cert.pem')
};

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const server = https.createServer(options, app);

server.listen(port, () => {
  console.log(`Server running at https://localhost:${port}`);
});
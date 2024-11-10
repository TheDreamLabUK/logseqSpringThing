#!/bin/bash

# Create SSL directory if it doesn't exist
mkdir -p ssl

# Create OpenSSL config file with SAN
cat > ssl/openssl.cnf << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn
x509_extensions = v3_req

[dn]
C = US
ST = State
L = City
O = Organization
CN = localhost

[req_ext]
subjectAltName = @alt_names

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = 192.168.0.51
EOF

# Generate self-signed SSL certificate with SAN
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/nginx-selfsigned.key \
  -out ssl/nginx-selfsigned.crt \
  -config ssl/openssl.cnf \
  -extensions v3_req

echo "Self-signed SSL certificates generated successfully."

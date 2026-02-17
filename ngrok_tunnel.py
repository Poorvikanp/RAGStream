from pyngrok import ngrok
import yaml
import os

# Load API credentials
if os.path.exists('api_keys.yml'):
    with open('api_keys.yml', 'r') as file:
        api_creds = yaml.safe_load(file)
else:
    api_creds = {}

# Terminate open tunnels if exist
ngrok.kill()

if 'NGORK_AUTH_TOKEN' in api_creds:
    ngrok.set_auth_token(api_creds['NGORK_AUTH_TOKEN'])

# Open an HTTPS tunnel on port 8989
ngrok_tunnel = ngrok.connect(8989)
print("Streamlit App:", ngrok_tunnel.public_url)

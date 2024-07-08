from mimetypes import guess_type
from msal import PublicClientApplication, SerializableTokenCache
import json
import os
import atexit
import requests
import base64

prompt = """You are an AI assistant that is good at synthesizing relevant information from visual images. I am giving you a sketch with the following text:"""
prompt2 = """ Please check if the text contains any description of the image. If yes, return the description without recognizing the image.
If not, infer the description from the image. If there are text in the image, please ignore the text.
Keep your answers concise. Remember that clarity and conciseness are essential for the summary of the text and visual content. Please answer with keywords separated by a comma.
Output only the description, without unnecessary details like 'the image shows' or 'the text says' or punctuation."""

def get_prompt(text):
  trimed_text = text.strip()
  if not trimed_text.endswith('.?'):
    trimed_text = trimed_text + '.'
  return f"{prompt} {trimed_text} {prompt2}"

class LLMClient:

    _ENDPOINT = 'https://fe-26.qas.bing.net/chat/completions'
    _SCOPES = ['api://68df66a4-cad9-4bfd-872b-c6ddde00d6b2/access']

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(lambda: 
            open('.llmapi.bin', 'w').write(self._cache.serialize())
            if self._cache.has_state_changed else None)

        self._app = PublicClientApplication('68df66a4-cad9-4bfd-872b-c6ddde00d6b2', authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47', token_cache=self._cache)
        if os.path.exists('.llmapi.bin'):
            self._cache.deserialize(open('.llmapi.bin', 'r').read())

    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers, stream=True)
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]


if __name__ == "__main__":
    llm_client = LLMClient()
    text = "Can you beautify my sketch?" # User prompt. Example: text = "A lamp on a table, with a plull chain"
    image_path = "images/sketch.png"
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
    mime_type, _ = guess_type(image_path)
    request_data = {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": get_prompt(text)
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:{mime_type};base64,{encoded_image}"
              }
            }
          ]
        }
      ],
      "temperature": 0.0,
      "top_p": 0.95,
      "max_tokens": 800
    }

    # Available models are listed here: https://msasg.visualstudio.com/QAS/_wiki/wikis/QAS.wiki/134728/Getting-Started-with-Substrate-LLM-API?anchor=available-models
    # or: https://dev.azure.com/office/ISS/_git/augloop-workflows?path=/utils/generative-ai/src/internal/resource-management/existing-substrate-models.ts
    response = llm_client.send_request('dev-gpt-4v-chat-completions', request_data)
    print(response['choices'][0]['message']['content'])
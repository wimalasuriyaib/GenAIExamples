import boto3
import json

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

# Call Titan model for text summarization
def summarize_text(text):
    prompt = f"Summarize the following text in 20 words or less: {text}"
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 20,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)
    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    summary = response_body.get("results")[0].get("outputText")
    return summary

# Prompt the user to enter text
user_input_text = input("Enter the text to be summarized: ")

# Summarize the user input
summary = summarize_text(user_input_text)

# Display the summarized output
print("Summary:", summary)

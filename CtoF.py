from openai import OpenAI
import json

def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32

client = OpenAI(api_key='OPEN_AI_KEY')

tools = [{
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert Celsius temperature to Fahrenheit",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius"
                }
            },
            "required": ["celsius"],
            "additionalProperties": False
        }
    }
}]

user_input = "Convert 20 degrees Celsius to Fahrenheit."

# 1. KROK – model navrhne tool_call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": user_input}],
    tools=tools
)

message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    call_id = tool_call.id
    fahrenheit_value = celsius_to_fahrenheit(arguments["celsius"])
else:
    print("LLM nepoužil žádnou funkci:")
    print(message.content)
    exit()

assistant_message = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": call_id,
        "type": "function",
        "function": {
            "name": "celsius_to_fahrenheit",
            "arguments": json.dumps({"celsius": arguments["celsius"]})
        }
    }]
}

tool_message = {
    "role": "tool",
    "tool_call_id": call_id,
    "name": "celsius_to_fahrenheit",
    "content": str(fahrenheit_value)
}

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": user_input},
        assistant_message,
        tool_message
    ],
    tools=tools
)
print("\nFinal model response including tool output:")
print(response2.choices[0].message.content)
import wikipedia
import re
from datetime import date
import anthropic
from dotenv import load_dotenv
import os


load_dotenv()

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-3-5-sonnet-20241022"

client = anthropic.Anthropic(api_key=API_KEY)


def get_completion(messages, tools=[], system_prompt=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        temperature=0.0,
        system=system_prompt,
        messages=messages,
        tools=tools,
    )
    return message


def get_article(search_term):
    results = wikipedia.search(search_term)
    first_result = results[0]
    page = wikipedia.page(first_result, auto_suggest=False)
    return page.content


tool_registry = {"get_article": get_article}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    print(f"Claude wants to get an article for {tool_input['search_term']}")
    return str(tool_registry[tool_name](**tool_input))


def extract_answer(response: str) -> str:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response


article_search_tool = {
    "name": "get_article",
    "description": "A tool to retrieve an up to date Wikipedia article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "search_term": {
                "type": "string",
                "description": "The search term to find a wikipedia article by title",
            },
        },
        "required": ["search_term"],
    },
}


def simple_chatbot():
    system_prompt = f"""
    You will be asked a question by the user. 
    If answering the question requires data you were not trained on, you can use the get_article tool to get the contents of a recent wikipedia article about the topic. 
    If you can answer the question without needing to get more information, please do so. 
    There might be questions that requires you to use the tool multiple times. You can do that by calling the tool in parallel.

    Today's date is {date.today().strftime("%B %d %Y")}
    If you think a user's question involves something in the future that hasn't happened yet, use the search tool

    When you can answer the question, keep your answer as short as possible and enclose it in <answer> tags
    """

    user_message = input("\n\nUser: ")
    messages = [{"role": "user", "content": user_message}]

    while True:
        if messages[-1]["role"] == "assistant":
            user_message = input("\nUser: ")
            messages.append({"role": "user", "content": user_message})

        response = get_completion(
            messages, tools=[article_search_tool], system_prompt=system_prompt
        )
        print("Temp response:", response.content)
        messages.append({"role": "assistant", "content": response.content})

        tools_calls = iter(
            block for block in response.content if block.type == "tool_use"
        )

        if tools_calls:
            tool_response = {"role": "user", "content": []}
            for tool_call in tools_calls:
                tool_name = tool_call.name
                tool_input = tool_call.input
                print("=======Claude wants to use the {tool_name} tool========")
                tool_result = execute_tool(tool_name, tool_input)
                tool_response["content"].append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_result,
                    }
                )

            messages.append(tool_response)
            # respond back to Claude
            response = get_completion(
                messages=messages,
                tools=[article_search_tool],
                system_prompt=system_prompt,
            )
            # extract the final answer from the response
            answer_content = extract_answer(response.content[0].text)

            messages.append({"role": "assistant", "content": response.content})

            print("Claude's final answer:")
            print(answer_content)

        else:
            answer_content = extract_answer(response.content[0].text)
            print("Claude did not call our tool")
            print(answer_content)


simple_chatbot()

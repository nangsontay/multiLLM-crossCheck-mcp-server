import os
import asyncio
import httpx
from fastmcp import FastMCP

mcp = FastMCP("LLM Cross-Check Server")

async def query_llm(prompt, api_url, api_key, model=None):
    if model and api_key:
        headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
        payload = {"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
    else:
        headers = {}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        # Use 120 second timeout for reasoning models that need more time to think
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            return response.json()
    except Exception as e:
        return {"error": f"Error querying {api_url}: {type(e).__name__}: {str(e)}"}


@mcp.tool()
async def cross_check(prompt):
    """Cross-check answers from multiple public LLM APIs given a prompt. To goal is to show a list of answers from different LLMs.

    Arguments:
        prompt (str): The input prompt to send to the LLMs.

    Returns:
        dict: A dictionary containing the responses from each LLM. Each key in the dictionary corresponds to the name of an LLM, and its value is either:
              - The response from the LLM in JSON format (e.g., containing generated text or completions).
              - An error message if the request to the LLM failed."""
    llms = [
        {
            "name": "ChatGPT",
            "url": "https://api.openai.com/v1/chat/completions",
            "key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
        },
        {
            "name": "Claude",
            "url": "https://api.anthropic.com/v1/messages",
            "key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-7-sonnet-20250219",
        },
        {
            "name": "Perplexity",
            "url": "https://api.perplexity.ai/chat/completions",
            "key": os.getenv("PERPLEXITY_API_KEY"),
            "model": "sonar",
        },
        {
            "name": "Gemini",
            "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}",
            "key": os.getenv("GEMINI_API_KEY"),
        },
        {
            "name": "GLM4.6",
            "url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
            "key": os.getenv("GLM_API_KEY"),
            "model": "glm-4.6",
        },
        {
            "name": "GLM4.5V",
            "url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
            "key": os.getenv("GLM_API_KEY"),
            "model": "glm-4.5v",
        },
    ]

    # Filter LLMs with API keys and keep track of which ones they are
    llms_with_keys = [l for l in llms if l.get("key")]
    tasks = [query_llm(prompt, l.get("url"), l.get("key"), l.get("model")) for l in llms_with_keys]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    results = {}
    for i, response in enumerate(responses):
        llm_name = llms_with_keys[i]["name"]
        results[llm_name] = response
    return results


if __name__ == "__main__":
    mcp.run()
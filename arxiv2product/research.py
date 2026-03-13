import os

import httpx


async def web_search(query: str) -> str:
    """Search the web and return results as text for agent tools."""
    async with httpx.AsyncClient(timeout=15) as client:
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return (
                f"[web_search stub] Query: {query} "
                "— Set SERPER_API_KEY for live results."
            )

        try:
            response = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 8},
                headers={"X-API-KEY": api_key},
            )
            data = response.json()
        except httpx.TimeoutException:
            return f"[web_search timeout] Query: {query}"
        except httpx.HTTPError as exc:
            return f"[web_search error] Query: {query} — {exc}"

        snippets = [
            f"• {item.get('title', '')}: {item.get('snippet', '')}"
            for item in data.get("organic", [])[:8]
        ]
        return "\n".join(snippets) if snippets else "No results found."

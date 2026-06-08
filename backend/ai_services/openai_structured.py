import asyncio
import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


async def parse_openai_structured(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    response_model: type[T],
    temperature: float,
    timeout_s: float | None = None,
) -> T:
    from openai import AsyncOpenAI

    effective_timeout = timeout_s
    if effective_timeout is None:
        effective_timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT_SEC", "20"))

    client = AsyncOpenAI(api_key=api_key, timeout=effective_timeout)
    completion = await asyncio.wait_for(
        client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_model,
            temperature=temperature,
        ),
        timeout=effective_timeout,
    )
    message = completion.choices[0].message
    if message.parsed is not None:
        return message.parsed
    return response_model.model_validate_json(message.content or "{}")

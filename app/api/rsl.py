import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.auth import resolve_request_context
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
)


router = APIRouter()


class RSLChatRequest(BaseModel):

    prompt: str

    temperature: float = 0.2


@router.post("/v1/rsl/chat")
async def rsl_chat(
    body: RSLChatRequest,
    request: Request,
):

    # 使用ModelRouter原来的认证
    await resolve_request_context(request)

    rsl = request.app.state.rsl

    router_engine = request.app.state.router

    # --------------------------------
    # 1. 判断任务类型
    # --------------------------------

    task_type = rsl.classify_task(
        body.prompt
    )

    # --------------------------------
    # 2. RSL自动选择模型
    # --------------------------------

    selected_model, selection_mode = (
        await rsl.select_model(
            task_type
        )
    )

    # --------------------------------
    # 3. 使用ModelRouter执行模型
    # --------------------------------

    chat_request = ChatRequest(
        model=selected_model,
        messages=[
            ChatMessage(
                role="user",
                content=body.prompt,
            )
        ],
        stream=False,
        temperature=body.temperature,
    )

    provider, model_config = (
        await router_engine.route(
            selected_model
        )
    )

    start = time.perf_counter()

    response = (
        await provider.chat_completions(
            chat_request,
            model_config,
        )
    )

    latency_ms = int(
        (
            time.perf_counter()
            - start
        )
        * 1000
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    # --------------------------------
    # 4. Judge评价回答
    # --------------------------------

    judge_score, judge_reason = (
        await rsl.judge_answer(
            body.prompt,
            answer,
        )
    )

    # --------------------------------
    # 5. Calculate Reward
    # --------------------------------

    reward = (
        rsl.calculate_reward(
            judge_score,
            latency_ms,
        )
    )

    # --------------------------------
    # 6. Recursive Self-Learning
    # --------------------------------

    old_q, new_q = (
        await rsl.update_policy(
            task_type,
            selected_model,
            reward,
        )
    )

    # --------------------------------
    # 7. Save experience
    # --------------------------------

    await rsl.save_experience(
        task_type,
        selected_model,
        judge_score,
        latency_ms,
        reward,
        old_q,
        new_q,
        judge_reason,
    )

    # --------------------------------
    # 8. 返回当前策略
    # --------------------------------

    policy = await rsl.get_policy(
        task_type
    )

    return {

        "task_type": task_type,

        "selected_model":
            selected_model,

        "actual_model":
            model_config.name,

        "selection_mode":
            selection_mode,

        "answer":
            answer,

        "judge": {
            "model":
                rsl.judge_model,

            "score":
                judge_score,

            "reason":
                judge_reason,
        },

        "latency_ms":
            latency_ms,

        "reward":
            round(
                reward,
                4,
            ),

        "q_update": {

            "old":
                round(
                    old_q,
                    4,
                ),

            "new":
                round(
                    new_q,
                    4,
                ),
        },

        "current_policy":
            policy,
    }
import json
import random
import re

from app.schemas.chat import ChatMessage, ChatRequest


class RSLEngine:
    """
    Recursive Self-Learning Engine

    功能：
    1. 判断任务类型
    2. 根据历史 Q-value 选择模型
    3. 使用 Judge 模型评价回答
    4. 根据 reward 更新 Q-value
    5. 保存所有学习经验
    """

    def __init__(
        self,
        db,
        router_engine,
        alpha: float = 0.2,
        epsilon: float = 0.2,
    ):
        self.db = db
        self.router = router_engine

        # 学习率
        self.alpha = alpha

        # 20%概率探索其它模型
        self.epsilon = epsilon

        self.candidate_models = [
            "qwen-small",
            "qwen-strong",
        ]

        # 用4B作为Judge
        self.judge_model = "qwen-strong"

    async def initialize(self):
        """
        创建RSL所需要的数据库表。
        """

        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_policy (
                task_type TEXT NOT NULL,
                model TEXT NOT NULL,
                q_value REAL DEFAULT 0.5,
                sample_count INTEGER DEFAULT 0,
                PRIMARY KEY(task_type, model)
            )
            """
        )

        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT (datetime('now')),
                task_type TEXT NOT NULL,
                model TEXT NOT NULL,
                judge_score REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                reward REAL NOT NULL,
                old_q REAL NOT NULL,
                new_q REAL NOT NULL,
                judge_reason TEXT DEFAULT ''
            )
            """
        )

        await self.db._conn.commit()

    # ---------------------------------------------------------
    # Task classification
    # ---------------------------------------------------------

    def classify_task(self, prompt: str) -> str:

        text = prompt.lower()

        coding_keywords = [
            "python",
            "java",
            "c++",
            "javascript",
            "代码",
            "编程",
            "程序",
            "算法",
            "function",
            "debug",
            "bug",
            "sql",
        ]

        math_keywords = [
            "数学",
            "计算",
            "方程",
            "证明",
            "积分",
            "微分",
            "概率",
            "calculate",
            "equation",
            "prove",
        ]

        translation_keywords = [
            "翻译",
            "translate",
            "translation",
            "译成",
        ]

        if any(word in text for word in coding_keywords):
            return "coding"

        if any(word in text for word in math_keywords):
            return "math"

        if any(word in text for word in translation_keywords):
            return "translation"

        return "general"

    # ---------------------------------------------------------
    # Q-value
    # ---------------------------------------------------------

    async def get_q_value(
        self,
        task_type: str,
        model: str,
    ) -> float:

        row = await self.db._fetchone(
            """
            SELECT q_value
            FROM rsl_policy
            WHERE task_type=? AND model=?
            """,
            (task_type, model),
        )

        if row is None:
            return 0.5

        return float(row["q_value"])

    async def get_policy(self, task_type: str):

        result = {}

        for model in self.candidate_models:

            row = await self.db._fetchone(
                """
                SELECT q_value, sample_count
                FROM rsl_policy
                WHERE task_type=? AND model=?
                """,
                (task_type, model),
            )

            if row is None:

                result[model] = {
                    "q_value": 0.5,
                    "sample_count": 0,
                }

            else:

                result[model] = {
                    "q_value": float(
                        row["q_value"]
                    ),
                    "sample_count": int(
                        row["sample_count"]
                    ),
                }

        return result

    # ---------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------

    async def select_model(self, task_type: str):

        # ε-greedy
        # 20%概率随机探索
        if random.random() < self.epsilon:

            model = random.choice(
                self.candidate_models
            )

            return model, "exploration"

        scores = {}

        for model in self.candidate_models:

            scores[model] = (
                await self.get_q_value(
                    task_type,
                    model,
                )
            )

        best_score = max(scores.values())

        best_models = [
            model
            for model, score in scores.items()
            if score == best_score
        ]

        model = random.choice(best_models)

        return model, "exploitation"

    # ---------------------------------------------------------
    # Judge
    # ---------------------------------------------------------

    async def judge_answer(
        self,
        prompt: str,
        answer: str,
    ):

        judge_prompt = f"""
You are a strict AI answer evaluator.

Evaluate the answer according to:
1. correctness
2. relevance
3. completeness

User question:

{prompt}

Model answer:

{answer}

Return ONLY JSON.

Example:

{{
    "score": 0.85,
    "reason": "The answer is mostly correct."
}}

score must be between 0 and 1.
"""

        judge_request = ChatRequest(
            model=self.judge_model,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are a strict evaluator. "
                        "Return JSON only."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=judge_prompt,
                ),
            ],
            stream=False,
            temperature=0.0,
            max_tokens=200,
        )

        provider, model_config = (
            await self.router.route(
                self.judge_model
            )
        )

        response = (
            await provider.chat_completions(
                judge_request,
                model_config,
            )
        )

        text = (
            response
            .choices[0]
            .message
            .content
        )

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if not match:
            return 0.5, "Judge JSON parse failed"

        try:

            data = json.loads(
                match.group(0)
            )

            score = float(
                data.get(
                    "score",
                    0.5,
                )
            )

            score = max(
                0.0,
                min(score, 1.0),
            )

            reason = str(
                data.get(
                    "reason",
                    "",
                )
            )

            return score, reason

        except Exception:

            return (
                0.5,
                "Judge JSON parse failed",
            )

    # ---------------------------------------------------------
    # Reward
    # ---------------------------------------------------------

    def calculate_reward(
        self,
        judge_score: float,
        latency_ms: int,
    ) -> float:

        # 延迟越低，这个值越接近1
        latency_score = (
            1.0 /
            (1.0 + latency_ms / 5000.0)
        )

        # 质量占85%
        # 延迟占15%
        reward = (
            0.85 * judge_score
            +
            0.15 * latency_score
        )

        return max(
            0.0,
            min(reward, 1.0),
        )

    # ---------------------------------------------------------
    # Recursive update
    # ---------------------------------------------------------

    async def update_policy(
        self,
        task_type: str,
        model: str,
        reward: float,
    ):

        old_q = await self.get_q_value(
            task_type,
            model,
        )

        new_q = (
            (1 - self.alpha) * old_q
            +
            self.alpha * reward
        )

        await self.db._execute(
            """
            INSERT INTO rsl_policy
            (
                task_type,
                model,
                q_value,
                sample_count
            )
            VALUES (?, ?, ?, 1)

            ON CONFLICT(task_type, model)

            DO UPDATE SET

                q_value=excluded.q_value,

                sample_count=
                    rsl_policy.sample_count + 1
            """,
            (
                task_type,
                model,
                new_q,
            ),
        )

        await self.db._conn.commit()

        return old_q, new_q

    # ---------------------------------------------------------
    # Save experience
    # ---------------------------------------------------------

    async def save_experience(
        self,
        task_type: str,
        model: str,
        judge_score: float,
        latency_ms: int,
        reward: float,
        old_q: float,
        new_q: float,
        judge_reason: str,
    ):

        await self.db._execute(
            """
            INSERT INTO rsl_experiences
            (
                task_type,
                model,
                judge_score,
                latency_ms,
                reward,
                old_q,
                new_q,
                judge_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_type,
                model,
                judge_score,
                latency_ms,
                reward,
                old_q,
                new_q,
                judge_reason,
            ),
        )

        await self.db._conn.commit()